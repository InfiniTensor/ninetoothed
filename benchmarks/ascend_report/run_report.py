"""Run the operator and model measurements needed by the Ascend report."""

import argparse
import datetime
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

if __package__:
    from . import summarize
    from .e2e_runner import read_config, sha256_file, write_json
else:
    import summarize
    from e2e_runner import read_config, sha256_file, write_json


def script_hashes():
    return {path.name: sha256_file(path) for path in Path(__file__).parent.glob("*.py")}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def run_child(command, log_path):
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        return process.wait()


def compile_report_data(output):
    data = {"generated_at": now()}
    operator_file = output / "operator-summary.json"
    model_file = output / "e2e/report-data.json"
    if operator_file.is_file():
        data["operators"] = json.loads(operator_file.read_text(encoding="utf-8"))
    if model_file.is_file():
        data["e2e"] = json.loads(model_file.read_text(encoding="utf-8"))
    write_json(output / "report-data.json", data)
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--phase", choices=("all", "operators", "e2e", "summary"), default="all"
    )
    parser.add_argument(
        "--ops",
        nargs="+",
        help="Restrict operators for a smoke run; keep empty for the complete report.",
    )
    args = parser.parse_args(argv)
    args.config = args.config.resolve()
    config = read_config(args.config)
    raw_config = json.loads(args.config.read_text(encoding="utf-8"))
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    scripts = script_hashes()
    identity = hashlib.sha256(
        json.dumps(
            {"config": raw_config, "scripts": scripts, "ops": args.ops}, sort_keys=True
        ).encode()
    ).hexdigest()

    if args.phase == "summary":
        if (output / "operators/results.json").exists():
            summarize.aggregate(
                output / "operators",
                output,
                require_full_plan=not (
                    args.ops or raw_config.get("operators", {}).get("ops")
                ),
            )
        if (output / "e2e/environment.json").exists():
            returncode = run_child(
                [
                    sys.executable,
                    str(Path(__file__).with_name("e2e.py")),
                    "--config",
                    str(args.config),
                    "--summarize-only",
                ],
                output / "e2e-summary.log",
            )
            if returncode:
                return returncode
        compile_report_data(output)
        return 0

    manifest_path = output / "pipeline.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not args.resume or manifest["identity"] != identity:
            raise ValueError(
                "Use a new output directory or resume with unchanged configuration and scripts."
            )
    else:
        manifest = dict(
            started=now(),
            identity=identity,
            config=raw_config,
            script_sha256=scripts,
            phases={},
        )

    for name, minimum_bytes in [("cache_dir", 2 * 1024**3), ("tmp_dir", 1024**3)]:
        directory = Path(config[name])
        directory.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(directory).free < minimum_bytes:
            raise ValueError(
                f"Insufficient free space at {directory}; use a data volume for compiler caches."
            )
    write_json(manifest_path, manifest)
    failure = 0
    try:
        if args.phase in ("all", "operators"):
            command = [
                sys.executable,
                str(Path(__file__).with_name("operators.py")),
                "--config",
                str(args.config),
            ]
            if args.resume:
                command.append("--resume")
            if args.ops:
                command += ["--ops", *args.ops]
            manifest["phases"]["operators"] = {"started": now()}
            write_json(manifest_path, manifest)
            failure = run_child(command, output / "operators.log")
            manifest["phases"]["operators"].update(finished=now(), returncode=failure)
            write_json(manifest_path, manifest)
            if (output / "operators/results.json").exists():
                stats = summarize.aggregate(
                    output / "operators",
                    output,
                    require_full_plan=not (
                        args.ops or raw_config.get("operators", {}).get("ops")
                    ),
                )
                if not stats["audit_passed"]:
                    failure = failure or 1
            if failure:
                return failure

        if args.phase in ("all", "e2e"):
            command = [
                sys.executable,
                str(Path(__file__).with_name("e2e.py")),
                "--config",
                str(args.config),
            ]
            if args.resume:
                command.append("--resume")
            manifest["phases"]["e2e"] = {"started": now()}
            write_json(manifest_path, manifest)
            failure = run_child(command, output / "e2e.log")
            manifest["phases"]["e2e"].update(finished=now(), returncode=failure)

        return failure
    except Exception as error:
        failure = failure or 1
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        unchanged = script_hashes() == scripts
        manifest.update(
            finished=now(),
            scripts_unchanged=unchanged,
            status="COMPLETE" if not failure and unchanged else "INCOMPLETE",
        )
        write_json(manifest_path, manifest)
        compile_report_data(output)
        if not unchanged and not failure:
            raise RuntimeError("Measurement scripts changed during the run.")


if __name__ == "__main__":
    raise SystemExit(main())
