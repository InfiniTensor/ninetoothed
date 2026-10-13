import os
import re
import json
import stat
import base64
import tarfile
import tempfile
import datetime
import paramiko
from contextlib import contextmanager

from mcp.server.fastmcp import FastMCP

# ── 加载配置 ───────────────────────────────────────────
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
with open(CONFIG_PATH) as f:
    C = json.load(f)

HOST   = C["host"]
PORT   = int(C.get("port", 22))
USER   = C["user"]
KEY    = C.get("key_filename")
PASS   = C.get("password")
REMOTE = C["remote_root"].rstrip("/")

# 默认环境变量前缀（MetaX C500 必需）
# 可通过 config.json 的 "default_env" 字段覆盖
DEFAULT_ENV = C.get("default_env", (
    "export MACA_PATH=/opt/maca"
    " && export MACA_CLANG_PATH=/opt/maca/mxgpu_llvm/bin"
    " && export PATH=/opt/conda/bin:/opt/maca/mxgpu_llvm/bin:/opt/maca/ompi/bin:/opt/maca/ucx/bin:/opt/mxdriver/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    " && export LD_LIBRARY_PATH=/opt/maca/lib:/opt/maca/ompi/lib:/opt/maca/ucx/lib:/opt/mxdriver/lib"
    " && export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1"
))

mcp = FastMCP("remote-gpu-server")


# ── SSH 连接封装 ──────────────────────────────────────
def ssh_connect():
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    transport_params = {
        "disabled_algorithms": {
            "pubkeys": ["rsa-sha2-256", "rsa-sha2-512"]
        }
    }
    try:
        if KEY:
            cli.connect(
                HOST, PORT, USER, key_filename=KEY,
                timeout=15, allow_agent=False, look_for_keys=False,
                **transport_params
            )
        else:
            cli.connect(
                HOST, PORT, USER, password=PASS,
                timeout=15, allow_agent=False, look_for_keys=False,
                **transport_params
            )
        return cli
    except Exception as e:
        raise ConnectionError(f"SSH 连接失败 ({HOST}:{PORT}): {e}") from e


def _sftp_mkdir_p(sftp, remote_dir):
    if not remote_dir or remote_dir == "/":
        return
    dirs = []
    current = remote_dir.rstrip("/")
    while current and current != "/":
        try:
            sftp.stat(current)
            break
        except FileNotFoundError:
            dirs.append(current)
            current = os.path.dirname(current)
    for d in reversed(dirs):
        sftp.mkdir(d)


@contextmanager
def _ssh_session():
    cli = ssh_connect()
    try:
        yield cli
    finally:
        cli.close()


def _exec(sftp_or_cli, cmd, timeout=60):
    """在已有连接上执行命令，返回 (stdout, stderr)"""
    cli = sftp_or_cli if isinstance(sftp_or_cli, paramiko.SSHClient) else None
    if cli is None:
        cli = sftp_or_cli._client if hasattr(sftp_or_cli, '_client') else None
    # fallback: 用传入对象本身的 exec_command（SSHClient）
    if hasattr(sftp_or_cli, 'exec_command'):
        _, stdout, stderr = sftp_or_cli.exec_command(cmd, timeout=timeout)
        return stdout.read().decode(), stderr.read().decode()
    return "", ""


# ── Helpers ───────────────────────────────────────────
def _is_python_cmd(cmd: str) -> bool:
    return bool(re.search(r'(?:^|\s)(?:[\w/]*python3?[\w.]*)\s', cmd.strip()))


def _extract_script_name(cmd: str) -> str:
    parts = cmd.strip().split()
    for p in parts[1:]:
        if p.endswith(".py"):
            return os.path.splitext(os.path.basename(p))[0]
    return "python_cmd"


def _resolve_remote(workdir: str | None) -> str:
    """将 workdir 解析为远端绝对路径"""
    if not workdir:
        return REMOTE
    if workdir.startswith("/"):
        return workdir.rstrip("/")
    return f"{REMOTE}/{workdir.strip('/')}"


# ───────────────────────────────────────────────────────
# MCP Tools
# ───────────────────────────────────────────────────────

@mcp.tool()
def ping() -> str:
    """测试 MCP 是否正常工作"""
    return "pong"


@mcp.tool()
def upload_code(src: str, dst: str) -> str:
    """
    上传本地文件或目录到服务器（目录自动 tar.gz 压缩后上传）

    Args:
        src: 本地路径 (如 C:\\workspace\\flash_attention 或单个 .py 文件)
        dst: 远端相对路径 (如 flash_attention/ 或 flash_attention/kernel.py)
    """
    remote_path = f"{REMOTE}/{dst.strip('/')}"

    with _ssh_session() as cli:
        sftp = cli.open_sftp()
        try:
            if os.path.isfile(src):
                parent = os.path.dirname(remote_path)
                if parent:
                    _sftp_mkdir_p(sftp, parent)
                sftp.put(src, remote_path)
                return f"✅ Uploaded file -> {remote_path}"

            # 目录上传：打包目录内容（不含目录壳），直接铺到 dst
            with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                # 打包：arcname="." 使解压后内容直接铺到目标目录
                with tarfile.open(tmp_path, "w:gz") as tar:
                    tar.add(src, arcname=".")
                size_kb = os.path.getsize(tmp_path) / 1024

                _sftp_mkdir_p(sftp, remote_path)
                sftp.put(tmp_path, remote_path + ".tar.gz")

                _, stdout, stderr = cli.exec_command(
                    f"tar -xzf {remote_path}.tar.gz -C {remote_path} && rm {remote_path}.tar.gz",
                    timeout=60
                )
                err = stderr.read().decode().strip()
                if err:
                    return f"❌ Remote extract failed: {err}"

                # 列出上传的文件供确认
                _, stdout2, _ = cli.exec_command(f"ls -1 {remote_path}", timeout=10)
                files = stdout2.read().decode().strip().split("\n")
                file_list = ", ".join(files[:10])
                suffix = f" ... (+{len(files)-10} more)" if len(files) > 10 else ""
                return f"✅ Uploaded dir ({size_kb:.0f} KB, {len(files)} files) -> {remote_path}\n   [{file_list}{suffix}]"
            finally:
                os.unlink(tmp_path)
        finally:
            sftp.close()


@mcp.tool()
def download_file(remote_path: str, local_path: str) -> str:
    """
    从服务器下载文件或目录到本地（目录自动 tar.gz 压缩后下载）

    Args:
        remote_path: 远端相对路径 (如 results/output.txt 或 results/)
        local_path: 本地保存路径 (如 C:\\workspace\\results\\output.txt)
    """
    full_remote = f"{REMOTE}/{remote_path.strip('/')}"

    with _ssh_session() as cli:
        sftp = cli.open_sftp()
        try:
            try:
                attr = sftp.stat(full_remote)
            except FileNotFoundError:
                return f"❌ Remote path not found: {full_remote}"

            if stat.S_ISDIR(attr.st_mode):
                with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
                    tmp_path = tmp.name
                try:
                    tar_remote = f"{full_remote}.tar.gz"
                    _, stdout, stderr = cli.exec_command(
                        f"tar -czf {tar_remote} -C {os.path.dirname(full_remote)} {os.path.basename(full_remote)}",
                        timeout=120
                    )
                    err = stderr.read().decode().strip()
                    if err:
                        return f"❌ Remote archive failed: {err}"

                    sftp.get(tar_remote, tmp_path)
                    cli.exec_command(f"rm {tar_remote}", timeout=10)

                    os.makedirs(local_path, exist_ok=True)
                    with tarfile.open(tmp_path, "r:gz") as tar:
                        tar.extractall(path=local_path)
                    size_kb = os.path.getsize(tmp_path) / 1024

                    entries = os.listdir(local_path)
                    basename = os.path.basename(full_remote)
                    if len(entries) == 1 and entries[0] == basename:
                        inner = os.path.join(local_path, basename)
                        for item in os.listdir(inner):
                            os.rename(os.path.join(inner, item), os.path.join(local_path, item))
                        os.rmdir(inner)

                    return f"✅ Downloaded dir ({size_kb:.0f} KB tar.gz) -> {local_path}"
                finally:
                    os.unlink(tmp_path)
            else:
                parent = os.path.dirname(local_path)
                if parent:
                    os.makedirs(parent, exist_ok=True)
                sftp.get(full_remote, local_path)
                return f"✅ Downloaded file -> {local_path}"
        finally:
            sftp.close()


@mcp.tool()
def run_test(cmd: str, timeout: int = 300, workdir: str = "", env: str = "default") -> str:
    """
    在服务器上运行测试命令。Python 命令自动生成带时间戳的日志文件。

    Args:
        cmd:     要执行的命令 (如 python test_matmul.py)
        timeout: 超时秒数，默认 300
        workdir: 工作目录，相对于 remote_root。如 "05_diag/" 等价于 /data/05_diag。
                 空字符串表示 remote_root 本身。
        env:     环境变量模式:
                 "default" — 自动注入 MACA_PATH/LD_LIBRARY_PATH 等 MetaX 必需环境变量
                 "none"    — 不注入任何环境变量（纯 bash 环境）
                 自定义字符串 — 作为命令前缀执行（如 "export FOO=bar && "）
    """
    cwd = _resolve_remote(workdir)

    if env == "default":
        env_prefix = DEFAULT_ENV + " && "
    elif env == "none":
        env_prefix = ""
    else:
        env_prefix = env.rstrip(" &") + " && "

    with _ssh_session() as cli:
        if _is_python_cmd(cmd):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            script_name = _extract_script_name(cmd)
            log_file = f"{REMOTE}/logs/{script_name}_{timestamp}.log"
            full_cmd = (
                f"mkdir -p {REMOTE}/logs && "
                f"cd {cwd} && "
                f"{env_prefix}"
                f"{{ {cmd}; }} 2>&1 | tee {log_file}"
            )
        else:
            log_file = None
            full_cmd = f"cd {cwd} && {env_prefix}{cmd}"

        stdin, stdout, stderr = cli.exec_command(full_cmd, timeout=timeout)
        out = stdout.read().decode()
        err = stderr.read().decode()

        lines = ["STDOUT:", out.rstrip()]
        if err.strip():
            lines += ["STDERR:", err.rstrip()]
        if log_file:
            lines += ["", f"[Log saved to server: {log_file}]"]
        return "\n".join(lines)


@mcp.tool()
def remote_ls(path: str = "") -> str:
    """
    列出远端目录内容（文件名、大小、修改时间）

    Args:
        path: 目录路径，相对于 remote_root。空字符串列出 remote_root 本身。
    """
    target = _resolve_remote(path) if path else REMOTE
    with _ssh_session() as cli:
        _, stdout, stderr = cli.exec_command(
            f"ls -lh --time-style=long-iso {target}", timeout=10
        )
        out = stdout.read().decode().strip()
        err = stderr.read().decode().strip()
        if err:
            return f"❌ {err}"
        if not out:
            return f"(empty directory: {target})"
        lines = out.split("\n")
        # 跳过 total 行
        if lines and lines[0].startswith("total"):
            lines = lines[1:]
        return f"📁 {target} ({len(lines)} items):\n" + "\n".join(lines)


@mcp.tool()
def remote_cat(path: str, lines: int = 0) -> str:
    """
    读取远端文件内容

    Args:
        path:  文件路径，相对于 remote_root
        lines: 读取行数，0 表示全部内容（默认），正整数表示前 N 行
    """
    target = _resolve_remote(path)
    with _ssh_session() as cli:
        if lines > 0:
            cmd = f"head -n {lines} {target}"
        else:
            cmd = f"cat {target}"
        _, stdout, stderr = cli.exec_command(cmd, timeout=10)
        out = stdout.read().decode()
        err = stderr.read().decode().strip()
        if err:
            return f"❌ {err}"
        return out


@mcp.tool()
def remote_md5(path: str) -> str:
    """
    计算远端文件 MD5，用于确认文件版本是否最新

    Args:
        path: 文件路径，相对于 remote_root
    """
    target = _resolve_remote(path)
    with _ssh_session() as cli:
        _, stdout, stderr = cli.exec_command(f"md5sum {target}", timeout=10)
        out = stdout.read().decode().strip()
        err = stderr.read().decode().strip()
        if err:
            return f"❌ {err}"
        return out


@mcp.tool()
def mx_smi(options: str = "") -> str:
    """
    查看沐曦 GPU 状态 (MetaX GPU)

    Args:
        options: mx-smi 的参数，例如：
                 ""        - 查看概览
                 "-l 1000" - 每秒刷新
                 "-i 0"    - 查看 GPU 0
                 "--show-usage" - 查看利用率
    """
    cmd = f"mx-smi {options}".strip()
    with _ssh_session() as cli:
        stdin, stdout, stderr = cli.exec_command(cmd, timeout=15)
        out = stdout.read().decode().strip()
        err = stderr.read().decode().strip()
        if out:
            return out
        if err:
            return f"[mx-smi error] {err}"
        return "GPU not available"


@mcp.tool()
def remote_rm(path: str, recursive: bool = False) -> str:
    """
    删除远端文件或目录。安全保护：禁止删除 remote_root 本身及系统路径。

    Args:
        path:      相对于 remote_root 的路径 (如 "05_diag" 或 "05_diag/stale.py")
        recursive: 若为目录，是否递归删除。默认 False，目录需显式开启。
    """
    target = _resolve_remote(path)

    # 安全防护：禁止删除 remote_root 本身、"/"、"/opt"、"/data" 等危险路径
    protected = {"", "/", REMOTE, "/data", "/opt", "/tmp", "/home", "/root"}
    if target.rstrip("/") in protected or target == REMOTE:
        return f"❌ Refused: protected path '{target}' cannot be deleted"

    with _ssh_session() as cli:
        # 先 stat 判断是否存在 + 类型
        _, stdout, stderr = cli.exec_command(
            f"if [ -e {target} ]; then "
            f"if [ -d {target} ]; then echo 'dir'; else echo 'file'; fi; "
            f"else echo 'missing'; fi",
            timeout=10,
        )
        kind = stdout.read().decode().strip()

        if kind == "missing":
            return f"⚠ {target} does not exist (nothing to do)"

        if kind == "dir":
            if not recursive:
                # 安全起见：列出目录内容，要求用户显式开启 recursive
                _, ls_out, _ = cli.exec_command(
                    f"ls -1 {target} | head -20", timeout=10
                )
                listing = ls_out.read().decode().strip().split("\n")
                preview = ", ".join(listing[:10])
                suffix = f" ... (+{len(listing)-10} more)" if len(listing) > 10 else ""
                return (
                    f"⚠ {target} is a directory with {len(listing)} items.\n"
                    f"   Preview: [{preview}{suffix}]\n"
                    f"   Re-run with recursive=True to delete."
                )
            rm_cmd = f"rm -rf {target}"
        else:
            rm_cmd = f"rm -f {target}"

        _, stdout2, stderr2 = cli.exec_command(rm_cmd, timeout=60)
        out = stdout2.read().decode().strip()
        err = stderr2.read().decode().strip()
        if err:
            return f"❌ rm failed: {err}"
        return f"✅ Deleted ({kind}) -> {target}"


@mcp.tool()
def remote_glob(pattern: str, path: str = "") -> str:
    """
    在远端目录中按 glob 模式查找文件（支持 ** 递归）

    Args:
        pattern: glob 模式 (如 "*.py", "*.log", "diag_*.py", "**/*.py")
        path:    搜索的根目录，相对于 remote_root。空字符串 = remote_root。
    """
    root = _resolve_remote(path) if path else REMOTE
    # 用 find + -name 实现 glob；** 用 -name 即可（find 默认递归）
    # 安全转义: 只允许 [A-Za-z0-9_*?.-] 字符
    safe = re.sub(r"[^A-Za-z0-9_\*\?\.\-]", "", pattern)
    if safe != pattern:
        return f"❌ Unsafe pattern '{pattern}' (allowed: letters, digits, _ * ? . -)"

    cmd = f"find {root} -name '{safe}' -type f"
    with _ssh_session() as cli:
        _, stdout, stderr = cli.exec_command(cmd, timeout=30)
        out = stdout.read().decode().strip()
        err = stderr.read().decode().strip()
        if err and not out:
            return f"❌ {err}"
        if not out:
            return f"(no matches for '{pattern}' under {root})"
        matches = out.split("\n")
        # 截掉 REMOTE 前缀，显示相对路径
        rel = [m[len(REMOTE):].lstrip("/") if m.startswith(REMOTE) else m for m in matches]
        header = f"🔍 {len(rel)} matches for '{pattern}' under {path or '.'}:"
        # 最多显示 100 条
        shown = rel[:100]
        suffix = f" ... (+{len(rel)-100} more)" if len(rel) > 100 else ""
        return header + "\n" + "\n".join(shown) + suffix


@mcp.tool()
def run_python(code: str, workdir: str = "", timeout: int = 120) -> str:
    """
    在服务器上运行一段内联 Python 代码，无需本地文件或上传。
    适合快速探针：版本检查、cache 清理、单行 benchmark 等。

    Args:
        code:     Python 代码 (多行用 \\n 分隔)
        workdir:  工作目录，相对于 remote_root
        timeout:  超时秒数，默认 120

    注意：自动注入 MetaX 必需环境变量 (同 run_test env="default")。
    """
    cwd = _resolve_remote(workdir)
    # 用 python -c 执行，对 code 做 shell 转义
    # 简单做法：把 code base64 编码后 decode 执行，避免转义地狱
    code_b64 = base64.b64encode(code.encode("utf-8")).decode("ascii")
    runner = (
        "import base64, sys; "
        "exec(base64.b64decode(sys.argv[1]).decode('utf-8'))"
    )
    # 防 shell 注入：code_b64 仅含 [A-Za-z0-9+/=]，安全
    full_cmd = (
        f"cd {cwd} && "
        f"{DEFAULT_ENV} && "
        f"python -c \"{runner}\" {code_b64}"
    )
    with _ssh_session() as cli:
        stdin, stdout, stderr = cli.exec_command(full_cmd, timeout=timeout)
        out = stdout.read().decode().rstrip()
        err = stderr.read().decode().rstrip()
        lines = []
        if out:
            lines += ["STDOUT:", out]
        if err:
            lines += ["STDERR:", err]
        if not lines:
            return "(no output)"
        return "\n".join(lines)


@mcp.tool()
def auto_bench(
    v0: str,
    v1: str,
    workdir: str = "",
    atol: float = 1e-2,
    repeat: int = 500,
    timeout: int = 600,
    runs: int = 1,
) -> str:
    """
    一键运行 auto_bench.py 对比 v0/v1 算子。自动注入环境变量，解析 speedup。

    Args:
        v0:      参考实现文件名 (如 05TriAttentionFallback.py)
        v1:      Triton 实现文件名 (如 05TriAttentionFallback_MateX.py)
        workdir: 文件所在目录 (如 "05_diag")，相对于 remote_root
        atol:    绝对误差容限，默认 1e-2
        repeat:  单次 auto_bench 的迭代次数，默认 500
        timeout: 单次超时秒数，默认 600
        runs:    重复运行次数 (默认 1)，>1 时输出 median/min/max 统计

    返回：每次运行的 PASS/FAIL + speedup，多跑时附加 median/min/max 统计。
    """
    single_cmd = (
        f"python auto_bench.py"
        f" --v0_file {v0} --v1_file {v1}"
        f" --atol {atol} --repeat {repeat}"
    )

    results = []
    per_run_lines = []
    for i in range(runs):
        out = run_test(cmd=single_cmd, timeout=timeout, workdir=workdir, env="default")
        # 解析 speedup 和 PASS/FAIL
        m_sp = re.search(r"speedup=([\d.]+)x", out)
        m_v0 = re.search(r"v0=([\d.]+)\s*ms", out)
        m_v1 = re.search(r"v1=([\d.]+)\s*ms", out)
        passed = "PASS accuracy" in out or "1 passed" in out
        speedup = float(m_sp.group(1)) if m_sp else None
        v0_ms = float(m_v0.group(1)) if m_v0 else None
        v1_ms = float(m_v1.group(1)) if m_v1 else None

        per_run_lines.append(
            f"Run {i+1}/{runs}: {'PASS' if passed else 'FAIL'}  "
            f"v0={v0_ms:.4f}ms  v1={v1_ms:.4f}ms  speedup={speedup:.3f}x"
            if speedup is not None else
            f"Run {i+1}/{runs}: {'PASS' if passed else 'FAIL'}  (parse failed)\n{out}"
        )
        if speedup is not None:
            results.append((passed, v0_ms, v1_ms, speedup))

    if runs == 1:
        # 单次：保持原有输出格式
        return out

    # 多次：输出表格 + 聚合统计
    header = f"auto_bench × {runs} runs ({v0} vs {v1}, workdir={workdir or '.'})"
    body = "\n".join(per_run_lines)

    if results:
        speedups = [r[3] for r in results]
        v0_times = [r[1] for r in results]
        v1_times = [r[2] for r in results]
        n_pass = sum(1 for r in results if r[0])

        speedups_sorted = sorted(speedups)
        median_sp = speedups_sorted[len(speedups_sorted) // 2]
        min_sp = speedups_sorted[0]
        max_sp = speedups_sorted[-1]
        median_v1 = sorted(v1_times)[len(v1_times) // 2]

        stats = (
            f"\n\n📊 Aggregate ({n_pass}/{runs} PASS):\n"
            f"  speedup: median={median_sp:.3f}x  min={min_sp:.3f}x  max={max_sp:.3f}x\n"
            f"  v1 (ms): median={median_v1:.4f}  min={min(v1_times):.4f}  max={max(v1_times):.4f}\n"
            f"  v0 (ms): median={sorted(v0_times)[len(v0_times)//2]:.4f}  "
            f"min={min(v0_times):.4f}  max={max(v0_times):.4f}"
        )
    else:
        stats = "\n\n⚠ No parseable speedup from any run"

    return f"{header}\n{'-'*len(header)}\n{body}{stats}"


# ── MCP 启动入口 ──────────────────────────────────────
if __name__ == "__main__":
    mcp.run()
