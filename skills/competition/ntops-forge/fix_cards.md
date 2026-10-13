# 失败诊断卡（Fix Cards）

`forge diagnose` 或流水线失败时，按匹配项修复。

## FC-001: Triton 交卷

**信号**: `FAIL: found Triton @triton.jit`  
**修复**: 删除 Triton，改用 `premake` + `application`；重跑 `forge run <op> --from guard`

## FC-002: TODO / 占位公式

**信号**: `application() still contains TODO`  
**修复**: 从 spec `formula` 或 `formulas.md` 填入；`preflight --strict`

## FC-003: pytest 全 skipped

**信号**: `SKIPPED: CUDA not available`  
**修复**: GPU 机 + `source .../activate base` + `doctor.py`

## FC-004: 脚本路径错误

**信号**: `can't open file '/root/scripts/...'`  
**修复**: `cd /root/work/skill` 后再跑 `forge`

## FC-005: pytest 误跑 conv2d

**信号**: `test_conv2d` 大量 FAILED  
**修复**: 只用 spec 的 `pytest: tests/test_<op>.py`，禁止 `pytest tests/` 全量

## FC-006: compare_ref 不一致

**信号**: `application() differs from reference`  
**修复**: 对照 spec `formula` 与官方 `reference` 内核，改 `application()` 一行

## FC-007: 未注册

**信号**: `not registered in __init__.py`  
**修复**: `register_op.py --name <op>` 或 `forge run --register`

## FC-008: examples 风格 make

**信号**: `kernel must use premake`  
**修复**: `--style ntops` 重新 codegen

## FC-009: 脚本版本过旧

**信号**: `unrecognized arguments: --finish`  
**修复**: 同步最新 `scripts/` 到 GPU，或改用 `forge.py run`

## FC-010: ntops 未安装

**信号**: `ModuleNotFoundError: No module named 'ntops'`  
**修复**: `cd <ntops-root> && pip install -e .`

## FC-011: gelu 参考内核无 application()

**信号**: `reference missing application()`（旧版 compare_ref）  
**修复**: 已支持 `default_application`；更新 `compare_ref.py` 后重跑 `forge gate`

## FC-012: pytest 未安装

**信号**: `No module named pytest`（PROVE 阶段）  
**修复**: `pip install pytest`；建议在 ntops 根目录 `pip install -e ".[dev]"` 或 `pip install -e .` 后重跑 `doctor.py` 与 `forge gate`
