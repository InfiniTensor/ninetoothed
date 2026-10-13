# NineToothed 算子开发 Skill

本包用于指导 AI 编码智能体完成 NineToothed/ntops 算子实现、测试、调试、
benchmark、generated source、AOT 和 InfiniCore 集成任务。重点是提高一次完成率：
先读仓库和相邻模式，再做最小生产代码修改，用 PyTorch 或仓库参考实现补
correctness test，失败时闭环修复，最后验证 patch 可应用性。

## 使用范围

- 逐元素、广播、归约、分块、mask、dtype 和边界行为。
- stride、offset、padding、dilation 和非连续布局相关任务。
- wrapper、export、generated source、AOT、benchmark 和 dispatch。
- failing test、性能回退、补丁不可应用和环境导入问题。

不适用于无关 Python 任务、普通文案、其他仓库的一般开发，也不会默认修改
NineToothed 编译器核心。

## 使用方法

在评测环境支持的 skill 目录中放入本文件夹，或显式调用
`$ninetoothed-operator-dev-skill`。先读 `SKILL.md`，再从
`references/index.md` 选择当前任务需要的一份或少量参考资料。

## 验证

在本目录运行：

```bash
PYTHONDONTWRITEBYTECODE=1 python -B -m unittest discover -s tests -p "test_*.py" -v
PYTHONPYCACHEPREFIX="$(mktemp -d)" python -m compileall -q scripts tests
python scripts/validate_skill_package.py .
python scripts/check_no_secrets.py .
python scripts/check_false_verified_claims.py .
python scripts/check_markdown_links.py .
```

最终 ZIP 解压到全新目录后，应重复同一组命令。外置 bytecode cache 用于避免
验证过程生成的 `__pycache__` 污染 skill 目录。

## 证据边界

- 已保留 add 与 softmax 两个真实短 benchmark 的 30 次原始样本；两项在当前
  selected shape 上都显示 ntops 性能回退，不外推为普遍结论。
- layout baseline 包含 production wrapper 与 correctness test 的真实补丁，并有
  GPU 测试结果；这也是 baseline 的明确优点。
- skill-enabled layout 与性能集成补丁已在记录提交上完成本地 LF 修复和
  `git apply --check`，但历史服务器失败不会被改写为成功。
- 不声称 full non-contiguous support、AOT、generated source 或 InfiniCore
  dispatch 已验证。
- 参赛者姓名、团队、GitHub ID 和签字已按最终签名版填写；push、PR 创建和平台上传仍需本人手动执行。
