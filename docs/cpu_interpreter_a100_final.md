# 最终版本 A100 验收（2026-09-19）

> 后续封版：字符串dtype兼容修复和当前源码的针对性A100结果见[最终验收](final_gpu_freeze_20260920.md)。本页保留原阶段数据，不扩大旧全库结果的版本范围。

> 版本提示：本页记录2026-09-19受测a86bea9。2026-09-20另行合入上游#218，最新组合与新增回归见[同步说明](upstream_sync_20260920.md)；本页GPU结果不自动覆盖该同步。

最终功能提交 a86bea9d32a43e122322acd52c0168b19682ffa2 在实际 A100-SXM4-40GB 单卡完成全库验收，代码和测试未作改动。此后收尾仅更新文档，个人 master 的同文件树版本也在本验收范围内。

## 结果

```text
1206 passed, 2 skipped in 1431.71s (0:23:51)
```

使用项目原工作流的 pytest、doctest 和覆盖率参数，收集1208个测试ID。已将collected ID、逐阶段outcomes与JUnit逐项核对，全部测试有结束记录，没有失败或遗漏。1206包含CPU/编译链和GPU测试，不能称为1206个独立GPU kernel。覆盖率为全库行覆盖率约87%，不是解释器分支覆盖率。

仅两项跳过，均要求至少两张GPU：

- tests/test_aot.py::test_add[True-45327-dtype0-bf16-cuda]
- tests/test_built_artifact_reload.py::test_triton_aot_handle_is_reusable_across_cuda_contexts

另行运行正式GPU差分15/15通过，覆盖9类程序，整数/布尔精确一致，FP32使用rtol=atol=1e-3；最大观测绝对误差1.9073486328125e-6。包含原始/优化后SSA、实际Triton GPU和独立参考对照，超过文档要求的至少3个程序。另有3/3强制trace=True的矩阵乘目标通过，保护最终几何复用实现、完整轨迹、输出guard和输入不变性。这15项与全库中的对应测试重叠，不重复累计；3个跟踪目标另列。

## 环境和源码

Ubuntu22.04，Python3.12.7，Torch2.5.0+cu124，Triton3.1.0，NumPy2.1.3，SymPy1.13.1，CUDA12.4，驱动550.127.05；16核CPU、32GB内存、A100-SXM4-40GB一张。203个GitHub源blob逐项一致，包括外部验收工具共209个冻结输入在测试前后SHA不变。迁移保留编译缓存；这是正确性验收，没有新的GPU性能结论。

## 两次失败与处理

1. 镜像Torch运行版本显示带+cu124，安装元数据实际为2.5.0。首次按显示版本约束pip解析失败，改用实际元数据固定版本后依赖安装及pip check通过，未升级Torch/Triton。
2. 镜像存在64位libcuda.so.1，但缺少libcuda.so链接，Triton编译最初失败。以任务私有目录指向已有64位驱动，并设置TRITON_LIBCUDA_PATH/LIBRARY_PATH/LD_LIBRARY_PATH；未修改驱动或项目代码。初始失败日志及中止的全库测试完整保留。
3. 链接修正后的第一台A100先通过15项差分和3个跟踪目标；全库在卷积测试中出现Xid48不可纠正显存双比特错误、Xid64重映射备用行耗尽，结果为111 passed、2 skipped、1 failed、1094 errors。后续错误均含uncorrectable ECC，发生在CUDA随机数初始化；该轮保留FAIL，不计为完整通过。
4. 通过平台迁移至另一台同型号A100。其当前/累计不可纠正ECC均0、行重映射正常；30GiB两种固定模式的显存读写检查通过（仅健康自检，不是完整硬件认证），随后用完全相同源码、测试及容差完整重跑，取得上面的1206 passed。结束后ECC仍0。

## 证据与收尾

最终11份原始输出1711669字节，SHA256 eb8adb6afedcc7447b782708eb01b987514523524090eb2c889785a6089dd486；分片、gzip、逐文件SHA及测试集合在本地独立复算通过。两次失败均独立保存，未覆盖原记录。所有结果备份核验后，GPU实例在11:13:11停止。

[完整冻结源码、失败/成功输出、环境和恢复审计工具](https://github.com/a962695448-rgb/ninetoothed/tree/6ea3cb3c7b9724a45625d07e33c48540ed6c8fd2/docs/validation/a100-final-20260919)。较早835项A100记录仍属于ed33273，900项RTX4090记录仍属于1b68040，不与本次计数相加。

本次补齐最终代码的A100验收；上游PR仍须维护者评审，个人仓库主分支合并不等同于已合入InfiniTensor官方主分支。
