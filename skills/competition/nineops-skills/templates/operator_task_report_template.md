# 算子任务报告模板

> 本文档由 Agent 完成实现后填写，用于记录任务过程、结果和反思。

## 基本信息

| 字段 | 值 |
|------|-----|
| 算子 | <!-- 算子名称 --> |
| DSL 模式 | <!-- 如 elementwise 1D / reduction 2D / matmul 2D --> |
| 输入 shape | <!-- 如 (1024,), (768, 768) --> |
| 输入 dtype | <!-- 如 float32, float16 --> |
| BLOCK_SIZE | <!-- 如 1024, 512 --> |
| 实现日期 | <!-- YYYY-MM-DD --> |

## 实现过程

### Step 1: 分析算子

<!-- 描述算子的计算逻辑、输入/输出关系、是否存在广播、归约操作 -->

### Step 2: 选择 DSL 模式

<!-- 参照 dsl_patterns.md，选择对应的 arrangement 模板 -->

### Step 3: 编写 Implementation

**Arrangement 设计:**

```python
# 贴出 arrangement 代码
```

**Application 设计:**

```python
# 贴出 application 代码
```

### Step 4: Torch 包装

```python
# 贴出 warp 代码
```

## Correctness 测试结果

| 测试场景 | Shape | Dtype | Broadcast | Contiguous | 结果 |
|----------|-------|-------|-----------|------------|------|
| 基础 | | | N/A | ✅ | ✅/❌ |
| 广播 | | | ✅ | ✅ | ✅/❌ |
| 非连续 | | | N/A | ❌ (transpose) | ✅/❌ |

## Benchmark 结果

| Shape | Dtype | BlockSize | 本实现(ms) | PyTorch(ms) | Speedup |
|-------|-------|-----------|------------|-------------|---------|
| | | | | | |

## Generated Source 检查

<!-- 关键发现，如果有问题则记录 -->

## 遇到的问题

1. <!-- 问题描述 --> → <!-- 解决方案 -->

## 反思

<!-- agent 自我反思：本次实现中哪些做法好，哪些可以改进 -->
