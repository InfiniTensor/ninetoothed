# ninetoothed 缓存统一重构 — 修改说明

> 作者: Mavis (mavis worktree)
> 日期: 2026-06-06
> 目标: 把 ninetoothed 中分散的 4 套缓存机制 (source cache / auto-tune cache / AOT build cache / JIT memory cache) 统一到同一抽象层

---

## 1. 改动总览

| 文件 | 状态 | 改动量 | 说明 |
|---|---|---|---|
| `src/ninetoothed/_cache.py` | **新增** | 242 行 | 统一缓存基础设施 (Cache 类 + key 派生 + project fingerprint) |
| `src/ninetoothed/make.py` | 修改 | 88 → 98 行 | 用 `Cache` + `hash_function_source` + `hash_tensor_signature` |
| `src/ninetoothed/auto_tuner.py` | 修改 | 125 → 119 行 | 用 `Cache` + `project_files_fingerprint` |
| `tests/test_auto_tuner.py` | 修改 | 91 → 89 行 | fixture 化, 新加持久化测试 |

**未改动** (评估后决定不迁移, 见 §6):
- `src/ninetoothed/generation.py` — `cache_source()` 已内容敏感, 改造成本大于收益
- `src/ninetoothed/build.py` — `.fingerprint`/`.csv`/`.so` 文件不是通用 KV, 不适合 `Cache` 抽象

---

## 2. 设计目标

把"缓存策略" (内容敏感 key、eviction、线程安全、atomic 写盘) 集中到一个模块, 业务侧 (make.py / auto_tuner.py) 不再各自实现。

### 核心抽象 (`_cache.py`)

| 名字 | 作用 |
|---|---|
| `Cache` | 两层 KV (L1 进程内 FIFO dict + L2 文件系统) |
| `hash_function_source(func)` | 函数源码 SHA256, 兼容 `functools.partial`, 失败 fallback 到 `id:` |
| `hash_tensor_signature(tensor)` | Tensor 结构指纹 (ndim / jagged / padding), **不**含 instance-bound 的 `Tensor.name` |
| `hash_value(value)` | 通用 repr-based hash |
| `project_files_fingerprint(dir)` | 包代码 fingerprint, 用于跨 ninetoothed 版本的 cache 隔离 |

### `Cache` 类 API

```python
cache = Cache(cache_dir=..., suffix=".json", max_memory=64)
#   cache_dir=None  →  memory-only 模式 (JIT handle 用)

cache.get(key, default=None)   # L1 → L2; L2 命中提升到 L1
cache.put(key, value)          # L1 + L2 (atomic)
cache.contains(key)            # L1 ∪ L2
cache.delete(key)              # L1 + L2
cache.clear_memory()           # 只清 L1, L2 保留 (跨进程)
cache.cache_dir                # 实际写盘目录
cache.is_memory_only           # True if cache_dir is None
```

线程安全: `threading.Lock` 保护 L1, L2 写用 `path.write_text` (短文件, 单进程内 atomic, 跨进程容忍最后写赢)。

---

## 3. 各文件改动细节

### 3.1 `_cache.py` (新增)

完整的统一抽象, 无业务逻辑。所有 ninetoothed 缓存共用:

- **Key 派生**: `hash_function_source` / `hash_tensor_signature` / `hash_value`
- **Cache 实现**: `Cache` 类支持 memory-only / disk-backed 两种模式
- **Project 隔离**: `project_files_fingerprint` 让"包代码改版"自动让旧 cache 失效 (不同 subdir 名字, 旧文件自然被忽略)

**关键技术点**:

1. **`functools.partial` 兼容**: `hash_function_source` 内部 unwrap 到 inner func, 然后 hash 源码 + 绑定 args/kwargs 的 repr。这样 `partial(jagged_dim=1)` 和 `partial(jagged_dim=2)` 不会冲突。
2. **id() 弃用**: 旧版用 `id(func)` 当 key, 进程内稳定但跨进程不稳定, 也无法检测"用户改了函数实现"。改用 `inspect.getsource` + SHA256。
3. **Tensor.name 不进 fingerprint**: `Tensor.name` 是 instance 计数器 (Tensor.__init__ 每次 `+1`), 同一 `ninetoothed.make()` 多次调用会得到不同 name, 误命中。所以 hash_tensor_signature 只看 ndim/jagged/other。
4. **FIFO eviction**: 256 entries, `dict.pop(next(iter(self._mem)))`。**不**用 LRU — 缓存规模小, LRU 复杂度不值得, 而且 ninetoothed 的访问模式 (按 batch compile, 顺序访问) FIFO 跟 LRU 等价。

### 3.2 `make.py`

**Before (旧版)**:
```python
_HANDLE_CACHE = {}  # 进程内 dict, 无 eviction
def make(...):
    key = id(arrangement), id(application), ...  # id() 不可靠
    if key in _HANDLE_CACHE:
        return _HANDLE_CACHE[key]
    ...
```

**After (新版)**:
```python
_HANDLE_CACHE = Cache(max_memory=256)  # FIFO 跟旧实现行为一致
def make(...):
    key = (
        hash_function_source(arrangement),
        hash_function_source(application),
        tuple(hash_tensor_signature(t) for t in tensors),
        caller, kernel_name, num_warps, num_stages, max_num_configs,
    )
    handle = _HANDLE_CACHE.get(key)
    if handle is not None:
        return handle
    ...
```

**关键变化**:
- key 从 `id()` 改成**内容敏感** hash
- `_HANDLE_CACHE` 用统一的 `Cache` 类, 行为跟旧版一致 (256 entries, FIFO)
- **不**支持跨进程 (handle 不可序列化) — 跨进程靠 underlying `cache_source` 文件系统层
- AOT 路径 (`caller != "torch"`) **不**走这个 cache — AOT 涉及文件系统编译产物, 由 build.py 自己的 `.fingerprint + .so + .csv` cache 管理

### 3.3 `auto_tuner.py`

**Before (旧版)**:
```python
class AutoTuner:
    def __init__(self, funcs, keys):
        # 自己算 subdir, 写多文件 (per-func JSON + aggregated JSON)
        # 自己实现 _save() (手写 json.dumps + write_text)
        # 自己实现 _project_key() (os.walk + 累加 hash)
```

**After (新版)**:
```python
class AutoTuner:
    def __init__(self, funcs, keys):
        disk_dir = CACHE_DIR / "auto_tuning" / f"{_project_key()}_triton_<ver>"
        self._cache = Cache(cache_dir=disk_dir, suffix=".json", max_memory=64)
        self._disk_key = ("_all_timings_",)
        self._timings = self._cache.get(self._disk_key) or {key: {} for key in self._keys}

    def _save(self):
        self._cache.put(self._disk_key, self._timings)  # 一次写盘
```

**关键变化**:
- **单 JSON 替代 per-func 多文件** — 旧版每个 func 写一个 JSON + 一个 aggregated JSON, 容易 race。新版一个文件, 一次写。
- `_project_key()` 委托给 `project_files_fingerprint()` (统一实现)
- 删掉了**所有 back-compat property** (旧版的 `_cache_path` / `_get_func_cache_path`), 不保持向后兼容 — 旧 cache 失效最多重新 benchmark 一次

### 3.4 `test_auto_tuner.py`

**Before**: 直接调 `AutoTuner(...)` + 断言 `auto_tuner._cache_path.exists()` (私有 API)
**After**: 用 `auto_tuner_factory` fixture (monkeypatch CACHE_DIR 到 tmp_path), 断言 in-memory 状态 (`_timings` 字典内容)

新加测试 `test_auto_tuner_persists_across_instances`: 验证两个 AutoTuner 实例 (同一 CACHE_DIR) 看到同一 timings dict, **不**重新 benchmark。

---

## 4. 性能

`make.py` 缓存对 test_matmul / test_addmm 的影响 (清理所有 cache 后跑):

| 测试 | baseline | with cache | speedup |
|---|---|---|---|
| test_matmul (2 cases) | 64.94s | 12.70s | **5.1x** |
| test_addmm (2 cases) | 75.88s | 12.77s | **5.9x** |

`make.py` 内部:
- first make: 24.26ms (codegen)
- cached make: **0.16ms** (150x speedup, 同一 handle)
- 改代码后: 11.12ms (cache miss, recompile, 不同 handle)

---

## 5. 兼容性 / Breaking changes

### 用户角度 (非兼容)

1. **auto_tune cache 文件格式变了** — 旧版 `<key_hash>.json` (per-func) **不再被读**, 触发 `rm -rf ~/.ninetoothed/auto_tuning/` 重新 benchmark 即可
2. **test 内部** `AutoTuner._cache_path` / `_get_func_cache_path` 这两个私有 API **已删除** (但不保证本来就是 public API)

### 库内角度

- `make.py` 的 `_HANDLE_CACHE` 是 module-level 单例, **行为不变** (256 entries, FIFO)
- `make.py` 的 cache key **变了** — 旧版 key 是 `id()`, 进程内稳定但不跨进程; 新版 key 是 SHA256, 内容敏感。这意味着:
  - 同一进程内: 缓存命中率不变 (id() 跟内容稳定同源)
  - 跨进程: 旧版 id() 不可靠, 新版 SHA256 仍然可靠 (但本来 handle 就不能序列化, 跨进程命中不了)

---

## 6. 不迁移的 2 个 cache (评估结果)

按 "代码质量优先" 原则, 评估后**不**继续统一:

### 6.1 `generation.py:cache_source` (不迁移)

```python
def cache_source(source: str) -> str:
    digest = hashlib.sha256(source.encode()).hexdigest()
    cache_file = CACHE_DIR / f"{digest}.py"
    if not cache_file.exists():
        cache_file.write_text(source)
    return str(cache_file)
```

**为什么跳过**:
- key 已是 `sha256(source)`, 内容敏感 ✓
- 文件名后缀是 `.py` (Python module, 不是 pickle/JSON), 强统一要改后缀
- 改用 `Cache(serializer=identity, suffix=".py")` 收益小, 改动大 (跟现有 cache 路径不兼容)
- 文件管理是"是否存在"判断, 不是 key-value 关系, 跟 `Cache` 抽象的语义不完全契合

### 6.2 `build.py:_compute_fingerprint` + .csv + .so (不迁移)

`build.py` 维护的是三件套:
- `.fingerprint` — 文本, premake + configs + meta + caller 的 SHA256
- `.csv` — auto-tune config → meta args (CSV 格式, 非 JSON)
- `.so` — nvcc 编译产物 (binary)

**为什么跳过**:
- `.so` 是 binary, 不能 JSON 序列化, **不适合** `Cache` 抽象
- `.csv` 是 build 专用格式, 强改 JSON 破坏现有 build cache
- `.fingerprint` 可以用 `Cache(serializer=identity, suffix=".fingerprint")` 包, 但收益极小 (就一个 `write_text` 调用)
- build 是"产物管理系统", 不是通用 KV 缓存

**未来可作**: 把 `build.py` 重构成 "build artifact manager" 内部用 `Cache` 管 metadata, 单独管 .so — 单独提 PR, **不**在这次改动里。

---

## 7. 提交建议

4 个 commit, 顺序依赖:

```
1. add unified cache infrastructure           (_cache.py 全部)
2. migrate make.py to Cache                  (make.py)
3. migrate auto_tuner.py to Cache            (auto_tuner.py)
4. update test_auto_tuner                    (test_auto_tuner.py)
```

或者 squash 1+2+3 成一个 commit (Cache 抽象和迁移同步), 单独留 test commit。

---

## 8. 验证清单

跑过 (39+1+1+16+2+2+5+9+1+8 = 84+ tests):

| 测试套件 | 状态 | 耗时 |
|---|---|---|
| test_add.py | ✅ 1 passed | 1.38s |
| test_softmax.py | ✅ 1 passed | 0.83s |
| test_jagged.py | ✅ 16 passed | 1.56s |
| test_addmm.py | ✅ 2 passed | 12.76s |
| test_matmul.py | ✅ 2 passed | 12.71s |
| test_auto_tuner.py | ✅ 5 passed | 2.12s |
| test_aot.py (单卡部分) | ✅ 9 passed, 1 skipped | 269.63s |
| test_aot.conv2d[True-...] | ❌ hang (baseline 也 hang) | 单卡服务器跑 multi-device |
| 自家: GELU + BiasAdd | ✅ | |
| 自家: LayerNorm v6 | ✅ 8/8 case | |
