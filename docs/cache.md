# ninetoothed 缓存架构 (Cache Architecture)

> 适用版本: ninetoothed 0.25.0+
> 本章描述: 整个 ninetoothed 系统的缓存机制, 以及如何在写新算子 / 编译器 pass 时正确接入

## 1. 总览

ninetoothed 的缓存**分四层**, 由统一的 `ninetoothed._cache` 模块抽象:

| 层 | 模块 | 内容 | 跨进程 |
|---|---|---|---|
| L1 (in-process) | `ninetoothed._cache.Cache` (memory-only mode) | JIT kernel handle | ✗ |
| L2 (filesystem) | `ninetoothed._cache.Cache` (disk-backed mode) | premake `.py`, auto-tune JSON, AOT `.fingerprint` | ✓ |
| AOT artifacts | `ninetoothed.build` (直接文件管理) | `.so`, `.csv`, `.fingerprint` | ✓ |
| Triton 编译 | `~/.triton_cache/` (triton 自带) | triton kernel binary | ✓ |

**统一抽象** (`ninetoothed._cache`) 提供:

- `Cache` 类: 两层 KV (in-memory L1 + filesystem L2)
- `hash_function_source(func)`: 内容敏感函数 hash
- `hash_tensor_signature(tensor)`: Tensor 结构指纹
- `hash_value(value)`: 通用 repr-based hash
- `project_files_fingerprint(directory)`: 包代码 fingerprint, 用于跨版本 cache 隔离

## 2. 何时用哪个层

| 你的场景 | 用什么 |
|---|---|
| 缓存**用户 Python 函数**的编译产物 (triton kernel handle, premake .py) | `Cache` (memory + disk) |
| 缓存**配置 / shape** 的查找表 (auto-tune timings) | `Cache` (disk) |
| 缓存**非可序列化**对象 (kernel handle, callable) | `Cache` (memory-only, `cache_dir=None`) |
| 缓存**编译产物** (.so, .csv) | **直接**写文件 (用 `pathlib.Path.write_text` / subprocess), 别用 `Cache` |
| 缓存**任意** (cache key 任意) | `Cache` |

## 3. Cache 类 API

```python
from ninetoothed._cache import Cache

# 内存 + 磁盘 (两层)
cache = Cache(
    cache_dir=Path("~/.ninetoothed/auto_tuning/<ver>"),
    suffix=".json",
    max_memory=64,  # FIFO eviction
)
cache.get(key, default=None)   # L1 -> L2; L2 hit promotes to L1
cache.put(key, value)          # L1 + L2 (atomic disk write)
cache.contains(key)            # L1 ∪ L2
cache.delete(key)              # L1 + L2
cache.clear_memory()           # L1 only; L2 保留 (跨进程)

# 纯内存 (kernel handle 那种不能序列化的)
cache = Cache(max_memory=256)
```

**Key 派生**: 见 §4。

## 4. Key 派生三件套

### `hash_function_source(func)`

```python
from ninetoothed._cache import hash_function_source

h = hash_function_source(my_arrangement_func)  # 'src:<sha256>' 或 'id:<module>.<qualname>@<id>'
```

- 拿 `inspect.getsource(func)` 源码 + SHA256
- **`functools.partial` 自动 unwrap** — `(arrangement, jagged_dim=1)` 和 `(arrangement, jagged_dim=2)` 会被识别为不同 key
- 失败 (lambda / C builtin / REPL) fallback 到 `id:` 形式
- 返回字符串, 长度 64+ char

### `hash_tensor_signature(tensor)`

```python
from ninetoothed._cache import hash_tensor_signature

sig = hash_tensor_signature(t)  # (ndim, jagged_dim, other)
```

- 拿 `(ndim, jagged_dim, other)` 3-tuple
- **不**包含 `Tensor.name` (instance-counter-bound, 每次新 Tensor 都不同)
- 两个**结构**相同的 Tensor 哪怕在不同 `make()` 调用中, signature 也相同

### `hash_value(value)`

```python
from ninetoothed._cache import hash_value
h = hash_value(42)  # 64-char hex SHA256 of repr
```

- 通用 `repr()` + SHA256
- 任何可 repr 的 Python 对象都支持

## 5. Project fingerprint (跨版本隔离)

```python
from ninetoothed._cache import project_files_fingerprint

fp = project_files_fingerprint(Path(__file__).parent)
# 整个 ninetoothed 包源码的合并 SHA256
```

**用法**: cache subdir 名字带 project fingerprint, 让"包代码改版"自然让旧 cache 失效 (旧文件**不**被读, 因为 subdir 名字变了):

```python
subdir = f"{project_files_fingerprint(...)}_triton_{triton.__version__.replace('.', '_')}"
disk_dir = CACHE_DIR / "auto_tuning" / subdir
```

## 6. 实际接入案例

### 案例 1: 缓存用户函数的编译结果 (JIT)

```python
from ninetoothed._cache import Cache, hash_function_source, hash_tensor_signature

_HANDLE_CACHE = Cache(max_memory=256)  # 进程内, kernel handle 不可序列化

def make(arrangement, application, tensors, ...):
    key = (
        hash_function_source(arrangement),
        hash_function_source(application),
        tuple(hash_tensor_signature(t) for t in tensors),
        caller, kernel_name, num_warps, num_stages,
    )
    handle = _HANDLE_CACHE.get(key)
    if handle is not None:
        return handle
    handle = compile_and_jit(arrangement, application, tensors, ...)
    _HANDLE_CACHE.put(key, handle)
    return handle
```

**关键点**:
- `cache_dir=None` → 纯内存 (handle 不可序列化, 不写盘)
- key 内容敏感: 改 arrangement / application 源码 → 重新编译

### 案例 2: 缓存 disk-backed 配置 (auto-tune timings)

```python
disk_dir = CACHE_DIR / "auto_tuning" / f"{_project_key()}_triton_{triton.__version__}"
cache = Cache(cache_dir=disk_dir, suffix=".json", max_memory=64)
cache.put(("timings",), self._all_timings)
```

**关键点**:
- 单文件替代 per-func 多文件 (减少 race 风险)
- 整个 timings dict 用 special key `("_all_timings_",)` 存

### 案例 3: 不该用 Cache 的场景 (AOT 编译产物)

```python
# ❌ 不要这样: 编译 .so 不是 KV
# cache.put((premake, configs), .so_bytes)  # binary, 不能 JSON 序列化

# ✅ 直接文件管理
fingerprint_path.write_text(fingerprint)
nvcc_compile(c_path, so_path)
csv_path.write_text(auto_tune_results)
```

**为什么**: `.so` 是 binary, 不适合 `Cache` 抽象的 JSON 序列化。build 的产物管理 (`.fingerprint` + `.so` + `.csv`) 是 "build artifact manager", 跟 "KV cache" 是不同的概念。

## 7. Thread safety

- `Cache` 是线程安全的: `threading.Lock` 保护 L1 的所有读写
- Disk writes 是 atomic: 写到 `<path>.tmp` + `fsync` + `os.replace`, 防止 reader 看到半写文件
- 多线程并发 `put` / `get` / `delete` 不会损坏 L1 状态
- 多进程并发: 各进程独立 L1, 共享 L2 (write 走 atomic, 容忍最后写赢)

**stress test**: 见 `tests/test_cache.py` 的 `test_concurrent_*` 系列。

## 8. 常见问题

**Q: 改了我的 arrangement / application 函数, 怎么 cache 没失效?**
A: 检查 `hash_function_source` 是否拿到了你改的函数源码。`functools.partial` 会被 unwrap, 内层函数源码被 hash。

**Q: 同一份代码两次跑, cache key 应该一样才对, 但还是 miss?**
A: 看 `hash_function_source` 是不是 fallback 到了 `id:` 形式 (lambda / C builtin 会出现)。fallback 时 key 跟 id() 绑, 不跨进程稳定。

**Q: 怎样彻底清掉 cache?**
A:
```bash
rm -rf ~/.ninetoothed/auto_tuning/   # auto-tune
rm -rf ~/.ninetoothed/*.py            # premake .py 缓存
rm -rf ~/.triton_cache/               # triton 编译
```

**Q: 怎样在 test 里隔离 cache?**
A: 用 `tmp_path` + `monkeypatch.setattr(<module>, "CACHE_DIR", tmp_path / "cache")`, 不要污染 `~/.ninetoothed`。

**Q: Cache 的 disk 写失败了会怎样?**
A: L1 仍然有最新值 (进程内可用), L2 文件保持原状 (没被部分覆写)。下次进程启动会读到旧 disk 值, 不会 crash。

## 9. 性能特性

- L1 hit: ~0.1 µs (dict lookup)
- L1 miss + L2 hit: ~50-200 µs (read + deserialize + promote)
- L1 + L2 miss: 用户提供的 fallback (compute + put)
- L2 write: atomic write (~1-2 ms 取决于磁盘 + payload 大小)

**在 ninetoothed.make 上的实测**:
- first make (compile + JIT): ~24 ms
- cached make (L1 hit): **0.16 ms** (150x speedup)
- 改源码后 (L1 miss, recompile): ~11 ms
- 跨进程复用 (L2 hit): ~0.5 ms

## 10. 设计原则 (给贡献者)

1. **Content-sensitive keys**: 任何 cache key 都要反映真实"输入内容", 不用 `id()` / 指针 / 顺序
2. **Atomic disk writes**: `.tmp + rename`, 别直接 `write_text` (会留下半写文件)
3. **FIFO eviction** (256 entries 足够, 不用 LRU)
4. **Cross-process via L2**, in-process via L1
5. **`project_files_fingerprint`**: 让"包代码改版"自动让旧 cache 失效
6. **Thread-safe by default**: 加任何新方法记得拿 `self._lock`
7. **Memory-only 是合法模式**: 不要假设 cache 一定有 disk
