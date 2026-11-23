# operator-mcp

一个用于远程服务器操作的 MCP Server。通过 SSH 连接到目标机器后，可以上传代码、执行测试命令，并查看 GPU 状态。

## 功能

- `ping`：快速确认 MCP 服务可用
- `upload_code`：上传本地文件或目录到远端
- `run_test`：在远端指定目录下执行测试命令
- `mx_smI`：查看远端 GPU 状态（优先调用 `mx-smi`）

## 环境要求

- Python 3.12+
- 可访问的 SSH 服务器
- 目标服务器上已配置好远程工作目录

## 安装

```bash
uv sync
```

## 配置

先复制 `config.example.json` 为 `config.json`，再填写 SSH 连接信息：

```bash
copy config.example.json config.json
```

`config.json` 示例：

```json
{
  "host": "your-host",
  "port": 22,
  "user": "your-user",
  "password": "your-password",
  "remote_root": "/data"
}
```

如果你使用密钥登录，可以改为：

```json
{
  "host": "your-host",
  "port": 22,
  "user": "your-user",
  "key_filename": "C:/path/to/id_rsa",
  "remote_root": "/data"
}
```

说明：

- `password` 和 `key_filename` 二选一
- `remote_root` 是远端基础目录，所有上传和测试都会在这个目录下进行

## 安装MCP

```bash
uv sync
```

## 工具说明

所有工具都以 `remote_root`（config.json 中的 `remote_root` 字段）为远端基础目录。`path` 参数都是相对路径。

### 连接与基础
- **`ping()`** — 返回 `pong`，确认服务可用
- **`upload_code(src, dst)`** — 上传本地文件/目录（目录自动 tar.gz 打包）
- **`download_file(remote_path, local_path)`** — 下载远端文件/目录到本地

### 测试与运行
- **`run_test(cmd, timeout=300, workdir="", env="default")`** — 在远端执行命令。`env="default"` 自动注入 MetaX 必需环境变量；Python 命令自动 tee 到 `logs/<script>_<timestamp>.log`。
- **`run_python(code, workdir="", timeout=120)`** — 运行内联 Python 代码，无需本地文件。适合快速探针、cache 清理、版本检查等。代码用 base64 编码后传输，避免 shell 转义。
- **`auto_bench(v0, v1, workdir="", atol=1e-2, repeat=500, timeout=600, runs=1)`** — 一键运行 auto_bench.py。`runs>1` 时输出 per-run 表格 + median/min/max 聚合统计。

### 远端文件操作
- **`remote_ls(path="")`** — 列出远端目录内容（含文件大小、修改时间）
- **`remote_cat(path, lines=0)`** — 读取远端文件（`lines=0` 读全部，正整数读前 N 行）
- **`remote_md5(path)`** — 计算远端文件 MD5，用于确认版本
- **`remote_glob(pattern, path="")`** — 按 glob 模式查找远端文件（支持 `**/*.py`、`diag_*.py` 等）
- **`remote_rm(path, recursive=False)`** — 删除远端文件或目录。内置安全保护：禁止删除 `remote_root`、`/data`、`/opt` 等危险路径；删目录需显式开启 `recursive=True`。

### GPU 监控
- **`mx_smi(options="")`** — 查看 MetaX GPU 状态。常用 options：`""`（概览）、`"-l 1000"`（每秒刷新）、`"-i 0"`（GPU 0）、`"--show-usage"`（利用率）。

## 注意事项

- 请勿把真实密码或密钥路径提交到公共仓库
- 上传目录时当前实现只处理目录下的普通文件，不会递归上传子目录
- `config.json` 中的连接信息会在启动时直接读取，修改后需要重启服务