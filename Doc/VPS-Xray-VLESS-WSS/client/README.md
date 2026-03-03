# Xray 本地客户端管理脚本

功能完善的一键配置脚本：安装 Xray、自动启动、配置开发环境代理，并写入永久环境变量。

## 功能特性

- ✅ **一键安装**：自动下载 Xray、生成配置、配置代理
- ✅ **自动启动**：新终端打开时自动检测并启动 Xray
- ✅ **智能管理**：支持 start/stop/restart/status/log 等命令
- ✅ **环境配置**：GitHub 走代理、HuggingFace 走镜像
- ✅ **永久生效**：配置写入 `.bashrc` 和 `.zshrc`，所有终端可用

## 快速开始

### 新机器一键安装

```bash
./xray-client.sh install
```

这会完成：
1. 下载安装 Xray 客户端
2. 生成配置文件（已预设 VPS 参数）
3. 配置 Git 使用 SOCKS5 代理
4. 写入永久环境变量
5. 启动 Xray

### 日常管理

```bash
./xray-client.sh start      # 启动 Xray
./xray-client.sh stop       # 停止 Xray
./xray-client.sh restart    # 重启 Xray
./xray-client.sh status     # 查看状态
./xray-client.sh log        # 查看日志
./xray-client.sh test       # 测试连接
```

## 自动启动机制

写入 `.bashrc` 的配置会在**每次打开新终端时**：

1. 检查 Xray 是否在运行
2. 如果没有运行且已安装，自动启动 Xray
3. 如果正在运行，设置代理环境变量

这样你只需打开新终端，代理就会自动就绪。

## 本地代理地址

| 协议 | 地址 | 端口 |
|------|------|------|
| SOCKS5 | 127.0.0.1 | 1080 |
| HTTP | 127.0.0.1 | 10808 |

## 环境变量配置

脚本自动配置以下永久环境变量：

```bash
# 代理设置（Xray 运行时自动生效）
export ALL_PROXY=socks5://127.0.0.1:1080
export HTTP_PROXY=http://127.0.0.1:10808
export HTTPS_PROXY=http://127.0.0.1:10808

# HuggingFace 镜像（始终生效）
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=0

# Go 代理
export GOPROXY=https://goproxy.cn,direct
```

## 使用示例

### Git 克隆 GitHub 仓库
```bash
git clone https://github.com/user/repo.git    # 自动走 SOCKS5 代理
```

### HuggingFace 下载模型
```bash
huggingface-cli download gpt2                  # 自动走 hf-mirror

# 或在 Python 中
python -c "from transformers import AutoTokenizer; t=AutoTokenizer.from_pretrained('bert-base-chinese')"
```

### curl 测试代理
```bash
curl -x socks5://127.0.0.1:1080 https://www.google.com
curl -x http://127.0.0.1:10808 https://api.github.com
```

## 配置文件

Xray 配置文件位置：`~/xray-client/config.json`

如需修改 VPS 参数，编辑后重启：
```bash
nano ~/xray-client/config.json
./xray-client.sh restart
```

## 目录结构

```
client/
├── README.md              # 本文件
├── xray-client.sh         # ⭐ 主管理脚本
├── config.json            # Xray 配置文件
├── config.json.tcp        # TCP 配置模板
├── config.json.ws         # WebSocket 配置模板
└── config.json.template   # 通用模板
```

## 卸载

```bash
./xray-client.sh uninstall
```

这会：
- 停止 Xray
- 删除 `~/xray-client/`
- 清除 Git 代理配置
- 提示手动删除 `.bashrc` 中的环境变量
