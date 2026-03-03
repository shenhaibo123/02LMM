#!/bin/bash
#
# Xray 本地客户端管理脚本
# 功能: 安装、启动、停止、查看状态，并配置开发环境代理
# 用法:
#   ./xray-client.sh install    # 首次安装并配置
#   ./xray-client.sh start      # 启动 Xray
#   ./xray-client.sh stop       # 停止 Xray
#   ./xray-client.sh status     # 查看状态
#   ./xray-client.sh restart    # 重启 Xray
#   ./xray-client.sh config     # 仅配置环境变量（不启动）
#

set -e

# ===== 颜色输出 =====
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# ===== 配置参数 =====
XRAY_VERSION="1.8.23"
XRAY_DIR="$HOME/xray-client"
CONFIG_FILE="$XRAY_DIR/config.json"
PID_FILE="$XRAY_DIR/xray.pid"
LOG_FILE="$XRAY_DIR/xray.log"

# VPS 连接信息
VPS_DOMAIN="zhliangqi.com"
VPS_UUID="08863772-6378-4501-955d-c428ae8069db"
VPS_PORT=443

# ===== 辅助函数 =====
print_help() {
    echo -e "${BLUE}Xray 本地客户端管理脚本${NC}"
    echo ""
    echo "用法: $0 <命令>"
    echo ""
    echo "命令:"
    echo "  install   首次安装 Xray 并配置环境（自动启动）"
    echo "  start     启动 Xray 客户端"
    echo "  stop      停止 Xray 客户端"
    echo "  restart   重启 Xray 客户端"
    echo "  status    查看 Xray 运行状态"
    echo "  log       查看 Xray 日志"
    echo "  config    仅配置环境变量（不启动 Xray）"
    echo "  test      测试代理连接"
    echo "  uninstall 卸载 Xray 客户端"
    echo ""
    echo "示例:"
    echo "  $0 install    # 新机器首次运行"
    echo "  $0 start      # 启动代理"
    echo "  $0 status     # 查看是否运行"
}

check_xray_running() {
    if [[ -f "$PID_FILE" ]]; then
        local pid=$(cat "$PID_FILE" 2>/dev/null)
        if kill -0 "$pid" 2>/dev/null; then
            return 0  # 正在运行
        fi
    fi
    # 再次检查进程
    local pid2=$(pgrep -f "xray run" | head -1)
    if [[ -n "$pid2" ]]; then
        echo "$pid2" > "$PID_FILE"
        return 0
    fi
    return 1  # 未运行
}

get_xray_pid() {
    if [[ -f "$PID_FILE" ]]; then
        cat "$PID_FILE" 2>/dev/null
    else
        pgrep -f "xray run" | head -1
    fi
}

# ===== 安装 Xray =====
install_xray() {
    echo -e "${BLUE}[1/5] 检测系统并下载 Xray...${NC}"

    OS=$(uname -s)
    ARCH=$(uname -m)

    case "$OS" in
        Linux)
            case "$ARCH" in
                x86_64)  PLATFORM="linux-64" ;;
                aarch64) PLATFORM="linux-arm64-v8a" ;;
                armv7l)  PLATFORM="linux-arm32-v7a" ;;
                *)       echo -e "${RED}不支持的架构: $ARCH${NC}"; exit 1 ;;
            esac
            ;;
        Darwin)
            case "$ARCH" in
                x86_64) PLATFORM="macos-64" ;;
                arm64)  PLATFORM="macos-arm64-v8a" ;;
                *)      echo -e "${RED}不支持的架构: $ARCH${NC}"; exit 1 ;;
            esac
            ;;
        *)
            echo -e "${RED}不支持的操作系统: $OS${NC}"
            exit 1
            ;;
    esac

    echo -e "  系统: $OS $ARCH"
    mkdir -p "$XRAY_DIR"

    # 下载 Xray
    if [[ ! -f "$XRAY_DIR/xray" ]]; then
        echo -e "  下载 Xray $XRAY_VERSION..."
        cd "$XRAY_DIR"
        DOWNLOAD_URL="https://github.com/XTLS/Xray-core/releases/download/v${XRAY_VERSION}/Xray-${PLATFORM}.zip"

        if command -v curl &>/dev/null; then
            curl -L -o xray.zip "$DOWNLOAD_URL" 2>/dev/null || wget -O xray.zip "$DOWNLOAD_URL"
        elif command -v wget &>/dev/null; then
            wget -O xray.zip "$DOWNLOAD_URL"
        else
            echo -e "${RED}需要 curl 或 wget${NC}"
            exit 1
        fi

        unzip -q -o xray.zip
        chmod +x xray
        rm -f xray.zip LICENSE README.md
        echo -e "${GREEN}  ✓ Xray 安装完成${NC}"
    else
        echo -e "${GREEN}  ✓ Xray 已存在${NC}"
    fi
}

# ===== 生成配置文件 =====
generate_config() {
    echo -e "${BLUE}[2/5] 生成 Xray 配置文件...${NC}"

    cat > "$CONFIG_FILE" << 'XRAY_CONFIG'
{
  "log": {
    "loglevel": "warning"
  },
  "inbounds": [
    {
      "tag": "socks-in",
      "protocol": "socks",
      "listen": "127.0.0.1",
      "port": 1080,
      "settings": {
        "auth": "noauth",
        "udp": true,
        "ip": "127.0.0.1"
      }
    },
    {
      "tag": "http-in",
      "protocol": "http",
      "listen": "127.0.0.1",
      "port": 10808,
      "settings": {}
    }
  ],
  "outbounds": [
    {
      "tag": "proxy",
      "protocol": "vless",
      "settings": {
        "vnext": [
          {
            "address": "YOUR_DOMAIN",
            "port": YOUR_PORT,
            "users": [
              {
                "id": "YOUR_UUID",
                "flow": "xtls-rprx-vision",
                "encryption": "none",
                "level": 0
              }
            ]
          }
        ]
      },
      "streamSettings": {
        "network": "tcp",
        "security": "tls",
        "tlsSettings": {
          "serverName": "YOUR_DOMAIN",
          "allowInsecure": false
        }
      }
    },
    {
      "tag": "direct",
      "protocol": "freedom",
      "settings": {}
    },
    {
      "tag": "block",
      "protocol": "blackhole",
      "settings": {}
    }
  ],
  "routing": {
    "domainStrategy": "IPIfNonMatch",
    "rules": [
      {
        "type": "field",
        "ip": ["geoip:private"],
        "outboundTag": "direct"
      },
      {
        "type": "field",
        "domain": ["geosite:cn"],
        "outboundTag": "direct"
      },
      {
        "type": "field",
        "ip": ["geoip:cn"],
        "outboundTag": "direct"
      }
    ]
  }
}
XRAY_CONFIG

    # 替换参数
    sed -i "s/YOUR_DOMAIN/$VPS_DOMAIN/g" "$CONFIG_FILE"
    sed -i "s/YOUR_UUID/$VPS_UUID/g" "$CONFIG_FILE"
    sed -i "s/YOUR_PORT/$VPS_PORT/g" "$CONFIG_FILE"

    echo -e "${GREEN}  ✓ 配置文件: $CONFIG_FILE${NC}"
}

# ===== 启动 Xray =====
start_xray() {
    if check_xray_running; then
        local pid=$(get_xray_pid)
        echo -e "${YELLOW}Xray 已在运行 (PID: $pid)${NC}"
        return 0
    fi

    echo -e "${BLUE}启动 Xray 客户端...${NC}"

    if [[ ! -f "$XRAY_DIR/xray" ]]; then
        echo -e "${RED}未找到 Xray，请先运行: $0 install${NC}"
        exit 1
    fi

    cd "$XRAY_DIR"
    nohup ./xray run -c "$CONFIG_FILE" > "$LOG_FILE" 2>&1 &
    local pid=$!
    sleep 2

    if kill -0 "$pid" 2>/dev/null; then
        echo "$pid" > "$PID_FILE"
        echo -e "${GREEN}✓ Xray 启动成功 (PID: $pid)${NC}"
        echo -e "  SOCKS5: 127.0.0.1:1080"
        echo -e "  HTTP:   127.0.0.1:10808"
        return 0
    else
        echo -e "${RED}✗ Xray 启动失败${NC}"
        echo -e "查看日志: $LOG_FILE"
        return 1
    fi
}

# ===== 停止 Xray =====
stop_xray() {
    echo -e "${BLUE}停止 Xray 客户端...${NC}"

    local stopped=0

    # 从 PID 文件停止
    if [[ -f "$PID_FILE" ]]; then
        local pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null
            stopped=1
        fi
        rm -f "$PID_FILE"
    fi

    # 从进程名停止
    local pids=$(pgrep -f "xray run" || true)
    if [[ -n "$pids" ]]; then
        echo "$pids" | xargs kill 2>/dev/null || true
        stopped=1
    fi

    if [[ $stopped -eq 1 ]]; then
        echo -e "${GREEN}✓ Xray 已停止${NC}"
    else
        echo -e "${YELLOW}Xray 未在运行${NC}"
    fi
}

# ===== 查看状态 =====
show_status() {
    echo -e "${BLUE}=== Xray 状态 ===${NC}"

    if check_xray_running; then
        local pid=$(get_xray_pid)
        echo -e "状态: ${GREEN}运行中${NC} (PID: $pid)"
        echo -e "配置文件: $CONFIG_FILE"
        echo -e "日志文件: $LOG_FILE"
        echo ""
        echo -e "${BLUE}本地代理地址:${NC}"
        echo -e "  SOCKS5: ${GREEN}127.0.0.1:1080${NC}"
        echo -e "  HTTP:   ${GREEN}127.0.0.1:10808${NC}"
        echo ""
        echo -e "${BLUE}测试连接:${NC}"
        local test_result=$(curl -s --max-time 5 -x socks5://127.0.0.1:1080 http://www.google.com -o /dev/null -w "%{http_code}" 2>/dev/null || echo "failed")
        if [[ "$test_result" == "200" ]]; then
            echo -e "  Google:   ${GREEN}✓ 正常${NC}"
        else
            echo -e "  Google:   ${RED}✗ 失败 (HTTP $test_result)${NC}"
        fi
    else
        echo -e "状态: ${RED}未运行${NC}"
        echo -e "启动命令: ${YELLOW}$0 start${NC}"
    fi
}

# ===== 配置环境变量 =====
config_env() {
    echo -e "${BLUE}[3/5] 配置 Git 代理...${NC}"

    # Git 代理
    git config --global http.https://github.com.proxy socks5://127.0.0.1:1080
    git config --global https.https://github.com.proxy socks5://127.0.0.1:1080
    echo -e "${GREEN}  ✓ Git 代理配置完成${NC}"

    echo -e "${BLUE}[4/5] 写入永久环境变量...${NC}"

    # 配置内容
    local proxy_marker="# ===== Xray 本地代理配置 ====="
    local proxy_config='
# ===== Xray 本地代理配置 =====
# 自动检测并启动 Xray
auto_start_xray() {
    local xray_dir="$HOME/xray-client"
    local pid_file="$xray_dir/xray.pid"

    # 检查是否已在运行
    if [[ -f "$pid_file" ]]; then
        local pid=$(cat "$pid_file" 2>/dev/null)
        if kill -0 "$pid" 2>/dev/null; then
            # Xray 正在运行，设置代理
            export ALL_PROXY=socks5://127.0.0.1:1080
            export HTTP_PROXY=http://127.0.0.1:10808
            export HTTPS_PROXY=http://127.0.0.1:10808
            export http_proxy=http://127.0.0.1:10808
            export https_proxy=http://127.0.0.1:10808
            return 0
        fi
    fi

    # 检查 xray 是否安装
    if [[ -f "$xray_dir/xray" ]] && [[ -f "$xray_dir/config.json" ]]; then
        # 自动启动 Xray
        cd "$xray_dir"
        nohup ./xray run -c "$xray_dir/config.json" > "$xray_dir/xray.log" 2>&1 &
        local new_pid=$!
        sleep 1
        if kill -0 "$new_pid" 2>/dev/null; then
            echo "$new_pid" > "$pid_file"
            echo "[Xray] 已自动启动 (PID: $new_pid)"
            export ALL_PROXY=socks5://127.0.0.1:1080
            export HTTP_PROXY=http://127.0.0.1:10808
            export HTTPS_PROXY=http://127.0.0.1:10808
            export http_proxy=http://127.0.0.1:10808
            export https_proxy=http://127.0.0.1:10808
        fi
    fi
}

# 运行自动启动
auto_start_xray

# HuggingFace 镜像配置
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=0

# Go 代理
export GOPROXY=https://goproxy.cn,direct
# ===== Xray 配置结束 =====
'

    # 写入 .bashrc
    if ! grep -q "$proxy_marker" "$HOME/.bashrc" 2>/dev/null; then
        echo "$proxy_config" >> "$HOME/.bashrc"
        echo -e "${GREEN}  ✓ 写入 ~/.bashrc${NC}"
    else
        echo -e "${YELLOW}  ~/.bashrc 已配置，跳过${NC}"
    fi

    # 写入 .zshrc（如果存在）
    if [[ -f "$HOME/.zshrc" ]] && ! grep -q "$proxy_marker" "$HOME/.zshrc" 2>/dev/null; then
        echo "$proxy_config" >> "$HOME/.zshrc"
        echo -e "${GREEN}  ✓ 写入 ~/.zshrc${NC}"
    fi

    # 立即生效
    auto_start_xray 2>/dev/null || true
    export HF_ENDPOINT=https://hf-mirror.com
    export HF_HUB_ENABLE_HF_TRANSFER=0
    export GOPROXY=https://goproxy.cn,direct
}

# ===== 测试连接 =====
test_connection() {
    echo -e "${BLUE}=== 测试代理连接 ===${NC}"

    if ! check_xray_running; then
        echo -e "${RED}Xray 未运行，先启动: $0 start${NC}"
        return 1
    fi

    echo -ne "Google:    "
    local google=$(curl -s --max-time 10 -x socks5://127.0.0.1:1080 http://www.google.com -o /dev/null -w "%{http_code}" 2>/dev/null || echo "failed")
    if [[ "$google" == "200" ]]; then
        echo -e "${GREEN}✓ OK${NC}"
    else
        echo -e "${RED}✗ Failed ($google)${NC}"
    fi

    echo -ne "GitHub:    "
    local github=$(curl -s --max-time 10 -x socks5://127.0.0.1:1080 https://api.github.com -o /dev/null -w "%{http_code}" 2>/dev/null || echo "failed")
    if [[ "$github" == "200" ]]; then
        echo -e "${GREEN}✓ OK${NC}"
    else
        echo -e "${RED}✗ Failed ($github)${NC}"
    fi

    echo -ne "HF Mirror: "
    local hf=$(curl -s --max-time 10 https://hf-mirror.com/api/models/gpt2 -o /dev/null -w "%{http_code}" 2>/dev/null || echo "failed")
    if [[ "$hf" == "200" ]] || [[ "$hf" == "307" ]]; then
        echo -e "${GREEN}✓ OK${NC}"
    else
        echo -e "${RED}✗ Failed ($hf)${NC}"
    fi
}

# ===== 查看日志 =====
show_log() {
    if [[ -f "$LOG_FILE" ]]; then
        echo -e "${BLUE}=== Xray 日志 (最后 50 行) ===${NC}"
        tail -50 "$LOG_FILE"
    else
        echo -e "${YELLOW}日志文件不存在${NC}"
    fi
}

# ===== 卸载 =====
uninstall() {
    echo -e "${BLUE}卸载 Xray 客户端...${NC}"
    stop_xray
    rm -rf "$XRAY_DIR"

    # 清除 Git 配置
    git config --global --unset http.https://github.com.proxy 2>/dev/null || true
    git config --global --unset https.https://github.com.proxy 2>/dev/null || true

    # 清除环境变量（需要手动编辑 .bashrc）
    echo -e "${YELLOW}请手动从 ~/.bashrc 和 ~/.zshrc 中删除 Xray 配置段落${NC}"

    echo -e "${GREEN}✓ 卸载完成${NC}"
}

# ===== 主程序 =====
main() {
    case "${1:-}" in
        install)
            echo -e "${BLUE}========================================${NC}"
            echo -e "${BLUE}  Xray 客户端一键安装配置${NC}"
            echo -e "${BLUE}========================================${NC}"
            echo ""
            install_xray
            generate_config
            config_env
            start_xray
            echo ""
            echo -e "${GREEN}[5/5] 安装完成！${NC}"
            echo ""
            test_connection
            ;;
        start)
            start_xray
            ;;
        stop)
            stop_xray
            ;;
        restart)
            stop_xray
            sleep 1
            start_xray
            ;;
        status)
            show_status
            ;;
        log)
            show_log
            ;;
        config)
            config_env
            echo ""
            echo -e "${GREEN}环境变量配置完成！重新打开终端生效${NC}"
            ;;
        test)
            test_connection
            ;;
        uninstall)
            uninstall
            ;;
        help|--help|-h)
            print_help
            ;;
        "")
            # 默认执行 install
            main "install"
            ;;
        *)
            echo -e "${RED}未知命令: $1${NC}"
            print_help
            exit 1
            ;;
    esac
}

main "$@"
