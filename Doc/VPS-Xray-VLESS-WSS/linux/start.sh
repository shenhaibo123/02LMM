#!/usr/bin/env bash
# 一键启动 Nginx(80) 与 Xray(443)（需已执行 install.sh）
# 用法: ./start.sh  或  sudo ./start.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XRAY_CFG="/usr/local/etc/xray/config.json"

if [[ ! -f "$XRAY_CFG" ]]; then
  echo "未找到 $XRAY_CFG，请先执行: sudo $SCRIPT_DIR/install.sh <域名> [邮箱]"
  exit 1
fi

echo "启动 Nginx 与 Xray..."
systemctl start nginx
systemctl start xray

echo ""
echo "========== 已启动 =========="
systemctl is-active --quiet nginx && echo "  Nginx:  运行中 (80)"  || echo "  Nginx:  未运行"
systemctl is-active --quiet xray  && echo "  Xray:   运行中 (443)" || echo "  Xray:   未运行"
echo ""

# 打印客户端连接信息（从已存在配置读取）
UUID=$(python3 -c "import json; d=json.load(open('$XRAY_CFG')); print(d['inbounds'][0]['settings']['clients'][0]['id'])" 2>/dev/null)
[[ -z "$UUID" ]] && UUID=$(grep -oE '"id":"[a-f0-9-]{36}"' "$XRAY_CFG" 2>/dev/null | head -1 | sed 's/"id":"//;s/"$//')
DOMAIN=$(grep -oE '/etc/letsencrypt/live/[^/]+/' "$XRAY_CFG" 2>/dev/null | head -1 | sed 's|/etc/letsencrypt/live/||;s|/||')
[[ -z "$DOMAIN" ]] && for d in /etc/letsencrypt/live/*/; do [[ -f "${d}fullchain.pem" ]] && DOMAIN=$(basename "$d") && break; done

if [[ -n "$UUID" && -n "$DOMAIN" ]]; then
  echo "客户端参数："
  echo "  地址: $DOMAIN  端口: 443  UUID: $UUID"
  echo "  传输: tcp  flow: xtls-rprx-vision  SNI: $DOMAIN"
  echo "  链接: vless://${UUID}@${DOMAIN}:443?security=tls&encryption=none&headerType=none&fp=chrome&type=tcp&flow=xtls-rprx-vision#vps"
  echo "=========================================="
fi
