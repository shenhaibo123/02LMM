#!/usr/bin/env bash
# 仅启动 Nginx(80) + Xray(443)，与 Linux 对齐；在 mac/ 目录执行
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAC_VPS="$SCRIPT_DIR/mac-vps"
[[ -d "$MAC_VPS" ]] || { echo "请先运行 ./install.sh"; exit 1; }

MIME_TYPES=""
for d in "$(brew --prefix)/opt/nginx/etc/nginx" "$(brew --prefix)/etc/nginx"; do
  [[ -f "$d/mime.types" ]] && MIME_TYPES="$d/mime.types" && break
done
[[ -n "$MIME_TYPES" ]] && NGINX_USER=$(id -un) && NGINX_GROUP=$(id -gn) && cat > "$MAC_VPS/nginx.conf" << NGINX_EOF
worker_processes 1;
pid $MAC_VPS/nginx.pid;
user $NGINX_USER $NGINX_GROUP;
events { worker_connections 64; }
http {
    include       $MIME_TYPES;
    default_type  application/octet-stream;
    sendfile      on;
    keepalive_timeout 65;
    server {
        listen 80;
        server_name _;
        root $MAC_VPS/html;
        index index.html;
        location / { try_files \$uri \$uri/ =404; }
    }
}
NGINX_EOF

echo "启动 Nginx（仅 80）..."
sudo brew services stop nginx 2>/dev/null || true
sudo nginx -s stop -c "$MAC_VPS/nginx.conf" 2>/dev/null || true
sudo pkill -9 nginx 2>/dev/null || true
sleep 1
PID80=$(lsof -ti :80 2>/dev/null | tr '\n' ' ')
[[ -n "$PID80" ]] && sudo kill -9 $PID80 2>/dev/null || true
sleep 0.5
sudo nginx -c "$MAC_VPS/nginx.conf" || { echo "Nginx 启动失败"; exit 1; }

echo "启动 Xray（443 直连）..."
pkill -f "xray run -c $MAC_VPS/config.json" 2>/dev/null || true
"$(brew --prefix)/bin/xray" run -c "$MAC_VPS/config.json" >> "$MAC_VPS/xray.log" 2>&1 &
echo $! > "$MAC_VPS/xray.pid"

set +e
UUID=$(python3 -c "import json; d=json.load(open('$MAC_VPS/config.json')); print(d['inbounds'][0]['settings']['clients'][0]['id'])" 2>/dev/null)
[[ -z "$UUID" ]] && UUID=$(grep -oE '"id":"[a-f0-9-]{36}"' "$MAC_VPS/config.json" 2>/dev/null | head -1 | sed 's/"id":"//;s/"$//')
LAN_IP=$(ipconfig getifaddr en0 2>/dev/null)
[[ -z "$LAN_IP" ]] && LAN_IP=$(ipconfig getifaddr en1 2>/dev/null)
[[ -z "$LAN_IP" ]] && LAN_IP=$(ifconfig 2>/dev/null | grep " inet " | grep -v 127.0.0.1 | awk '{print $2}' | grep -E "^(192\.168\.|10\.)" | head -1)
set -e

echo ""
echo "=============================================="
echo "  服务已启动"
echo "  本机:  http://localhost  （80 主页）"
echo "  代理:  443 直连，VLESS+TCP+TLS+XTLS-Vision"
echo "  UUID:  $UUID"
echo "  局域网: ${LAN_IP:-（未取到）}"
echo "  链接:  vless://${UUID}@${LAN_IP:-IP}:443?security=tls&encryption=none&headerType=none&fp=chrome&type=tcp&flow=xtls-rprx-vision#mac"
echo "  手机 v2rayNG: 导入链接后须填 SNI=localhost、勾选允许不安全证书"
echo "  停止:  ./stop.sh"
echo "=============================================="
