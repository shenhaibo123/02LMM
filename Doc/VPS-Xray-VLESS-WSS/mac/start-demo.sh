#!/usr/bin/env bash
# 演示模式：8080（主页）、8443（代理），无需 sudo；在 mac/ 目录执行
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAC_VPS="$SCRIPT_DIR/mac-vps"
DEMO_PORT_80=8080
DEMO_PORT_443=8443

[[ -d "$MAC_VPS" ]] || { echo "请先运行 ./install.sh"; exit 1; }

MIME_TYPES=""
for d in "$(brew --prefix)/opt/nginx/etc/nginx" "$(brew --prefix)/etc/nginx"; do
  [[ -f "$d/mime.types" ]] && MIME_TYPES="$d/mime.types" && break
done
[[ -z "$MIME_TYPES" ]] && { echo "未找到 nginx mime.types"; exit 1; }
NGINX_USER=$(id -un)
NGINX_GROUP=$(id -gn)
cat > "$MAC_VPS/nginx-demo.conf" << EOF
worker_processes 1;
pid $MAC_VPS/nginx-demo.pid;
error_log $MAC_VPS/nginx-demo-error.log;
user $NGINX_USER $NGINX_GROUP;
events { worker_connections 64; }
http {
    access_log $MAC_VPS/nginx-demo-access.log;
    include       $MIME_TYPES;
    default_type  application/octet-stream;
    sendfile      on;
    server {
        listen $DEMO_PORT_80;
        server_name _;
        root $MAC_VPS/html;
        index index.html;
        location / { try_files \$uri \$uri/ =404; }
    }
}
EOF

sed 's/"port": *443/"port": '"$DEMO_PORT_443"'/' "$MAC_VPS/config.json" > "$MAC_VPS/config-demo.json"

echo "演示模式启动（${DEMO_PORT_80}=主页 ${DEMO_PORT_443}=代理，无需 sudo）..."
pkill -f "xray run -c $MAC_VPS/config.json" 2>/dev/null || true
pkill -f "xray run -c $MAC_VPS/config-demo.json" 2>/dev/null || true
pkill -f "xray-server-debug.json" 2>/dev/null || true
sleep 1
nginx -s stop -c "$MAC_VPS/nginx.conf" 2>/dev/null || true
nginx -s stop -c "$MAC_VPS/nginx-demo.conf" 2>/dev/null || true

nginx -c "$MAC_VPS/nginx-demo.conf" || true
nohup "$(brew --prefix)/bin/xray" run -c "$MAC_VPS/config-demo.json" >> "$MAC_VPS/xray-demo.log" 2>&1 &
echo $! > "$MAC_VPS/xray-demo.pid"
sleep 5
for i in $(seq 1 20); do
  nc -z 127.0.0.1 "$DEMO_PORT_443" 2>/dev/null && break
  sleep 1
done
set +e
LAN_IP=$(ipconfig getifaddr en0 2>/dev/null)
[[ -z "$LAN_IP" ]] && LAN_IP=$(ipconfig getifaddr en1 2>/dev/null)
[[ -z "$LAN_IP" ]] && LAN_IP=$(ifconfig 2>/dev/null | grep " inet " | grep -v 127.0.0.1 | awk '{print $2}' | grep -E "^(192\.168\.|10\.)" | head -1)
UUID=$(python3 -c "import json; d=json.load(open('$MAC_VPS/config.json')); print(d['inbounds'][0]['settings']['clients'][0]['id'])" 2>/dev/null)
set -e
echo "已启动: 主页 http://localhost:$DEMO_PORT_80  代理端口 $DEMO_PORT_443"
echo "  链接(演示): vless://${UUID}@${LAN_IP:-IP}:${DEMO_PORT_443}?security=tls&encryption=none&headerType=none&fp=chrome&type=tcp&flow=xtls-rprx-vision#mac"
echo "  手机 v2rayNG: SNI=localhost、勾选允许不安全证书"
echo ""
echo "===== 代理链路测试 ====="
SKIP_PORT_CHECK=1 PROXY_PORT=$DEMO_PORT_443 "$SCRIPT_DIR/test-proxy-443.sh" || true
echo ""
echo "===== 验证 80/443 ====="
export VERIFY_PORT_80=$DEMO_PORT_80 VERIFY_PORT_443=$DEMO_PORT_443
"$SCRIPT_DIR/verify.sh"
