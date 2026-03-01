#!/usr/bin/env bash
# Mac 本地验证：与 Linux 一致，80=Nginx 主页，443=Xray 直连（VLESS+TCP+TLS+XTLS-Vision，无 path）
# 用法: 在 mac/ 目录执行 ./install.sh；从本层读取 default、config.json、www

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAC_VPS="$SCRIPT_DIR/mac-vps"

echo "===== Mac 本地验证（与 Linux 对齐：80 主页，443 Vision 直连）====="
[[ "$(uname -s)" = Darwin ]] || { echo "仅支持 macOS"; exit 1; }

echo "[1/6] 检查本层模板（default、config.json、www）..."
for f in "default" "config.json"; do
  [[ -f "$SCRIPT_DIR/$f" ]] || { echo "缺少: $SCRIPT_DIR/$f"; exit 1; }
done
[[ -d "$SCRIPT_DIR/www" ]] || { echo "缺少: $SCRIPT_DIR/www"; exit 1; }

echo "[2/6] 安装 Nginx、Xray（若未安装）..."
command -v brew &>/dev/null || { echo "请先安装 Homebrew: https://brew.sh"; exit 1; }
brew list nginx &>/dev/null || brew install nginx
brew list xray &>/dev/null || brew install xray

echo "[3/6] 生成自签名证书与目录..."
mkdir -p "$MAC_VPS/certs" "$MAC_VPS/html"
cp -r "$SCRIPT_DIR/www/"* "$MAC_VPS/html/" 2>/dev/null || true
[[ -f "$MAC_VPS/certs/cert.pem" ]] || openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout "$MAC_VPS/certs/key.pem" -out "$MAC_VPS/certs/cert.pem" \
  -subj "/CN=localhost" -addext "subjectAltName=DNS:localhost,IP:127.0.0.1" 2>/dev/null

echo "[4/6] 生成 UUID 与 Xray 配置..."
UUID="$("$(brew --prefix)/bin/xray" uuid 2>/dev/null || true)"
[[ -z "$UUID" ]] && UUID="$(uuidgen 2>/dev/null || echo "a1b2c3d4-e5f6-4789-a012-000000000001")"
sed -e "s/YOUR_UUID/$UUID/g" \
  -e "s|/etc/letsencrypt/live/YOUR_DOMAIN/fullchain.pem|$MAC_VPS/certs/cert.pem|g" \
  -e "s|/etc/letsencrypt/live/YOUR_DOMAIN/privkey.pem|$MAC_VPS/certs/key.pem|g" \
  "$SCRIPT_DIR/config.json" > "$MAC_VPS/config.json"

MIME_TYPES=""
for d in "$(brew --prefix)/opt/nginx/etc/nginx" "$(brew --prefix)/etc/nginx"; do
  [[ -f "$d/mime.types" ]] && MIME_TYPES="$d/mime.types" && break
done
[[ -z "$MIME_TYPES" ]] && { echo "未找到 nginx mime.types"; exit 1; }
NGINX_USER=$(id -un)
NGINX_GROUP=$(id -gn)
cat > "$MAC_VPS/nginx.conf" << NGINX_EOF
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

echo "[5/6] 启动 Nginx（仅 80，需输入本机密码）..."
sudo brew services stop nginx 2>/dev/null || true
sudo nginx -s stop -c "$MAC_VPS/nginx.conf" 2>/dev/null || true
sudo nginx -c "$MAC_VPS/nginx.conf" || { echo "Nginx 启动失败"; exit 1; }

echo "[6/6] 启动 Xray（443 直连）..."
pkill -f "xray run -c $MAC_VPS/config.json" 2>/dev/null || true
"$(brew --prefix)/bin/xray" run -c "$MAC_VPS/config.json" >> "$MAC_VPS/xray.log" 2>&1 &
echo $! > "$MAC_VPS/xray.pid"

set +e
LAN_IP=$(ipconfig getifaddr en0 2>/dev/null)
[[ -z "$LAN_IP" ]] && LAN_IP=$(ipconfig getifaddr en1 2>/dev/null)
[[ -z "$LAN_IP" ]] && LAN_IP=$(ifconfig 2>/dev/null | grep " inet " | grep -v 127.0.0.1 | awk '{print $2}' | grep -E "^(192\.168\.|10\.)" | head -1)
set -e
echo ""
echo "========== Mac 本地验证已启动 =========="
echo "  80 主页:  http://localhost"
echo "  443 代理: VLESS+TCP+TLS+XTLS-Vision，无 path"
echo "  UUID:     $UUID"
echo "  局域网:   ${LAN_IP:-（未取到）}"
echo "  链接:     vless://${UUID}@${LAN_IP:-IP}:443?security=tls&encryption=none&headerType=none&fp=chrome&type=tcp&flow=xtls-rprx-vision#mac"
echo "  手机 v2rayNG: 导入链接后须填 SNI=localhost、勾选允许不安全证书"
echo "  停止:     ./stop.sh"
echo "========================================"
