#!/usr/bin/env bash
# VLESS + TCP + TLS + XTLS-Vision 一键安装（80 主页由 Nginx，443 由 Xray 直连，无 path）
# 用法: sudo ./install.sh <域名> [邮箱]
# 域名需已解析到本机公网 IP；邮箱用于 Let's Encrypt 通知（可选）；UUID 自动生成

set -e

DOMAIN="${1:?用法: sudo ./install.sh <域名> [邮箱]，例如: sudo ./install.sh proxy.example.com}"
EMAIL="${2:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 必须 root 运行
[[ "$EUID" -eq 0 ]] || { echo "请使用 sudo 运行此脚本"; exit 1; }

echo "[1/9] 检查本层模板文件（default、config.json、www）..."
for f in "default" "config.json"; do
  [[ -f "$SCRIPT_DIR/$f" ]] || { echo "缺少文件: $SCRIPT_DIR/$f"; exit 1; }
done
[[ -d "$SCRIPT_DIR/www" ]] || { echo "缺少目录: $SCRIPT_DIR/www（本地主页）"; exit 1; }

echo "[2/9] 安装 Nginx、Certbot、curl..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq nginx certbot curl >/dev/null

echo "[3/9] 安装 Xray-core（官方脚本）..."
bash -c "$(curl -L https://github.com/XTLS/Xray-install/raw/main/install-release.sh)" @ install

echo "[4/9] 申请 TLS 证书（Let's Encrypt）..."
systemctl stop nginx 2>/dev/null || true
if [[ -n "$EMAIL" ]]; then
  certbot certonly --standalone -d "$DOMAIN" --non-interactive --agree-tos --email "$EMAIL"
else
  certbot certonly --standalone -d "$DOMAIN" --non-interactive --agree-tos --register-unsaved-email
fi

echo "[5/9] 生成 UUID..."
UUID="$("/usr/local/bin/xray" uuid 2>/dev/null || true)"
[[ -z "$UUID" ]] && UUID="$(cat /proc/sys/kernel/random/uuid 2>/dev/null || true)"
[[ -z "$UUID" ]] && { echo "无法生成 UUID"; exit 1; }
echo "     UUID: $UUID"

echo "[6/9] 部署本地主页到 /var/www/html..."
mkdir -p /var/www/html
cp -r "$SCRIPT_DIR/www/"* /var/www/html/
chown -R www-data:www-data /var/www/html 2>/dev/null || true

echo "[7/9] 写入 Nginx 站点配置并启动..."
sed "s/YOUR_DOMAIN/$DOMAIN/g" "$SCRIPT_DIR/default" > /etc/nginx/sites-available/default
nginx -t
systemctl start nginx
systemctl enable nginx

echo "[8/9] 写入 Xray 配置并启动（443 直连，VLESS+TCP+TLS+XTLS-Vision）..."
mkdir -p /usr/local/etc/xray
sed -e "s/YOUR_UUID/$UUID/g" -e "s/YOUR_DOMAIN/$DOMAIN/g" "$SCRIPT_DIR/config.json" > /usr/local/etc/xray/config.json
systemctl enable xray
systemctl restart xray

echo "[9/9] 完成"

echo ""
echo "========== 安装完成 =========="
echo "域名:     $DOMAIN"
echo "UUID:     $UUID"
echo "端口:     443 (Xray 直连，VLESS+TCP+TLS+XTLS-Vision，无 path)"
echo ""
echo "客户端："
echo "  协议: VLESS  地址: $DOMAIN  端口: 443  TLS: 开"
echo "  传输: tcp  flow: xtls-rprx-vision  SNI: $DOMAIN  无 path"
echo ""
echo "  链接: vless://${UUID}@${DOMAIN}:443?security=tls&encryption=none&headerType=none&fp=chrome&type=tcp&flow=xtls-rprx-vision#vps"
echo "  （客户端用域名连接时 SNI 自动带域名，一般无需手填）"
echo ""
echo "Nginx: 仅 80 正常主页；443 由 Xray 直连"
echo "一键启动: cd $SCRIPT_DIR && ./start.sh"
echo "=========================================="
