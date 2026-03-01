#!/usr/bin/env bash
# 测试本机 443 的 Xray 是否正确转发（与 Mac 的 test-proxy-443.sh 同逻辑）
# 用法: ./test-proxy-443.sh  或  sudo ./test-proxy-443.sh
# 依赖: 已执行 install.sh，Xray 在 443 监听

set -e
TMP="/tmp/xray-client-test-$$"
mkdir -p "$TMP"
cleanup() { kill "$XRAY_PID" 2>/dev/null; rm -rf "$TMP"; }
trap cleanup EXIT

XRAY_CFG="/usr/local/etc/xray/config.json"
[[ -f "$XRAY_CFG" ]] || { echo "未找到 $XRAY_CFG，请先执行 linux/install.sh"; exit 1; }

UUID=$(python3 -c "import json; d=json.load(open('$XRAY_CFG')); print(d['inbounds'][0]['settings']['clients'][0]['id'])" 2>/dev/null)
[[ -z "$UUID" ]] && UUID=$(grep -oE '"id":"[a-f0-9-]{36}"' "$XRAY_CFG" | head -1 | sed 's/"id":"//;s/"$//')
[[ -z "$UUID" ]] && { echo "无法读取 UUID"; exit 1; }

DOMAIN=$(grep -oE '/etc/letsencrypt/live/[^/]+/' "$XRAY_CFG" 2>/dev/null | head -1 | sed 's|/etc/letsencrypt/live/||;s|/||')
if [[ -z "$DOMAIN" ]]; then
  for d in /etc/letsencrypt/live/*/; do
    [[ -f "${d}fullchain.pem" ]] && DOMAIN=$(basename "$d") && break
  done
fi
[[ -z "$DOMAIN" ]] && { echo "无法确定域名（检查 /usr/local/etc/xray/config.json 或 /etc/letsencrypt/live/）"; exit 1; }

CERT_PEM="/etc/letsencrypt/live/$DOMAIN/fullchain.pem"
PIN_SHA256=""
[[ -f "$CERT_PEM" ]] && PIN_SHA256=$(openssl x509 -in "$CERT_PEM" -noout -fingerprint -sha256 2>/dev/null | sed 's/.*=//;s/://g' | tr 'A-F' 'a-f')
TLS_EXTRA=""
[[ -n "$PIN_SHA256" ]] && TLS_EXTRA=", \"pinnedPeerCertSha256\": \"$PIN_SHA256\""

if ! (ss -tlnp 2>/dev/null | grep -q ':443 ') && ! (netstat -tlnp 2>/dev/null | grep -q ':443 ') && ! (lsof -i :443 2>/dev/null | grep -q LISTEN); then
  echo "错误: 本机 443 无进程监听。请先执行 install.sh 并运行 ./start.sh"
  exit 1
fi

cat > "$TMP/client.json" << EOF
{
  "inbounds": [{"listen": "127.0.0.1", "port": 1080, "protocol": "socks", "settings": {"udp": true}}],
  "outbounds": [{
    "protocol": "vless",
    "settings": {
      "vnext": [{"address": "127.0.0.1", "port": 443, "users": [{"id": "$UUID", "encryption": "none", "flow": "xtls-rprx-vision"}]}]
    },
    "streamSettings": {
      "network": "tcp",
      "security": "tls",
      "tlsSettings": {"serverName": "$DOMAIN", "fingerprint": "chrome"$TLS_EXTRA}
    }
  }]
}
EOF

XRAY="/usr/local/bin/xray"
[[ -x "$XRAY" ]] || { echo "未找到 $XRAY，请先执行 install.sh"; exit 1; }

"$XRAY" run -c "$TMP/client.json" 2>"$TMP/client-err.log" & XRAY_PID=$!
sleep 2
if ! ss -tlnp 2>/dev/null | grep -q ':1080 '; then
  if ! netstat -tlnp 2>/dev/null | grep -q ':1080 '; then
    echo "客户端未监听 1080，可能连 443 失败。错误输出："
    cat "$TMP/client-err.log" 2>/dev/null || true
    exit 1
  fi
fi

TEST_URL="https://www.google.com"
echo "通过本机 127.0.0.1:443 代理访问 $TEST_URL ..."
set +e
HTTP_CODE=$(curl -x socks5h://127.0.0.1:1080 -I --connect-timeout 10 -s -o /dev/null -w "%{http_code}" "$TEST_URL")
CURL_EXIT=$?
set -e
echo "HTTP $HTTP_CODE (curl exit: $CURL_EXIT)"
if [[ "$HTTP_CODE" = "200" ]]; then
  echo "本机 443 Xray 转发验证成功"
else
  echo "代理访问失败或超时"
  cat "$TMP/client-err.log" 2>/dev/null || true
  exit 1
fi
