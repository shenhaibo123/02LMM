#!/usr/bin/env bash
# 通过本机 443（或 8443 演示）代理用 curl 访问 Google，验证转发；在 mac/ 目录执行
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAC_VPS="$SCRIPT_DIR/mac-vps"
TMP="/tmp/xray-client-test-$$"
mkdir -p "$TMP"
cleanup() { kill "$XRAY_PID" 2>/dev/null; rm -rf "$TMP"; }
trap cleanup EXIT

UUID=$(python3 -c "import json; d=json.load(open('$MAC_VPS/config.json')); print(d['inbounds'][0]['settings']['clients'][0]['id'])" 2>/dev/null)
[[ -z "$UUID" ]] && UUID=$(grep -oE '"id":"[a-f0-9-]{36}"' "$MAC_VPS/config.json" | head -1 | sed 's/"id":"//;s/"$//')
[[ -z "$UUID" ]] && { echo "无法读取 UUID"; exit 1; }

PROXY_PORT="${PROXY_PORT:-443}"
if [[ -z "$SKIP_PORT_CHECK" ]]; then
  if ! lsof -i ":${PROXY_PORT}" 2>/dev/null | grep -q LISTEN; then
    echo "错误: 127.0.0.1:$PROXY_PORT 无进程监听。请先 ./start.sh 或 ./start-demo.sh"
    exit 1
  fi
fi

CERT_PEM="$MAC_VPS/certs/cert.pem"
PIN_SHA256=""
[[ -f "$CERT_PEM" ]] && PIN_SHA256=$(openssl x509 -in "$CERT_PEM" -noout -fingerprint -sha256 2>/dev/null | sed 's/.*=//;s/://g' | tr 'A-F' 'a-f')
TLS_EXTRA=""
[[ -n "$PIN_SHA256" ]] && TLS_EXTRA=", \"pinnedPeerCertSha256\": \"$PIN_SHA256\""

cat > "$TMP/client.json" << EOF
{
  "inbounds": [{"listen": "127.0.0.1", "port": 1080, "protocol": "socks", "settings": {"udp": true}}],
  "outbounds": [{
    "protocol": "vless",
    "settings": {
      "vnext": [{"address": "127.0.0.1", "port": $PROXY_PORT, "users": [{"id": "$UUID", "encryption": "none", "flow": "xtls-rprx-vision"}]}]
    },
    "streamSettings": {
      "network": "tcp",
      "security": "tls",
      "tlsSettings": {"serverName": "localhost", "fingerprint": "chrome"$TLS_EXTRA}
    }
  }]
}
EOF

XRAY=$(brew --prefix 2>/dev/null)/bin/xray
[[ -x "$XRAY" ]] || XRAY=xray
"$XRAY" run -c "$TMP/client.json" 2>"$TMP/client-err.log" & XRAY_PID=$!
sleep 2
if ! lsof -i :1080 2>/dev/null | grep -q LISTEN; then
  echo "客户端未监听 1080。错误："
  cat "$TMP/client-err.log" 2>/dev/null || true
  exit 1
fi
TEST_URL="https://www.google.com"
echo "通过代理 127.0.0.1:$PROXY_PORT 访问 $TEST_URL ..."
set +e
HTTP_CODE=$(curl -x socks5h://127.0.0.1:1080 -I --connect-timeout 10 -s -o /dev/null -w "%{http_code}" "$TEST_URL")
CURL_EXIT=$?
set -e
echo "HTTP $HTTP_CODE (curl exit: $CURL_EXIT)"
if [[ "$HTTP_CODE" = "200" ]]; then
  echo "代理链路验证成功"
else
  echo "代理访问失败或超时"
  cat "$TMP/client-err.log" 2>/dev/null || true
  [[ -f "$TMP/client-err.log" ]] && cp "$TMP/client-err.log" "$SCRIPT_DIR/client-err.log" 2>/dev/null
  exit 1
fi
