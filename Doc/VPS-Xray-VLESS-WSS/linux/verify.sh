#!/usr/bin/env bash
# 验证已运行的服务：80 主页是否可访问、443 端口 TLS 是否通（需先 start.sh）
# 用法: 在 linux/ 目录执行 ./verify.sh

set -e
P80="${VERIFY_PORT_80:-80}"
P443="${VERIFY_PORT_443:-443}"

# 未指定 VERIFY_SNI 时从 Xray 配置或证书目录推断证书域名（与 Mac 一致：用证书对应 SNI）
SNI="${VERIFY_SNI:-}"
if [[ -z "$SNI" ]]; then
  XRAY_CFG="/usr/local/etc/xray/config.json"
  if [[ -f "$XRAY_CFG" ]]; then
    SNI=$(grep -oE '/etc/letsencrypt/live/[^/]+/' "$XRAY_CFG" 2>/dev/null | head -1 | sed 's|/etc/letsencrypt/live/||;s|/||')
  fi
  [[ -z "$SNI" ]] && for d in /etc/letsencrypt/live/*/ 2>/dev/null; do [[ -f "${d}fullchain.pem" ]] && SNI=$(basename "$d") && break; done
  [[ -z "$SNI" ]] && SNI="localhost"
fi

echo "===== 1. 验证 $P80 端口（主页）====="
HTTP_CODE=$(curl -s -o /tmp/verify-80.html -w "%{http_code}" --connect-timeout 2 "http://127.0.0.1:$P80/" 2>/dev/null || echo "000")
if [[ "$HTTP_CODE" == "200" ]]; then
  echo "  $P80 端口: HTTP $HTTP_CODE OK"
  grep -q "姚明" /tmp/verify-80.html 2>/dev/null && echo "  内容: 含「姚明」介绍页"
else
  echo "  $P80 端口: 未响应或非 200 (code=$HTTP_CODE)，请先执行 ./start.sh"
fi

echo ""
echo "===== 2. 验证 $P443 端口（TLS 握手，SNI=$SNI）====="
if (echo "Q" | timeout 3 openssl s_client -connect "127.0.0.1:$P443" -servername "$SNI" -brief 2>/dev/null) | head -5; then
  echo "  $P443 端口: 可连接，TLS 握手正常（Xray 在监听）"
else
  echo "  $P443 端口: 连接超时或失败，请先执行 ./start.sh"
fi

echo ""
echo "===== 验证结束 ====="
