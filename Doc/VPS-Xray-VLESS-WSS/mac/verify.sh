#!/usr/bin/env bash
# 验证已运行的服务：80 主页、443 TLS；需先 start.sh 或 start-demo.sh；在 mac/ 目录执行
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
P80="${VERIFY_PORT_80:-80}"
P443="${VERIFY_PORT_443:-443}"

echo "===== 1. 验证 $P80 端口（主页）====="
HTTP_CODE=$(curl -s -o /tmp/verify-mac-80.html -w "%{http_code}" --connect-timeout 2 "http://localhost:$P80/" 2>/dev/null || echo "000")
if [[ "$HTTP_CODE" == "200" ]]; then
  echo "  $P80 端口: HTTP $HTTP_CODE OK"
  grep -q "姚明" /tmp/verify-mac-80.html 2>/dev/null && echo "  内容: 含「姚明」介绍页"
  open "http://localhost:$P80/" 2>/dev/null && echo "  已用浏览器打开"
else
  echo "  $P80 端口: 未响应 (code=$HTTP_CODE)，请先 ./start.sh 或 ./start-demo.sh"
fi

echo ""
echo "===== 2. 验证 $P443 端口（TLS 握手）====="
if (echo "Q" | timeout 3 openssl s_client -connect "localhost:$P443" -servername localhost -brief 2>/dev/null) | head -5; then
  echo "  $P443 端口: 可连接，TLS 握手正常（Xray 在监听）"
else
  echo "  $P443 端口: 连接超时或失败，请先 ./start.sh 或 ./start-demo.sh"
fi
echo ""
echo "===== 验证结束 ====="
