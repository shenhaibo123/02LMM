#!/usr/bin/env bash
# 校验本层目录的 Nginx（default）与 Xray（config.json）配置语法，不启动服务
# 用法: 在 linux/ 目录执行 ./validate.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
TMP="/tmp/v2ray-validate-$$"
mkdir -p "$TMP"
cleanup() { rm -rf "$TMP"; }
trap cleanup EXIT

echo "===== 校验 Nginx 配置（本层 default）====="
if ! command -v nginx &>/dev/null; then
  echo "  跳过：未安装 nginx（执行 install.sh 后可做完整校验）"
else
  mkdir -p "$TMP/letsencrypt/live/testdomain"
  touch "$TMP/letsencrypt/live/testdomain/fullchain.pem" "$TMP/letsencrypt/live/testdomain/privkey.pem"
  sed 's/YOUR_DOMAIN/testdomain/g' default | sed "s|/etc/letsencrypt|$TMP/letsencrypt|g" > "$TMP/default.conf"
  cat > "$TMP/nginx.conf" << EOF
error_log $TMP/nginx-error.log;
worker_processes 1;
events { worker_connections 10; }
http {
  access_log $TMP/nginx-access.log;
  include "$TMP/default.conf";
}
EOF
  if nginx -t -c "$TMP/nginx.conf" 2>&1; then
    echo "  Nginx 配置语法正确"
  else
    echo "  Nginx 配置有误，请检查 default"
    exit 1
  fi
fi

echo ""
echo "===== 校验 Xray 配置（本层 config.json）====="
XRAY=""
for cmd in xray /usr/local/bin/xray; do
  if command -v "$cmd" &>/dev/null; then XRAY="$cmd"; break; fi
done
if [[ -z "$XRAY" ]]; then
  echo "  跳过：未安装 xray（执行 install.sh 后可做完整校验）"
else
  TEST_UUID="a1b2c3d4-e5f6-4789-a012-000000000001"
  mkdir -p "$TMP/letsencrypt/live/testdomain"
  touch "$TMP/letsencrypt/live/testdomain/fullchain.pem" "$TMP/letsencrypt/live/testdomain/privkey.pem"
  sed -e "s/YOUR_UUID/$TEST_UUID/g" -e "s/YOUR_DOMAIN/testdomain/g" -e "s|/etc/letsencrypt|$TMP/letsencrypt|g" config.json > "$TMP/config.json"
  if "$XRAY" run -c "$TMP/config.json" -test 2>&1; then
    echo "  Xray 配置通过 -test"
  else
    code=0; timeout 2 "$XRAY" run -c "$TMP/config.json" 2>&1 || code=$?
    if [[ $code -eq 0 || $code -eq 124 ]]; then
      echo "  Xray 配置可加载"
    else
      echo "  Xray 配置可能有问题，请检查 config.json"
      exit 1
    fi
  fi
fi

echo ""
echo "===== 校验完成 ====="
