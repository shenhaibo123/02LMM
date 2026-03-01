#!/usr/bin/env bash
# 停止 Nginx 与 Xray；在 mac/ 目录执行
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAC_VPS="$SCRIPT_DIR/mac-vps"

sudo brew services stop nginx 2>/dev/null || true
[[ -f "$MAC_VPS/nginx.pid" ]] && sudo kill -9 "$(sudo cat "$MAC_VPS/nginx.pid" 2>/dev/null)" 2>/dev/null || true
[[ -f "$MAC_VPS/nginx-demo.pid" ]] && kill -9 "$(cat "$MAC_VPS/nginx-demo.pid" 2>/dev/null)" 2>/dev/null || true
rm -f "$MAC_VPS/nginx.pid" "$MAC_VPS/nginx-demo.pid"
sudo nginx -s stop -c "$MAC_VPS/nginx.conf" 2>/dev/null || true
nginx -s stop -c "$MAC_VPS/nginx-demo.conf" 2>/dev/null || true
sudo pkill -9 nginx 2>/dev/null || true
sleep 0.5
PID80=$(lsof -ti :80 2>/dev/null | tr '\n' ' ')
[[ -n "$PID80" ]] && sudo kill -9 $PID80 2>/dev/null && echo "已释放 80"
echo "已停止 Nginx"

[[ -f "$MAC_VPS/xray.pid" ]] && kill "$(cat "$MAC_VPS/xray.pid" 2>/dev/null)" 2>/dev/null || true
[[ -f "$MAC_VPS/xray-demo.pid" ]] && kill "$(cat "$MAC_VPS/xray-demo.pid" 2>/dev/null)" 2>/dev/null || true
rm -f "$MAC_VPS/xray.pid" "$MAC_VPS/xray-demo.pid"
pkill -f "xray run -c $MAC_VPS/config.json" 2>/dev/null || true
pkill -f "xray run -c $MAC_VPS/config-demo.json" 2>/dev/null || true
pkill -f "xray-server-debug.json" 2>/dev/null || true
PID443=$(lsof -ti :443 2>/dev/null | tr '\n' ' ')
[[ -n "$PID443" ]] && sudo kill -9 $PID443 2>/dev/null && echo "已释放 443"
echo "已停止 Xray"
echo "Mac 本地验证已停止"
