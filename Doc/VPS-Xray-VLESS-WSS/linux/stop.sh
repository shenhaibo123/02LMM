#!/usr/bin/env bash
# 停止 Nginx 与 Xray
# 用法: ./stop.sh  或  sudo ./stop.sh

echo "停止 Nginx 与 Xray..."
systemctl stop nginx 2>/dev/null || true
systemctl stop xray  2>/dev/null || true
echo "已停止。"
