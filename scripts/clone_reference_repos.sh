#!/usr/bin/env bash
# 克隆参考项目到 reference/ 目录，不修改其内容，仅供对照与参考。
# 用法：在项目根目录执行 bash scripts/clone_reference_repos.sh

set -e
REF_DIR="${1:-reference}"
mkdir -p "$REF_DIR"
cd "$REF_DIR"

clone_if_missing() {
  local url="$1"
  local name="$2"
  if [ -d "$name" ]; then
    echo "[skip] $name 已存在"
  else
    echo "[clone] $url -> $name"
    git clone --depth 1 "$url" "$name"
  fi
}

clone_if_missing "https://github.com/jingyaogong/minimind.git" "minimind"
clone_if_missing "https://github.com/Morton-Li/QiChat.git" "QiChat"
clone_if_missing "https://github.com/EleutherAI/lm-evaluation-harness.git" "lm-evaluation-harness"

echo "参考仓库已就绪于 $REF_DIR/"
