#!/usr/bin/env python3
"""
检查本机 Python 版本是否 >= 3.10。若否，打印升级说明并退出码 1。
用于：创建 venv 前建议先运行此脚本，确保本机已安装 3.10+。

用法:
  python3 scripts/ensure_python.py
  python3.10 scripts/ensure_python.py
"""
import sys


def main():
    if sys.version_info >= (3, 10):
        print(f"Python {sys.version_info.major}.{sys.version_info.minor} 符合要求 (>= 3.10)")
        return 0
    print("本项目依赖 Python 3.10+（例如 requirements 中 matplotlib==3.10.0）。")
    print(f"当前: Python {sys.version_info.major}.{sys.version_info.minor}")
    print("\n本机升级建议（macOS / Linux）：")
    print("  # Homebrew（macOS/Linux）")
    print("  brew install python@3.10")
    print("  # 创建 venv 时指定：")
    print("  python3.10 -m venv .venv")
    print("  source .venv/bin/activate")
    print("\n  # 或 pyenv")
    print("  pyenv install 3.10.16")
    print("  pyenv local 3.10.16")
    print("  python -m venv .venv && source .venv/bin/activate")
    return 1


if __name__ == "__main__":
    sys.exit(main())
