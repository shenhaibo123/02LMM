# 文件与脚本关系说明

根目录仅 **README.md**；**linux/** 与 **mac/** 为同级本层目录，各自包含 default、config.json、www、以及全部脚本，**均从本层读取**，不依赖上级目录。

---

## 目录结构

```
VPS-Xray-VLESS-WSS/
├── README.md           # 总览、validate vs verify 说明、入口链接
├── FILES.md            # 本文件
│
├── linux/              # Linux（VPS）本层
│   ├── README.md
│   ├── default, config.json, www/, nginx.conf   # 本层模板
│   ├── install.sh, start.sh, stop.sh
│   ├── validate.sh, verify.sh, test-proxy-443.sh
│   └── ...
│
└── mac/                # Mac 本层
    ├── README.md
    ├── default, config.json, www/, nginx.conf   # 本层模板
    ├── install.sh, start.sh, stop.sh, start-demo.sh
    ├── validate.sh, verify.sh, test-proxy-443.sh
    └── mac-vps/        # 运行时生成（证书、配置、日志）
```

---

## validate 与 verify

| 脚本 | 作用 |
|------|------|
| **validate.sh** | 在本层目录执行，校验本层 `default`、`config.json` 的**语法**（不启动服务）。 |
| **verify.sh** | 在本层目录执行，验证**已运行**的 80/443 服务是否可访问（需先 start 或 start-demo）。 |

---

## 脚本对照（本层执行、本层读取）

| 目的 | linux/ | mac/ |
|------|--------|------|
| 安装/初始化 | `install.sh` | `install.sh` |
| 一键启动 | `start.sh` | `start.sh` |
| 停止 | `stop.sh` | `stop.sh` |
| 演示模式 | — | `start-demo.sh` |
| 校验配置语法 | `validate.sh` | `validate.sh` |
| 验证已运行服务 | `verify.sh` | `verify.sh` |
| 测试 443 转发 | `test-proxy-443.sh` | `test-proxy-443.sh` |
