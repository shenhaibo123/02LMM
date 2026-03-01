# Mac 本地验证

与 **linux/** 同级：80 端口 Nginx 主页，443 端口 Xray 直连（VLESS+TCP+TLS+XTLS-Vision）。**本层目录**自包含：`default`、`config.json`、`www/` 与所有脚本均在本目录，脚本从本层读取；运行时生成 `mac-vps/`（证书、配置、日志）。

## 功能清单（均已实现并可验证）

| 功能 | 说明 | 验证方式 |
|------|------|----------|
| 一键安装 | `./install.sh`：装依赖、生成自签证书与 UUID、写配置、启动 80/443 | 安装后访问 http://localhost、用 v2rayNG 连局域网 IP:443（SNI=localhost、允许不安全） |
| 一键启动/停止 | `./start.sh` / `./stop.sh`：启停 Nginx + Xray，释放 80/443 | 执行后看输出链接与端口；stop 后 80/443 释放 |
| 演示模式 | `./start-demo.sh`：8080/8443 无需 sudo，自动跑 test + verify | 不占 80/443，脚本内会调 test-proxy-443 与 verify |
| 配置语法校验 | `./validate.sh`：本层 `default`、`config.json` 语法检查，不启动服务 | 执行通过即 Nginx/Xray 配置合法（Nginx 可能有一条 error log 权限告警，可忽略） |
| 运行中验证 | `./verify.sh`：检查 80 主页、443 TLS 握手，可选打开浏览器 | 需先 start 或 start-demo，再执行 |
| 443 转发测试 | `./test-proxy-443.sh`：本机起客户端经 443 访问 Google | 需先 start；演示模式用 `PROXY_PORT=8443 ./test-proxy-443.sh` |
| vless:// 链接输出 | start/install/start-demo 结束会打印可导入 v2rayNG 的链接 | 输出中含 UUID、IP、端口与参数 |

**说明**：Mac 版为**本地验证/自用**用途，**不提供开机自启**（无类似 Linux 的 systemd）。若需开机自启，可自行用 launchd 包装 `start.sh`。

## validate 与 verify 区别

| 脚本 | 作用 |
|------|------|
| **validate.sh** | **校验配置语法**，不启动服务。检查本层 `default`（Nginx）和 `config.json`（Xray）是否合法。 |
| **verify.sh** | **验证已运行的服务**。需先执行 `start.sh` 或 `start-demo.sh`，再运行：检查 80 主页、443 TLS 握手，并可打开浏览器。 |

## 一键安装（首次）

```bash
cd mac
./install.sh
```

需 Homebrew；80/443 需 sudo。UUID 自动生成。

## 一键启动 / 停止

```bash
./start.sh
# 停止
./stop.sh
```

## 演示模式（无需 sudo，8080/8443）

```bash
./start-demo.sh
```

会自动执行 test-proxy-443 与 verify。`start.sh` / `install.sh` / `start-demo.sh` 会输出 **vless://** 链接（替换好 IP 与 UUID），可直接复制到 v2rayNG 导入。

## 手机 v2rayNG 能打开网页但代理失败？

80 网页走的是 Nginx，手机能访问说明局域网通。代理走的是 **443 的 Xray + TLS**，服务端证书是自签且只签了 **localhost**（没有你的局域网 IP）。Xray 开了 `rejectUnknownSni`，只认 TLS 里的 **SNI=localhost**。

- **在 v2rayNG 里**：地址填 Mac 的局域网 IP（如 192.168.43.8），端口 443（演示模式填 8443）。  
- **必须**：把 **SNI / 服务器名称 / TLS 服务器名** 设为 **localhost**，并**勾选「允许不安全证书」**。  
这样握手时带 SNI=localhost，服务端才会接受并出示 localhost 证书，代理才能通。

**为什么正常 VPN（用域名+正式证书）不用填 SNI？** 因为这类代理使用**公网域名 + 正式证书**（如 Let's Encrypt）：客户端填的地址就是域名，TLS 握手时 SNI 会自动带该域名，证书也是为该域名签发的，所以无需手填。本方案 Mac 端用的是**自签证书且只签了 localhost**，用 IP 连时必须手动把 SNI 设为 localhost 才能通过校验。

## 测试 443 转发

```bash
./test-proxy-443.sh
# 演示模式时：PROXY_PORT=8443 ./test-proxy-443.sh
```

## 校验配置（不启动服务）

```bash
./validate.sh
```

若出现 `could not open error log file ... Permission denied` 的告警，属 nginx 默认路径权限问题，**不影响**语法校验结果（以 “syntax is ok / Configuration OK” 为准）。

## 验证已运行服务（80/443）

```bash
./verify.sh
# 演示模式后：VERIFY_PORT_80=8080 VERIFY_PORT_443=8443 ./verify.sh
```

## 本层文件

| 文件 | 作用 |
|------|------|
| `default` | Nginx 站点模板（仅 80） |
| `config.json` | Xray 配置模板 |
| `www/` | 本地主页源码 |
| `install.sh` | 一键安装 |
| `start.sh` | 一键启动 |
| `stop.sh` | 停止 |
| `start-demo.sh` | 演示模式 8080/8443 |
| `validate.sh` | 校验配置语法 |
| `verify.sh` | 验证已运行服务 |
| `test-proxy-443.sh` | 测试 443 转发 |
| `mac-vps/` | 运行时目录（install 后生成） |
