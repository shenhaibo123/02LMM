# Linux（VPS）部署

与 **mac/** 同级：80 端口 Nginx 主页，443 端口 Xray 直连（VLESS+TCP+TLS+XTLS-Vision）。**本层目录**自包含：`default`、`config.json`、`www/` 与所有脚本均在本目录，脚本从本层读取配置。

## 功能清单（与 Mac 对齐，均已实现）

| 功能 | 说明 | 验证方式 |
|------|------|----------|
| 一键安装 | `sudo ./install.sh <域名> [邮箱]`：装依赖、Let's Encrypt 证书、UUID、写配置、启动并开机自启 | 安装结束会打印 vless:// 链接；访问 http://你的域名、客户端用域名连接 |
| 一键启动/停止 | `./start.sh` / `./stop.sh`：启停 Nginx + Xray（systemd） | start 后打印域名、UUID、vless:// 链接；stop 后服务停止 |
| 配置语法校验 | `./validate.sh`：本层 `default`、`config.json` 语法检查，不启动服务 | 执行通过即配置合法 |
| 运行中验证 | `./verify.sh`：检查 80 主页、443 TLS（自动用证书域名做 SNI） | 需先 start，再执行 |
| 443 转发测试 | `./test-proxy-443.sh`：本机起客户端经 443 访问 Google | 需先 install + start |
| vless:// 链接 | install/start 结束会打印可导入客户端的链接（域名+443） | 客户端用域名连接时 SNI 自动带域名，无需手填 |

**与 Mac 的差异**：Linux 使用**域名 + Let's Encrypt 证书**，客户端填域名即可，SNI 与证书一致；Mac 为本地验证用自签 localhost 证书，手机用 IP 连时须在客户端填 SNI=localhost、允许不安全证书。

## 一键安装（首次）

需 root，域名已解析到本机公网 IP。**UUID 自动生成**，无需手填。

```bash
sudo ./install.sh <域名> [邮箱]
# 示例: sudo ./install.sh proxy.example.com me@example.com
```

- **域名**：在域名商处已做 A 记录解析到本机公网 IP。
- **邮箱**：可选，用于 Let's Encrypt 到期提醒。

脚本会：安装 Nginx、Certbot、Xray → 申请证书 → **自动生成 UUID** → 写配置并启动、开机自启。

## 一键启动（安装后）

```bash
./start.sh
# 或 sudo ./start.sh
```

启动 Nginx(80) 与 Xray(443)，并打印客户端参数与 **vless://** 链接（可直接导入 v2rayN/v2rayNG 等）。

## 停止

```bash
./stop.sh
# 或 sudo ./stop.sh
```

## 测试 443 转发

确认本机 443 的 Xray 是否正确转发（经代理访问 Google）：

```bash
./test-proxy-443.sh
# 或 sudo ./test-proxy-443.sh
```

成功输出：`HTTP 200`、`本机 443 Xray 转发验证成功`。

## validate 与 verify 区别

| 脚本 | 作用 |
|------|------|
| **validate.sh** | **校验配置语法**，不启动服务。检查本层 `default`（Nginx）和 `config.json`（Xray）是否合法（`nginx -t`、`xray run -c ... -test`）。 |
| **verify.sh** | **验证已运行的服务**。需先执行 `start.sh`，再运行本脚本：检查 80 主页是否返回 200、443 端口 TLS 握手是否成功。 |

## 文件说明

| 文件 | 作用 |
|------|------|
| `default` | Nginx 站点模板（仅 80） |
| `config.json` | Xray 配置模板 |
| `www/` | 本地主页源码 |
| `install.sh` | 一键安装（依赖、证书、UUID、配置、启动） |
| `start.sh` | 一键启动 Nginx + Xray，并打印客户端信息 |
| `stop.sh` | 停止 Nginx 与 Xray |
| `validate.sh` | 校验本层配置语法（不启动服务） |
| `verify.sh` | 验证已运行服务的 80/443（需先 start） |
| `test-proxy-443.sh` | 测试本机 443 转发（curl 经代理访问 Google） |
