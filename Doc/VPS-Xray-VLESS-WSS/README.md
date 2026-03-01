# VPS-Xray-VLESS-WSS

在 VPS（Linux）或本机（Mac）上部署 **80 端口 Nginx 主页 + 443 端口 Xray 直连**（VLESS+TCP+TLS+XTLS-Vision，无 path）。  
根目录仅本说明；**Linux 与 Mac 各占一层目录，同级、自包含**。

---

## 使用说明（VPS 从零到可用）

按下面顺序做，**只需在安装时填写一个域名**，其余由一键脚本完成（证书、UUID、配置、启动）。

1. **先申请域名**  
   在任意域名服务商购买域名（如 Cloudflare、Namecheap、阿里云、腾讯云等）。记下你要用来做代理的域名，例如 `proxy.example.com`。

2. **再购买 VPS**  
   在 Vultr、Hostwinds、DigitalOcean 等购买一台 Linux VPS（推荐 Ubuntu/Debian），记下分配到的 **公网 IP**。

3. **绑定域名和 VPS**  
   在 **域名服务商的控制台** 里，为该域名添加 **A 记录**：  
   - 主机/主机记录：你要用的子域名（如 `proxy`，或 `@` 表示根域名）  
   - 记录类型：**A**  
   - 记录值/指向：填 **VPS 的公网 IP**  
   保存后等待 DNS 生效（几分钟到几十分钟不等）。可用 `ping 你的域名` 检查是否已解析到该 IP。

4. **把本文件夹放到 VPS 上**  
   - 用 `git clone` 拉取本仓库后进入 `Doc/VPS-Xray-VLESS-WSS/`，或  
   - 把整个 `VPS-Xray-VLESS-WSS` 文件夹用 scp/rsync 上传到 VPS 的任意目录。

5. **在 VPS 上一键安装并启动**  
   在 VPS 上执行（把 `<域名>` 换成你在第 1 步准备的域名，如 `proxy.example.com`）：
   ```bash
   cd VPS-Xray-VLESS-WSS/linux
   sudo ./install.sh <域名> [邮箱]
   ```
   - **域名**：必填，且必须已在第 3 步解析到本机公网 IP（脚本会用该域名申请 Let's Encrypt 证书）。  
   - **邮箱**：可选，用于 Let's Encrypt 到期提醒。  
   脚本会自动：安装 Nginx、Certbot、Xray → 用域名申请证书 → 生成 UUID → 写入 Nginx/Xray 配置 → 启动并设置开机自启。**无需再改其它参数**。

6. **安装完成即已后台运行**  
   安装脚本结束时会打印：域名、UUID、**vless:// 链接**、客户端填写说明。此时 80 主页和 443 代理已在运行。  
   之后若重启 VPS，服务会开机自启；也可在 `linux/` 目录下手动执行 `./start.sh` 启动、`./stop.sh` 停止。

7. **客户端按说明填写**  
   安装/启动时会打印 **vless://** 链接，可复制到 v2rayN/v2rayNG 等导入。或手动填写：  
   - 协议：VLESS，地址：**你的域名**，端口：443，开启 TLS  
   - 传输：tcp，flow：xtls-rprx-vision，SNI：**你的域名**（与地址一致，用域名连接时一般无需手填）  

这样 VPS 即可正常提供 80 网页与 443 代理；**唯一需要你提前准备并填写的就是「域名」**（且域名已解析到该 VPS）。

---

## 目录结构

```
VPS-Xray-VLESS-WSS/
├── README.md          # 本文件（根目录仅此）
├── linux/              # Linux（VPS）本层目录：default、config.json、www、脚本
│   ├── README.md
│   ├── default, config.json, www/, nginx.conf
│   ├── install.sh, start.sh, stop.sh
│   ├── validate.sh, verify.sh, test-proxy-443.sh
│   └── ...
├── mac/                # Mac 本层目录：同上 + start-demo.sh，运行时生成 mac-vps/
│   ├── README.md
│   ├── default, config.json, www/, nginx.conf
│   ├── install.sh, start.sh, stop.sh, start-demo.sh
│   ├── validate.sh, verify.sh, test-proxy-443.sh
│   └── mac-vps/        # 安装/启动后生成
└── FILES.md            # 文件与脚本关系说明（可选）
```

- **根目录**：只有本 README，不从根目录读配置。
- **linux/** 与 **mac/**：各自本层目录内包含 `default`、`config.json`、`www/`，所有脚本（install、start、stop、validate、verify、test）均在本层执行、从本层读取上述文件。

---

## validate 与 verify 的关系

| 名称 | 作用 | 何时用 |
|------|------|--------|
| **validate** | **校验配置文件语法**，不启动任何服务。对当前目录下的 `default`（Nginx）和 `config.json`（Xray）做语法检查（如 `nginx -t`、`xray run -c ... -test`）。 | 改完配置后、或部署前，确认本层配置合法。 |
| **verify** | **验证已运行的服务**。需要先执行 start（或 start-demo），再运行 verify：检查 80 主页是否返回 200、443 端口 TLS 握手是否成功；Mac 版还会打开浏览器。 | 启动服务后，确认 80/443 已正常监听并可访问。 |

总结：**validate = 检查“配置对不对”；verify = 检查“服务跑起来后通不通”。**

---

## 使用方式

- **Linux（VPS）**：按上文「使用说明」完成域名解析后，进入 `linux/`，执行 `sudo ./install.sh <域名> [邮箱]` 即可完成安装与启动；详见 [linux/README.md](linux/README.md)。日常启停：`./start.sh` / `./stop.sh`。
- **Mac（本机验证）**：进入 `mac/`，按 [mac/README.md](mac/README.md) 执行。一键安装：`./install.sh`；一键启动：`./start.sh`；演示模式（无需 sudo）：`./start-demo.sh`。

两套脚本与模板互不依赖根目录，仅各自本层目录。
