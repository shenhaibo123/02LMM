---
name: clash-linux-proxy
description: 在 Linux 上使用 Clash 配置（YAML）进行代理。当用户询问在 Linux 用 Clash、Clash Verge、mihomo、代理配置、VPN 配置、7890 端口时使用。
---

# Linux 上使用 Clash 配置

本 skill 记录如何在 Linux 上使用已有的 Clash YAML 配置（如从 Clash Verge 导出的配置）进行代理，便于下次直接按步骤操作。

## 配置与端口说明

- **配置格式**：Clash / Clash Verge 的 YAML（含 `proxies`、`proxy-groups`、`rules`、可选 `rule-providers`）。
- **常用本地端口**（以配置中为准，典型默认）：
  - HTTP 代理：`127.0.0.1:7890`
  - SOCKS5：`127.0.0.1:7891`
  - 混合端口（HTTP+SOCKS）：`7892`
- **控制 API**：部分客户端提供 `127.0.0.1:9090` 用于面板或外部控制。

---

## 方案一：Clash Verge Rev（图形界面，推荐）

适合带桌面的 Linux，与 macOS/Windows 使用方式一致。

### 安装

1. 打开 [Clash Verge Rev Releases](https://github.com/clash-verge-rev/clash-verge-rev/releases)。
2. 下载 Linux 版本：
   - 通用：`.AppImage`，赋予执行权限后直接运行。
   - 或按发行版选择：`.deb`（Debian/Ubuntu）、`.rpm` 等。

```bash
# 示例：AppImage
chmod +x Clash.Verge.Revision.*.AppImage
./Clash.Verge.Revision.*.AppImage
```

### 导入配置

1. 将已有的 Clash 配置文件（如 `xxx.yaml`）复制到本机。
2. 在 Clash Verge Rev 中：**配置 / Profiles** → **导入** 或 **新建**，选择该 YAML 文件。
3. 将该配置设为**当前使用**的配置。

### 使用代理

- 在设置中开启 **系统代理**，或
- 手动设置系统/浏览器代理：
  - HTTP/HTTPS：`127.0.0.1:7890`
  - SOCKS5：`127.0.0.1:7891`

---

## 方案二：mihomo 命令行（无图形界面）

适合服务器、无桌面环境或只用终端的场景。

### 安装 mihomo

mihomo 是 Clash 内核的兼容实现，支持标准 Clash 配置。

1. 打开 [mihomo GitHub Releases](https://github.com/MetaCubeX/mihomo/releases)。
2. 下载对应架构的 Linux 版本，例如：
   - `mihomo-linux-amd64-v*.gz`（x86_64）
   - `mihomo-linux-arm64-v*.gz`（ARM64）

```bash
# 示例：amd64，版本号以实际为准
wget https://github.com/MetaCubeX/mihomo/releases/download/v1.18.10/mihomo-linux-amd64-v1.18.10.gz
gunzip mihomo-linux-amd64-v*.gz
chmod +x mihomo-linux-amd64-*
sudo mv mihomo-linux-amd64-* /usr/local/bin/mihomo
```

### 放置配置

1. 将 Clash 的 YAML 配置文件放到本机，例如：`~/.config/mihomo/config.yaml`。
2. 若配置中使用 `path: "./ruleset/xxx.yaml"` 等相对路径，需保证运行时的当前目录或路径一致；也可在 YAML 中改为绝对路径，或把规则集放到同一目录。

### 启动

```bash
# 前台运行
mihomo -f ~/.config/mihomo/config.yaml

# 或指定目录（便于解析相对路径）
mihomo -f ~/.config/mihomo/config.yaml -d ~/.config/mihomo
```

### 开机自启（systemd）

1. 创建 service 文件，例如 `/etc/systemd/system/mihomo.service`：

```ini
[Unit]
Description=Mihomo Clash Proxy
After=network.target

[Service]
Type=simple
User=你的用户名
ExecStart=/usr/local/bin/mihomo -f /home/你的用户名/.config/mihomo/config.yaml -d /home/你的用户名/.config/mihomo
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

2. 启用并启动：

```bash
sudo systemctl daemon-reload
sudo systemctl enable mihomo
sudo systemctl start mihomo
sudo systemctl status mihomo
```

### 终端使用代理

```bash
# 当前 shell 生效
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export all_proxy=socks5://127.0.0.1:7891

# 仅当前命令
http_proxy=http://127.0.0.1:7890 https_proxy=http://127.0.0.1:7890 curl -I https://www.google.com
```

---

## 故障排查

| 现象 | 可能原因 | 处理 |
|------|----------|------|
| 规则集下载失败、启动报错 | 配置中 `rule-providers` 使用 HTTP URL（如 jsdelivr），本机网络无法访问 | 在能直连的环境先运行一次拉取规则集；或把规则集改为本地文件 |
| 无代理效果 | 未设置系统/应用代理或环境变量 | 确认代理指向 `127.0.0.1:7890`（HTTP）或 `7891`（SOCKS5） |
| 端口被占用 | 7890/7891 已被其他程序占用 | 修改 YAML 中 `port`、`socks-port`，或关闭占用端口的进程 |
| 节点不可用 | 服务器、订阅或网络问题 | 在 Clash Verge 或其它设备上先测同一配置是否可用 |

---

## 小结

- **有桌面**：用 Clash Verge Rev，导入 YAML → 选为当前配置 → 开系统代理。
- **无桌面/服务器**：用 mihomo，放好 `config.yaml` → 运行 `mihomo -f config.yaml` → 设置 `http_proxy`/`https_proxy` 或应用代理为 `127.0.0.1:7890`。

配置中的 **UUID、服务器地址、端口** 属于敏感信息，不要提交到公开仓库或截图外发。
