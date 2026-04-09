---
title: "WSL2 shell 使用优化 - 交互响应速度、代理配置"
categories: "code"
tags: ["wsl"]
id: "8483c569b3895079"
date: 2026-04-09 13:37:05
cover: "/assets/images/banner/7b1491d13dfb97a4.webp"
---

:::note
我承认，世界上最好的 linux 发行版，那就是 -- WSL
:::

## 0. 背景

本人虽然使用 Windows 电脑，但一直使用 WSL2 工作，所有项目代码等都在 WSL2 里，也更习惯于 linux 命令行操作。结合 vscode 的 remote 套件连接 WSL 进行开发非常方便。
但默认配置下在使用 shell ，可能会有代理配置难或代理无法生效、shell 响应速度较慢（敲个 `ls`，`cd ..`, `ip a` 体感上都要卡顿 0.5 ~ 1s）等问题，简直要怀疑人生。

如果你也有和我一样的感觉，看完本文，还你一个丝滑流畅的 WSL shell 体验！

## 1. 切换到 fish

本人原先也赶时髦使用 zsh + oh-my-zsh，但是其实日常使用中发现 zsh 启动和命令响应确实一般，尤其是每次打开终端都要加载一堆插件和配置，体验不佳。所以决定切换到 fish （虽然也有赶时髦因素），启动速度和命令响应速度明显提升，而且自带语法高亮和自动补全，非常顺手。

### 1.1 安装 fish 和 fisher

```bash
sudo apt-add-repository ppa:fish-shell/release-4
sudo apt update
sudo apt install fish

fish # 进入 fish

curl -sL https://raw.githubusercontent.com/jorgebucaran/fisher/main/functions/fisher.fish | source && fisher install jorgebucaran/fisher 
#（如果有代理问题，可以先看下一小节）
```

fisher 推荐安装插件：

```bash
fisher install jorgebucaran/autopair.fish # autopair 补全括号引号
fisher install franciscolourenco/done # 长时间任务完成系统提示
```

切换为默认 shell：

```bash
sudo chsh -s /usr/bin/fish
```

ok，干掉 oh-my-zsh 和 zsh 吧！

```bash
# 删除 oh-my-zsh 相关文件
rm -rf ~/.oh-my-zsh
# 删除 zsh 配置文件和历史记录
rm -f ~/.zshrc ~/.zshrc.pre-oh-my-zsh ~/.zsh_history ~/.zprofile

# 删除 powerlevel10k 主题的配置文件
rm -f ~/.p10k.zsh

# 删除 p10k 的缓存文件夹
rm -rf ~/.cache/p10k-*

# 彻底卸载 Zsh 及其系统级配置文件
sudo apt remove --purge zsh

# 清理不再需要的孤立依赖包
sudo apt autoremove
```

### 1.2 配置 fish 同时解决命令行使用 Windows 代理问题

fish 的配置文件是 `~/.config/fish/config.fish`
以下是我把原来 zsh 的配置扔给 gemini 返回的 fish 配置，朋友们可以按需修改

```bash
if status is-interactive
# Commands to run in interactive sessions can go here
end

# ==========================================
# 1. PATH 环境变量管理
# ==========================================
# fish_add_path 是更优雅且高性能的做法，它会自动去重并追加到全局 $PATH 前面
fish_add_path /usr/local/cuda-12.9/bin
fish_add_path /home/xxx/.local/share/pnpm
fish_add_path /home/xxx/.influxdb/
fish_add_path /usr/lib/ccache

# 我们排除掉绝大部分 Windows 挂载路径
set -gx PATH (string match -v '*/mnt/*' $PATH)

# 手动加回你需要的高频 Windows 工具,只要路径明确，不会造成全局扫描负担
set -p PATH "/mnt/d/Program Files/Microsoft VS Code/bin"
set -p PATH "/mnt/d/Program Files/cursor/resources/app/bin"

# ==========================================
# 2. 全局环境变量 （使用 set -gx 替代 export)
# ==========================================
set -gx PNPM_HOME /home/xxx/.local/share/pnpm

# 动态获取 CUDA 架构版本 (Fish 使用圆括号进行命令替换）
set -gx TORCH_CUDA_ARCH_LIST (nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)

# 注意：LD_LIBRARY_PATH 仍需保留传统拼接，因为 fish_add_path 仅针对 $PATH
if set -q LD_LIBRARY_PATH
    set -gx LD_LIBRARY_PATH /usr/local/cuda-12.9/lib64:$LD_LIBRARY_PATH
else
    set -gx LD_LIBRARY_PATH /usr/local/cuda-12.9/lib64
end

# ==========================================
# 3. 别名设定
# ==========================================
alias ll="ls -al"

# ==========================================
# 4. 自定义函数：WSL2 代理控制
# ==========================================
# Fish 中不使用大括号 {}，而是使用 function ... end 的块结构
function proxy --description "Enable terminal proxy for WSL"
    # 核心修复：直接读取真实的路由网关，避开 10.255.255.254 这个 WSL DNS 代理大坑
    set -l host_ip (ip route | grep default | awk '{print $3}')

    set -l port 10808

    # 应用代理
    set -gx ALL_PROXY "socks5://$host_ip:$port"
    set -gx all_proxy "socks5://$host_ip:$port"
    set -gx http_proxy "http://$host_ip:$port"
    set -gx https_proxy "http://$host_ip:$port"

    git config --global http.proxy "socks5://$host_ip:$port"
    git config --global https.proxy "socks5://$host_ip:$port"

    echo -e "Proxy ON: \033[32m$host_ip:$port\033[0m"

    # 测试连通性 (grep HTTP 可以同时兼容 HTTP/1.1 和 HTTP/2 的返回）
    echo "Testing connection to google.com..."
    if curl -I -s --connect-timeout 3 https://www.google.com | grep -q "HTTP"
        echo -e "\033[32m[SUCCESS] Proxy is working perfectly!\033[0m"
    else
        echo -e "\033[31m[FAILED] 代理仍然失败。\033[0m"
        echo "排查提示：你的 Windows 防火墙已关，请检查是否安装了【火绒】或【360】等第三方杀毒软件，它们会无视系统防火墙设置并静默拦截局域网请求！"
    end
end

function unproxy
    # set -e 用于删除变量 （相当于 unset)
    set -e ALL_PROXY
    set -e all_proxy
    set -e http_proxy
    set -e https_proxy
    
    git config --global --unset http.proxy
    git config --global --unset https.proxy
    
    echo "Terminal proxy deactivated"
end

# ==========================================
# 5. 加载外部脚本与虚拟环境
# ==========================================
# 去掉命令提示符前缀修改
set -gx VIRTUAL_ENV_DISABLE_PROMPT 1

# 注意 1：Fish 激活 Python 虚拟环境需要调用专属的 .fish 脚本
if test -f ~/.venv/bin/activate.fish
    source ~/.venv/bin/activate.fish
end

# 注意 2：如果 ~/.local/bin/env 是纯 Bash 语法 （含有 export 等），Fish 无法直接 source。
# 建议将其内容直接转写到 config.fish 中，或者使用标准的 `set` 语法重写该文件。
if test -f ~/.local/bin/env
    source ~/.local/bin/env.fish
end

# 启用 starship
starship init fish | source
# 启用 zoxide
zoxide init fish | source
```

**网络代理**
有的朋友可能会说直接使用 mirrored 网络就好了，然后配置 127.0.0.1 的代理端口即可。但本人时机使用体验，即使配置了 mirrored 网络，某些情况下，尤其是笔记本休眠，重新打开电脑，网络就有可能有问题。反正是挺玄学的。

最稳妥的还是使用 NAT 网络，然后通过 Windows 开启局域网代理，wsl 通过局域网 ip 访问代理。

- 在 Windows 上开启局域网代理（如 Clash、v2rayN 等），确保允许来自局域网的连接；
- 在 WSL 中获取 Windows 主机 IP（NAT 模式下），使用 ip route 获取 ip：具体见上面配置

### 1.3 安装和配置 starship（可选）

此步骤是命令行提示符美化/护眼需求，无此需求可直接跳过；

在Windows里下载并安装 Nerd Font 字体：<https://www.nerdfonts.com/font-downloads>

- 我选了 Proto Nerd Font
- Windows Terminal 配置 ubuntu 字体
  - 设置-左侧配置文件-ubuntu-右侧外观-字体：填入`0xProto Nerd Font Mono`

wsl 内 安装 starship：

```bash
curl -sS https://starship.rs/install.sh | sh

echo 'starship init fish | source' >> ~/.config/fish/config.fish
```

starship 配置：

```toml
# ~/.config/starship.toml

# 核心：极致流畅，不显示没用的模块
add_newline = false

# 左侧：精简的用户信息 + 目录 + Git
format = """
[ ](bold white)\
$username\
$hostname\
$directory\
$git_branch\
$git_status\
$character"""

# 右侧：Python（仅在环境内） + 命令耗时 + 时间
right_format = """
$python\
$cmd_duration\
$time\
[ ](bold white)"""

[username]
style_user = "green bold"
format = "[$user]($style)"
show_always = true

[hostname]
ssh_only = false
style = "cyan bold"
format = "[@$hostname]($style) "

[directory]
style = "blue bold"
format = "[$path]($style) "
truncation_length = 3
fish_style_pwd_dir_length = 1 # 路径太长时会自动缩写

[python]
symbol = " "
format = '[${symbol}py${version}](cyan) '
detect_extensions = [] # 不根据文件后缀探测（提升性能）
detect_files = []      # 不根据文件探测（提升性能）

[git_branch]
symbol = " "
style = "magenta bold"
format = "[$symbol$branch]($style) "

[git_status]
style = "red bold"
# 既然追求高性能，我们可以让 Git 状态显示得更直接
format = '([$all_status$ahead_behind]($style) )'

[character]
# 既然用 0xProto，用这个 Lambda 符号非常对味
success_symbol = "[λ](bold green)"
error_symbol = "[λ](bold red)"

[cmd_duration]
min_time = 500 # 只有命令超过 0.5s 才显示耗时，避免干扰
format = "[took $duration]($style) "
style = "yellow bold"

[time]
disabled = false
style = "#666666"
format = "[$time]($style)"
time_format = "%T"
```

重开一个 shell 窗口体验一下

## 2. 速度优化

wsl shell 交互响应慢来自于两个方面：

- 95%概率的绝对元凶 - Windows defender 的安全扫描：你每敲一个命令都会被 Windows defender 拦截检查风险，然后放行，这里系统开销简直爆炸
  - 解决方法：去 `Windows defender-病毒和威胁防护-管理设置-排除项-添加排除项`，把 wsl 的 vhdx 文件直接加入排除项
  - 我这里因为安装在 D 盘了，朋友们按需调整
- shell 插件过多，或插件逻辑复杂
  - 卸载掉无用或者不常用的插件
  - 如果你是 fish 用户，检查一下你的 shell 插件，用 fisher 安装过 `jethrokuan/z`（纯 fish 脚本实现）
    - 强烈建议卸载，这玩意的逻辑简直天才才写得出来，它会
      - 调用 mktemp 创建临时文件；调用 date 获取时间戳；最离谱的是：它每次 cd 都会调用一次 awk 去扫描并重写整个数据库文件，然后再用 mv 覆盖回去。
        - 这里每次都会导致一坨 fork/exec 进程调用开销，在 Linux 原生内核里，fork() 很轻。但在 WSL2 中，由于虚拟化层对指令的拦截和 Windows Defender 的监控，每次执行 awk 或 mktemp 都要经历一次昂贵的上下文切换。cd 一次，它连着触发 4-5 个进程调用，开销爆炸。
    - 解决：卸载 `jethrokuan/z`，如果有需求用 `zoxide` 替代，性能和体验都更好（ zsh 的 z 插件应该也是类似的，一样建议卸载）
- PATH清理：
  - WSL2 默认把 Windows 的 PATH 全部塞进 Linux，导致 Shell 在查找命令时跨文件系统扫描，这也是 ls 等命令卡顿的元凶之一：
    - 解决方法：移除/mnt路径，但我保留了code和cursor
  - **避坑**：config.fish 中手动清洗 PATH 时，建议直接使用 set -p PATH 来添加 Windows 路径，而不是使用 fish_add_path。因为后者带有路径去重和持久化缓存逻辑，在手动重置 PATH 的脚本中可能会因为幂等性判断导致添加失败。

```bash
fisher remove jethrokuan/z

curl -sS https://raw.githubusercontent.com/ajeetdsouza/zoxide/main/install.sh | bash

zoxide init fish | source

fisher install ajeetdsouza/zoxide.fish
```

**总结**：在 WSL2 中，所有涉及 cd 或 prompt 触发的逻辑，必须尽可能避免 Fork 外部进程。能用 Rust/C++ 二进制解决的，绝不用 Shell 脚本。能减少 I/O 往返的，绝不读写磁盘。

**如果配置完还是慢？**
请检查你的 ~/.config/fish/conf.d/ 目录。很多老插件（如 OMF）即使卸载了也会在这里留下残余的 .fish 脚本，每次启动都会被静默加载。如果遇到不明卡顿，运行 fish --profile /tmp/fish.prof -c "exit" 抓一下耗时排名，真相就在里面。

## 3. 结束

经过以上配置，你的 WSL2 shell 应该已经非常丝滑了，命令响应速度大幅提升，代理问题也彻底得到解决。

以上

如有错误，欢迎指正，感谢~

本文首发于 <https://www.wingedge777.com>, 可以随意转载
