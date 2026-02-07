# 🤖 TG-Grok-Bot

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Telegram](https://img.shields.io/badge/Bot-Telegram-blue.svg)](https://telegram.org/)
[![Model](https://img.shields.io/badge/Model-Grok--4.1--Fast-orange.svg)](https://x.ai/)
[![Docker](https://img.shields.io/badge/Docker-Ready-green.svg)](https://www.docker.com/)

一个功能强大的 Telegram 机器人，基于 **Grok-4.1-Fast** 模型，集成了多模态交互、高画质图片生成（支持 Yor Forger 专属场景）、语音输出及长期记忆功能。

---

## ✨ 核心特性

- **🚀 极速响应**：全局统一使用 `grok-4.1-fast`，提供更快的对话反馈。
- **🎨 意境绘图**：
  - `/img`：常规图片生成。
  - `/scene`：**精调场景绘图**，固定《间谍过家家》**约尔·福杰 (Yor Forger)** 形象，支持第一人称 POV 视角及 R18/NSFW 内容。
- **🎙️ 拟人语音**：集成阿里云 DashScope (CosyVoice/Qwen-TTS)，支持根据语境自动触发语音回复。
- **🧠 智能记忆**：基于 SQLite 的持久化记忆系统，自动生成会话摘要，像老朋友一样记住你的偏好。
- **👁️ 视觉增强**：集成 OCR 功能，发送图片即可识别文字并进行 AI 深度解读。
- **🎭 角色扮演**：支持自定义 Persona 人设，内置多种预设模式。

---

## 🛠️ 快速开始

### 方式一：Docker 部署 (推荐)

1. 克隆仓库：
   ```bash
   git clone https://github.com/Arturia169/tg-grok-bot.git
   cd tg-grok-bot
   ```
2. 配置环境变量：
   ```bash
   cp .env.example .env
   # 编辑 .env 文件，填写你的 API Keys
   ```
3. 启动服务：
   ```bash
   docker-compose up -d
   ```

### 方式二：本地运行

1. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
2. 运行机器人：
   ```bash
   python bot.py
   ```

---

## 🎮 命令指南

| 命令 | 说明 |
| :--- | :--- |
| `/ask` | 开启 AI 对话之旅 |
| `/img` | 唤醒生成式画笔 |
| `/scene` | 查看约尔的专属视角场景图 |
| `/model` | 查看/切换当前使用的模型 |
| `/persona`| 切换不同的人设风格 |
| `/voice` | 开启/关闭语音回复模式 |
| `/ocr` | 开启/关闭图片文字识别 |
| `/mem` | 查看机器人对你的长期记忆 |
| `/gf` | 切换“女友”对话模式 |

---

## ⚙️ 环境变量配置

在 `.env` 文件中，你可以自定义以下核心参数：

- `TELEGRAM_BOT_TOKEN`: 你的 Telegram Bot Token
- `OPENAI_BASE_URL`: OpenAI 兼容接口地址
- `OPENAI_API_KEY`: Grok API Key
- `DEFAULT_MODEL`: `grok-4.1-fast`
- `IMAGE_MODEL`: `grok-imagine-1.0`
- `TTS_ENABLED`: 是否启用语音功能 (0/1)
- `ALLOWED_CHAT_IDS`: 允许使用的用户/群组 ID (逗号分隔)

---

## 📁 项目结构

```text
├── bot.py           # 核心逻辑
├── personas/        # AI 人设配置文件
├── data/            # 存储 SQLite 数据库与缓存
├── Dockerfile       # 镜像构建文件
└── .env.example     # 环境配置模版
```

## 🤝 贡献与反馈

如果你有任何好的建议或发现了 Bug，欢迎提交 **Issue** 或 **Pull Request**！

---

> [!NOTE]
> 本项目仅供学习研究使用，建议在合规范围内调用 API 接口。

