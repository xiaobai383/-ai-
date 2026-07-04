# 个人 AI 工作流助手 v1.1

> 云端智力，本地边界 — 豆包式会话 · 可控上传 · 向量检索 · 模型兜底

一个面向个人知识和工作文档的 AI 工作流助手。使用云端大模型保证智能效果，同时通过本地预处理、敏感信息脱敏、会话持久化和成本统计，让每一次 AI 调用都可控、可审计、可追踪。v1.1 重构为**豆包式多轮对话**架构，新增会话管理、流式输出、实时余额展示和知识库 RAG 会话注入。

## 快速开始

### 方式一：Docker（推荐）

无需安装 Python，只要装了 Docker 就能跑：

```bash
# 1. 克隆仓库
git clone <your-repo-url>
cd 个人-AI-工作流助手

# 2. 配置 API Key
cp .env.example .env
# 编辑 .env，填入真实的 OPENAI_API_KEY

# 3. 启动
docker compose up -d

# 4. 打开浏览器
# http://127.0.0.1:7860
```

### 方式二：一键脚本

```bash
# Windows
setup.bat

# Linux / macOS
chmod +x setup.sh && ./setup.sh
```

脚本会自动：创建虚拟环境 → 安装依赖 → 复制 `.env` → 运行测试。

### 方式三：手动安装

```bash
# 1. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置
cp .env.example .env
# 编辑 .env 填入 OPENAI_API_KEY

# 4. 启动
python app.py
```

访问 http://127.0.0.1:7860

### 方式四：FastAPI 服务（v0.2 新增）

```bash
# 启动 FastAPI 服务
python app.py --api

# API 文档
# http://127.0.0.1:8000/docs
```

## 运行测试

```bash
python -m pytest tests/ -v
# 或者
python app.py --test
```

## 四种运行模式

| 模式 | 说明 |
|------|------|
| 🚀 快速模式 | 直接调用云端 LLM，追求效果和速度 |
| 🛡️ 隐私增强（推荐） | 本地检测敏感信息，自动脱敏后上传 |
| ✋ 手动确认 | 发送前展示上传预览，用户确认后才发 |
| 🏠 本地兜底 | 不上传云端，仅做本地规则分析 |

## 核心能力

- 📄 **文件解析** — 支持 txt / md，PDF / DOCX 可扩展
- 🔍 **敏感信息检测** — 手机号、邮箱、身份证、API Key、密码、路径
- 🎭 **智能脱敏** — 自动替换为占位符，保留映射表
- 💰 **成本预估** — 上传前估算 token 和费用
- 📊 **RunLog 审计** — 每步执行日志、token 消耗、费用统计
- 🛑 **熔断保护** — token / 费用超限自动中止
- 🔒 **路径白名单** — 防止读取敏感系统文件

### v1.1 新增

- 💬 **豆包式会话** — 左会话列表 + 右对话区，类似 Doubao/Claude 的聊天体验
- 🔄 **多轮对话引擎** — 滑动窗口（默认保留最近 6 轮），历史上下文自动拼接
- 💾 **会话持久化** — JSON 文件存储会话与消息，重启后历史不丢失
- ⚡ **流式输出** — 打字机效果逐 token 展示，实时提取真实 token 消耗
- 🧠 **知识库 RAG 注入** — 上传文件经 ChromaDB 向量检索，自动注入对话上下文
- 💰 **实时余额** — 从 DeepSeek API 同步真实余额，取代本地估算
- 🏠 **本地模型兜底** — 云端 API 不可用时自动降级到 Ollama 本地模型
- 📈 **交互仪表盘** — 费用趋势、模式/模型分布、Token 消耗

### v0.2 新增功能

- 🌐 **FastAPI 接口** — RESTful API，支持桌面端/Web 前端接入
- 📋 **工作流模板** — YAML 定义可复用的工作流
- 📝 **多格式输出** — 支持 Markdown、纯文本、JSON、HTML

### v0.3 新增功能

- 📁 **文件夹监听** — 自动检测文件变更并触发处理
- ⏰ **定时任务** — APScheduler 定时执行预设工作流
- 📦 **批量处理** — 多文件串行批量执行 + 聚合报告
- 🔔 **桌面通知** — plyer 通知 + JSONL 日志兜底

## 前置要求

### 必需
- Python 3.10+
- 云端 LLM API Key（DeepSeek / OpenAI 兼容）

### v1.0 可选（启用向量检索 / 本地兜底）
- [Ollama](https://ollama.com) 本地运行
  ```bash
  ollama pull nomic-embed-text   # embedding 模型
  ollama pull qwen2.5:1.5b       # 兜底 LLM
  ```

## 配置

编辑 `config.yaml` 可调整：

```yaml
model:
  name: deepseek-v4-flash     # 默认使用 DeepSeek V4 Flash 思考模式

limits:
  max_file_size_mb: 5          # 单文件上限
  max_tool_calls_per_request: 15
  max_tokens_per_request: 50000
  max_cost_per_request_yuan: 0.5  # 单次费用上限

paths:
  allowed:                     # 路径白名单
    - data/
    - output/
  blocked_patterns:            # 禁止访问的文件
    - .env
    - "*.key"

# v0.2 新增配置
output:
  format: markdown                 # 默认输出格式

workflow:
  templates_dir: workflows         # 工作流模板目录

api:
  host: 127.0.0.1                  # FastAPI 服务地址
  port: 8000                       # FastAPI 服务端口
  gradio_port: 7861                # Gradio 界面端口（默认 7860）
```

## 项目结构

```
├── app.py                  # 入口（--api 启动 FastAPI）
├── config.yaml             # 应用配置
├── Dockerfile              # Docker 镜像
├── docker-compose.yml      # Docker 编排
├── setup.bat / setup.sh    # 一键安装脚本
├── src/
│   ├── agent/              # Agent 编排 + 多轮对话引擎 + System Prompt
│   ├── api/                # FastAPI 接口（v0.2 新增）
│   ├── tools/              # 文件操作、脱敏、成本估算
│   ├── workflow/           # 预处理、上传策略、后处理、模板
│   ├── knowledge/          # 向量知识库（ChromaDB）+ 会话存储
│   ├── fallback/           # 本地模型兜底（Ollama）
│   ├── monitor/            # 文件夹监听（v0.3）
│   ├── scheduler/          # 定时任务（v0.3）
│   ├── ui/                 # Gradio 豆包式会话界面
│   └── config.py           # 配置加载
├── workflows/              # 工作流模板目录（v0.2 新增）
│   ├── summarize.yaml      # 文档总结模板
│   ├── extract_todos.yaml  # 待办提取模板
│   ├── risk_analysis.yaml  # 风险分析模板
│   └── meeting_notes.yaml  # 会议纪要模板
├── tests/                  # pytest 测试（150+ 用例）
└── data/                   # 测试数据、缓存、偏好存储
```

## 技术栈

- **Agent 框架**: LangChain
- **模型**: DeepSeek V4 Flash（也支持 OpenAI / 通义千问等兼容 API）
- **UI**: Gradio（豆包式会话布局）
- **会话持久化**: JSON 文件（SessionStore）
- **向量检索**: ChromaDB + Ollama nomic-embed-text
- **API**: FastAPI + Uvicorn（v0.2 新增）
- **测试**: pytest + E2E 管线测试
- **部署**: Docker / Docker Compose

## API 使用示例（v0.2）

### 执行任务

```bash
curl -X POST http://127.0.0.1:8000/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "query": "总结这篇文档的要点",
    "mode": "privacy_enhanced",
    "files": ["data/test_sample.md"]
  }'
```

### 使用工作流模板

```bash
curl -X POST http://127.0.0.1:8000/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "query": "分析会议记录",
    "workflow_template": "meeting_notes",
    "files": ["data/meeting.md"]
  }'
```

### 列出可用模板

```bash
curl http://127.0.0.1:8000/templates
```

---

*更多文档请查看 [最终方案.md](./最终方案.md) 了解架构设计细节。*

## 版本路线

| 版本 | 目标 |
|------|------|
| v0.1 ✅ | 可控上传 + 文档处理闭环 + RunLog |
| v0.2 ✅ | FastAPI + 工作流模板 + 多格式输出 |
| v0.3 ✅ | 文件夹监听、定时任务、批量处理、通知 |
| v1.0 ✅ | 向量检索 + 本地模型兜底 + 仪表盘 |
| v1.1 ✅ | 豆包式多轮对话 + 会话管理 + 流式输出 + 实时余额 |
