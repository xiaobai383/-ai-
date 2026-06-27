# 个人 AI 工作流助手

> 云端智力，本地边界 — 可控上传、可审计执行

一个面向个人知识和工作文档的 AI 工作流助手。使用云端大模型保证智能效果，同时通过本地预处理、敏感信息脱敏、上传预览、执行日志和成本统计，让每一次 AI 调用都可控、可审计、可追踪。

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
```

## 项目结构

```
├── app.py                  # 入口
├── config.yaml             # 应用配置
├── Dockerfile              # Docker 镜像
├── docker-compose.yml      # Docker 编排
├── setup.bat / setup.sh    # 一键安装脚本
├── src/
│   ├── agent/              # Agent 编排 + System Prompt
│   ├── tools/              # 文件操作、脱敏、成本估算
│   ├── workflow/           # 预处理、上传策略、后处理
│   ├── observability/      # RunLog 可观测
│   ├── ui/                 # Gradio 界面
│   └── config.py           # 配置加载
├── tests/                  # pytest 测试（128 个用例）
└── data/                   # 测试数据和缓存
```

## 技术栈

- **Agent 框架**: LangChain
- **模型**: DeepSeek V4 Flash（也支持 OpenAI / 通义千问等兼容 API）
- **UI**: Gradio
- **测试**: pytest
- **部署**: Docker / Docker Compose

## 版本路线

| 版本 | 目标 |
|------|------|
| v0.1 ✅ | 可控上传 + 文档处理闭环 + RunLog |
| v0.2 | FastAPI + 工作流模板 + 用户偏好记忆 |
| v0.3 | 文件夹监听、定时任务、批量处理 |
| v1.0 | 向量检索、本地模型兜底、仪表盘 |
