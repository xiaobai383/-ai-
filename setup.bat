@echo off
chcp 65001 >nul
echo ============================================
echo   个人 AI 工作流助手 - 一键安装
echo ============================================
echo.

:: Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未找到 Python，请先安装 Python 3.11+
    echo        下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [1/4] 创建虚拟环境...
if not exist venv (
    python -m venv venv
) else (
    echo       虚拟环境已存在，跳过
)

echo.
echo [2/4] 激活虚拟环境并安装依赖...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip -q
pip install -r requirements.txt -q
echo       依赖安装完成

echo.
echo [3/4] 配置环境变量...
if not exist .env (
    copy .env.example .env >nul
    echo       .env 已创建（从 .env.example 复制）
    echo       >>> 请编辑 .env 文件，填入你的 OPENAI_API_KEY <<<
) else (
    echo       .env 已存在，跳过
)

echo.
echo [4/4] 运行测试...
python -m pytest tests/ -v

echo.
echo ============================================
echo   安装完成！
echo.
echo   启动方式:
echo     venv\Scripts\activate
echo     python app.py
echo.
echo   然后访问 http://127.0.0.1:7860
echo ============================================
pause
