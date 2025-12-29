#!/bin/bash
# === Инициализация нового Python-проекта с Conda (без Docker, без venv) ===
# Автор: Игорь (обновлено и исправлено)

# --------------------------------------------------------------
# 📌 КАК ИСПОЛЬЗОВАТЬ ЭТОТ СКРИПТ:
#
# 1. Сохранить файл, например:
#       conda_vscode.sh
#
# 2. Сделать файл исполняемым:
#       chmod +x conda_vscode.sh
#
# 3. Запустить:
#       ./conda_vscode.sh
#
# --------------------------------------------------------------

PROJECT_ROOT=$(pwd)
echo "📁 Инициализация Conda-проекта в: $PROJECT_ROOT"

# ============================================================
# 1. Проверка наличия Condа
# ============================================================
if ! command -v conda &> /dev/null; then
    echo "❌ Conda не найдена. Установи Miniconda:"
    echo "   brew install --cask miniconda"
    exit 1
fi

# Загружаем conda hooks (ОБЯЗАТЕЛЬНО!)
echo "🔧 Загружаю conda hooks..."
source "$(conda info --base)/etc/profile.d/conda.sh"

# ============================================================
# 2. Создание окружения
# ============================================================
read -p "🧱 Введи имя окружения (по умолчанию: projectenv): " ENV_NAME
ENV_NAME=${ENV_NAME:-projectenv}

if conda info --envs | grep -qw "$ENV_NAME"; then
    echo "✅ Окружение '$ENV_NAME' уже существует"
else
    echo "⚙ Создаю окружение '$ENV_NAME' с Python 3.10..."
    conda create -y -n "$ENV_NAME" python=3.10
fi

# ============================================================
# 3. Активация окружения и установка пакетов
# ============================================================
echo "🐍 Активирую окружение '$ENV_NAME'..."
conda activate "$ENV_NAME"

echo "📦 Устанавливаю FAISS и базовые пакеты..."
conda install -y -c conda-forge faiss-cpu
pip install black pylint

# Узнаём путь к Python в Conda
PYTHON_PATH=$(which python)
echo "🔎 Найден интерпретатор: $PYTHON_PATH"

# ============================================================
# 4. Создание служебных файлов (VSCode, gitignore, requirements)
# ============================================================
mkdir -p .vscode

# VS Code settings
cat > .vscode/settings.json << EOF
{
    "python.defaultInterpreterPath": "$PYTHON_PATH",
    "editor.formatOnSave": true,
    "files.autoSave": "afterDelay",
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "editor.inlineSuggest.enabled": false,
    "github.copilot.suggestionShowOnTriggerOnly": true
}
EOF

# .gitignore (включая ClaudeCode)
cat > .gitignore << 'EOF'
# ==============================
#   ClaudeCode (Anthropic)
# ==============================
.claude/
claude/
claude_cache/
.claude-cache/
.claude_history/
.claude_history.json
.claude.json
claude.json
claude_prompts/
.claude_prompts/
claude_tmp/
.claude_tmp/
*.prompt.md
*.prompt
*.claude-prompt

# ==============================
#   Codeium (если используется)
# ==============================
.codeium/
.codeiumrc

# ==============================
#   Python
# ==============================
__pycache__/
*.pyc
*.pyo
*.pyd
*.log

# ==============================
#   Environments
# ==============================
venv/
.env
.env.local
*.conda
*.yml

# ==============================
#   macOS
# ==============================
.DS_Store

# ==============================
#   VSCode
# ==============================
.vscode/*
!.vscode/settings.json
!.vscode/extensions.json
!.vscode/launch.json
!.vscode/tasks.json

# ==============================
#   Other
# ==============================
*.bak
*.tmp
*.swp
*.lock
EOF

# requirements.txt
cat > requirements.txt << 'EOF'
black
pylint
aiogram==3.*
SQLAlchemy==2.*
asyncpg
python-dotenv
openai
EOF

# ============================================================
# 5. Финальное сообщение
# ============================================================
echo ""
echo "✅ Проект инициализирован!"
echo "🐍 Активируй окружение:"
echo "   conda activate $ENV_NAME"
echo ""
echo "📦 Установи зависимости:"
echo "   pip install -r requirements.txt"
echo ""
echo "🚀 Готово! Открывай проект в VS Code и запускай main.py"
echo ""
echo "🔍 Контроль:"
echo "   Текущее окружение: \$CONDA_DEFAULT_ENV"
echo "   Python: $PYTHON_PATH"
