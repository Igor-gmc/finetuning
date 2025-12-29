"""
Тестовая версия main.py с детальным выводом
"""
import sys
print("=" * 80, flush=True)
print("Начало работы скрипта", flush=True)
print("=" * 80, flush=True)

print("\n1. Импорт модулей...", flush=True)
import os
import warnings
warnings.filterwarnings('ignore')

print("  - os: OK", flush=True)

import torch
print("  - torch: OK", flush=True)

from dotenv import load_dotenv
print("  - dotenv: OK", flush=True)

from openai import OpenAI
print("  - openai: OK", flush=True)

from peft import PeftModel
print("  - peft: OK", flush=True)

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
print("  - transformers: OK", flush=True)

print("\n2. Импорт локальных модулей...", flush=True)
from fine_tuning import fine_tune_qwen
print("  - fine_tuning: OK", flush=True)

from rag_system import create_rag_database
print("  - rag_system: OK", flush=True)

print("\n3. Загрузка переменных окружения...", flush=True)
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
openai_api_key = os.getenv("OPENAI_API_KEY")
print(f"  - HF_TOKEN: {'OK' if hf_token else 'MISSING'}", flush=True)
print(f"  - OPENAI_API_KEY: {'OK' if openai_api_key else 'MISSING'}", flush=True)

print("\n4. Проверка файлов...", flush=True)
csv_path = "selected_qa_full.csv"
if os.path.exists(csv_path):
    import pandas as pd
    print(f"  - {csv_path}: найден", flush=True)
    print("  - Загрузка CSV для подсчета строк...", flush=True)
    df = pd.read_csv(csv_path)
    print(f"  - Строк в датасете: {len(df)}", flush=True)
else:
    print(f"  - {csv_path}: НЕ НАЙДЕН!", flush=True)
    sys.exit(1)

print("\n✓ Все проверки пройдены!", flush=True)
print("=" * 80, flush=True)
print("\nСкрипт готов к запуску main функций.", flush=True)
print("Для полного запуска используйте: python main.py", flush=True)
