"""
Главный скрипт для сравнения Fine-tuning и RAG подходов
"""

import os
import warnings

import torch
from dotenv import load_dotenv
from openai import OpenAI
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import pandas as pd

from fine_tuning_fixed import fine_tune_qwen
from rag_system import create_rag_database

warnings.filterwarnings("ignore")


# Системная инструкция
SYSTEM_INSTRUCTION = """Вы профессиональный менеджер компании Университет Искусственного интеллекта.
Компания занимается продажей курсов по AI.
Ответьте на вопрос (запрос) так, чтобы человек захотел после ответа купить обучение.
Если вы не знаете ответа, просто скажите, что не знаете, не пытайтесь выдумывать ответ."""

# Тестовые вопросы
QUESTIONS = [
    "Привет, а что такое курс CHATGPT PROFESSIONAL и его стоимость?",
    "Привет, а какие курсы есть по Python?",
    "Какие есть направления курсов у УИИ?",
    "Сколько по времени учиться на курсе по GPT?",
    "Есть трудоустройство после обучения?",
]


class RAGGenerator:
    """Генератор ответов на основе RAG + Qwen (без fine-tuning)"""

    def __init__(self, rag_system, hf_token):
        self.rag = rag_system
        self.hf_token = hf_token
        self.model = None
        self.tokenizer = None
        self._load_base_model()

    def _load_base_model(self):
        """Загрузка базовой модели Qwen без fine-tuning"""
        import platform

        print("\n🔄 Загрузка базовой модели Qwen для RAG...", flush=True)
        from tqdm import tqdm

        model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        is_macos = platform.system() == "Darwin"

        if is_macos:
            print(
                "⚠️ INT4 квантизация отключена на macOS (bitsandbytes требует CUDA)",
                flush=True,
            )

        with tqdm(total=100, desc="Загрузка Qwen", unit="%") as pbar:
            pbar.update(10)

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, token=self.hf_token, trust_remote_code=True
            )
            pbar.update(30)

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            pbar.update(10)

            if is_macos:
                # На macOS без квантизации
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    token=self.hf_token,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
            else:
                # На Linux/Windows с квантизацией
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    token=self.hf_token,
                    trust_remote_code=True,
                )
            pbar.update(50)

        print("✓ Базовая модель загружена!", flush=True)

    def generate_answer(self, question):
        """Генерация ответа с использованием RAG"""
        # Получаем контекст из RAG
        context = self.rag.get_context_for_question(question, top_k=3)

        # Формируем промпт
        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": f"Контекст: {context}\n\nВопрос: {question}"},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Генерация
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Извлекаем только ответ ассистента
        if "assistant" in response.lower():
            parts = response.split("assistant")
            if len(parts) > 1:
                response = parts[-1].strip()

        return response


class FineTunedGenerator:
    """Генератор ответов на основе fine-tuned модели"""

    def __init__(self, model_path, hf_token):
        self.model_path = model_path
        self.hf_token = hf_token
        self.model = None
        self.tokenizer = None
        self._load_finetuned_model()

    def _load_finetuned_model(self):
        """Загрузка fine-tuned модели"""
        import platform

        print("\n🔄 Загрузка fine-tuned модели...", flush=True)
        from tqdm import tqdm

        base_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        is_macos = platform.system() == "Darwin"

        if is_macos:
            print(
                "⚠️ INT4 квантизация отключена на macOS (bitsandbytes требует CUDA)",
                flush=True,
            )

        with tqdm(total=100, desc="Загрузка Fine-tuned", unit="%") as pbar:
            pbar.update(10)

            # Загрузка базовой модели с/без квантизации
            if is_macos:
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    token=self.hf_token,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
            else:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    token=self.hf_token,
                    trust_remote_code=True,
                )
            pbar.update(40)

            # Загрузка LoRA адаптера
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
            pbar.update(30)

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )
            pbar.update(15)

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            pbar.update(5)

        print("✓ Fine-tuned модель загружена!", flush=True)

    def generate_answer(self, question):
        """Генерация ответа с использованием fine-tuned модели"""
        # Формируем промпт (без явного контекста, модель уже обучена)
        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": question},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Генерация
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Извлекаем только ответ ассистента
        if "assistant" in response.lower():
            parts = response.split("assistant")
            if len(parts) > 1:
                response = parts[-1].strip()

        return response


def analyze_with_gpt4(results_text, openai_api_key):
    """
    Анализ результатов с помощью GPT-4o-mini

    Args:
        results_text: Текст с результатами обеих моделей
        openai_api_key: API ключ OpenAI

    Returns:
        Текст анализа
    """
    print("\n🔄 Запуск анализа с GPT-4o-mini...")
    from tqdm import tqdm

    with tqdm(total=100, desc="Анализ GPT-4o-mini", unit="%") as pbar:
        client = OpenAI(api_key=openai_api_key)
        pbar.update(10)

        prompt = f"""Проанализируй качество ответов двух подходов к созданию AI-ассистента для продвижения курсов Университета Искусственного Интеллекта:

1. RAG (Retrieval-Augmented Generation) + Qwen2.5-1.5B-Instruct без Fine-tuning
2. Qwen2.5-1.5B-Instruct + Fine-tuning

Критерии оценки:
- Точность и релевантность ответов
- Наличие мотивирующих элементов для покупки курсов
- Естественность и связность текста
- Полнота информации
- Соответствие стилю профессионального менеджера

Результаты тестирования:

{results_text}

Предоставь детальный анализ:
1. Сравнение по каждому вопросу
2. Общие сильные и слабые стороны каждого подхода
3. Рекомендация, какой подход лучше подходит для данной задачи
4. Предложения по улучшению"""
        pbar.update(20)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Ты эксперт по оценке качества AI моделей и NLP систем.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=2000,
        )
        pbar.update(60)

        analysis = response.choices[0].message.content
        pbar.update(10)

    print("✓ Анализ завершен!")
    return analysis


def main():
    """Главная функция"""
    print("=" * 80)
    print("Fine-tuning vs RAG: Сравнение подходов для чат-бота УИИ")
    print("=" * 80)

    # Загрузка переменных окружения
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    csv_path = "selected_qa_full.csv"
    finetuned_model_path = "./qwen_finetuned"
    # Для практического запуска ограничим число примеров (чтобы не тянуть весь большой csv)
    MAX_SAMPLES = 50  # уменьшено для быстрой проверки прогресс-бара
    subset_csv_path = "selected_qa_subset.csv"

    if MAX_SAMPLES is not None:
        if not os.path.exists(subset_csv_path):
            print(
                f"Создаю подвыборку из {csv_path} с {MAX_SAMPLES} примерами -> {subset_csv_path}"
            )
            df = pd.read_csv(csv_path, nrows=MAX_SAMPLES)
            df.to_csv(subset_csv_path, index=False)
        csv_path_to_use = subset_csv_path
    else:
        csv_path_to_use = csv_path

    # Шаг 1: Создание RAG базы
    print("\n" + "=" * 80)
    print("Шаг 1/4: Создание RAG базы данных")
    print("=" * 80)
    rag_system = create_rag_database(csv_path_to_use)

    # Шаг 2: Fine-tuning модели
    print("\n" + "=" * 80)
    print("Шаг 2/4: Fine-tuning модели Qwen")
    print("=" * 80)

    def finetuned_is_complete(path):
        # Проверяем наличие ключевых файлов LoRA/PEFT
        return os.path.isdir(path) and os.path.exists(
            os.path.join(path, "adapter_config.json")
        )

    if not finetuned_is_complete(finetuned_model_path):
        print(
            f"Fine-tuned модель не найдена или неполная: {finetuned_model_path}. Запускаю дообучение..."
        )
        fine_tune_qwen(
            csv_path=csv_path_to_use,
            hf_token=hf_token,
            system_instruction=SYSTEM_INSTRUCTION,
            output_dir=finetuned_model_path,
            max_samples=MAX_SAMPLES,
        )
    else:
        print(f"✓ Fine-tuned модель уже существует: {finetuned_model_path}")

    # Шаг 3: Генерация ответов
    print("\n" + "=" * 80)
    print("Шаг 3/4: Генерация ответов обоими подходами")
    print("=" * 80)

    # Инициализация генераторов
    rag_generator = RAGGenerator(rag_system, hf_token)
    finetuned_generator = FineTunedGenerator(finetuned_model_path, hf_token)

    # RAG ответы
    print("\n📝 Генерация ответов с помощью RAG...")
    rag_results = []
    from tqdm import tqdm

    for i, question in enumerate(
        tqdm(QUESTIONS, desc="RAG генерация", unit="вопрос"), 1
    ):
        answer = rag_generator.generate_answer(question)
        rag_results.append((question, answer))

    # Fine-tuned ответы
    print("\n📝 Генерация ответов с помощью Fine-tuned модели...")
    finetuned_results = []
    for i, question in enumerate(
        tqdm(QUESTIONS, desc="Fine-tuned генерация", unit="вопрос"), 1
    ):
        answer = finetuned_generator.generate_answer(question)
        finetuned_results.append((question, answer))

    # Формирование текста результатов
    results_text = "RAG (Retrieval-Augmented Generation) Qdrant + Qwen2.5-1.5B-Instruct (с INT4 квантизацией) без Fine-tuning\n"
    results_text += "=" * 80 + "\n"

    for i, (question, answer) in enumerate(rag_results, 1):
        results_text += f"\nВопрос {i}: {question}\n"
        results_text += f"Ответ {i}: {answer}\n"
        results_text += "=" * 80 + "\n"

    results_text += "\n\nQwen2.5-1.5B-Instruct (с INT4 квантизацией) + Fine-tuning на дата сете selected_qa_full.csv\n"
    results_text += "=" * 80 + "\n"

    for i, (question, answer) in enumerate(finetuned_results, 1):
        results_text += f"\nВопрос {i}: {question}\n"
        results_text += f"Ответ {i}: {answer}\n"
        results_text += "=" * 80 + "\n"

    # Сохранение результатов в текстовый файл
    with open("result.txt", "w", encoding="utf-8") as f:
        f.write(results_text)

    print("\n✓ Результаты сохранены в result.txt")

    # Сохранение результатов в DataFrame
    print("\n📊 Экспорт результатов в DataFrame...")

    # DataFrames для RAG и Fine-tuned результатов
    rag_df = pd.DataFrame(
        [(i + 1, q, a) for i, (q, a) in enumerate(rag_results)],
        columns=["№ вопроса", "Вопрос", "Ответ RAG"],
    )

    finetuned_df = pd.DataFrame(
        [(i + 1, q, a) for i, (q, a) in enumerate(finetuned_results)],
        columns=["№ вопроса", "Вопрос", "Ответ Fine-tuned"],
    )

    # Объединённый DataFrame для сравнения
    comparison_df = pd.DataFrame(
        {
            "№ вопроса": [i + 1 for i in range(len(QUESTIONS))],
            "Вопрос": QUESTIONS,
            "Ответ RAG": [a for _, a in rag_results],
            "Ответ Fine-tuned": [a for _, a in finetuned_results],
        }
    )

    # Сохранение в CSV
    rag_df.to_csv("results_rag.csv", index=False, encoding="utf-8")
    finetuned_df.to_csv("results_finetuned.csv", index=False, encoding="utf-8")
    comparison_df.to_csv("results_comparison.csv", index=False, encoding="utf-8")

    print("✓ DataFrame сохранены:")
    print("  - results_rag.csv")
    print("  - results_finetuned.csv")
    print("  - results_comparison.csv")

    # Вывод таблиц
    print("\n" + "=" * 80)
    print("RAG РЕЗУЛЬТАТЫ:")
    print("=" * 80)
    print(rag_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("FINE-TUNED РЕЗУЛЬТАТЫ:")
    print("=" * 80)
    print(finetuned_df.to_string(index=False))

    # Шаг 4: Анализ с помощью GPT-4o-mini
    print("\n" + "=" * 80)
    print("Шаг 4/4: Анализ результатов с помощью GPT-4o-mini")
    print("=" * 80)

    analysis = analyze_with_gpt4(results_text, openai_api_key)

    # Добавление анализа в файл
    with open("result.txt", "a", encoding="utf-8") as f:
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("АНАЛИЗ КАЧЕСТВА ОТВЕТОВ (GPT-4o-mini)\n")
        f.write("=" * 80 + "\n\n")
        f.write(analysis if analysis else "Анализ не выполнен")

    print("\n✓ Анализ добавлен в result.txt")

    print("\n" + "=" * 80)
    print("✅ ВСЕ ЗАДАЧИ ВЫПОЛНЕНЫ!")
    print("=" * 80)
    print("\nРезультаты сохранены в файле: result.txt")
    print("\nВключает:")
    print("  - Ответы RAG системы")
    print("  - Ответы Fine-tuned модели")
    print("  - Детальный анализ от GPT-4o-mini")


if __name__ == "__main__":
    main()
