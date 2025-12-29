"""
Fine-tuning Qwen2.5-1.5B-Instruct с использованием LoRA
"""

import os
import pandas as pd
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from transformers.trainer_callback import ProgressCallback
from transformers.trainer_callback import ProgressCallback
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


class QwenFineTuner:
    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct", hf_token=None):
        """
        Инициализация Fine-tuner для Qwen модели

        Args:
            model_name: Название модели на HuggingFace
            """
            Fine-tuning Qwen2.5-1.5B-Instruct с использованием LoRA
            """

            import os
            import pandas as pd
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                TrainingArguments,
                Trainer,
                BitsAndBytesConfig,
            )
            from transformers.trainer_callback import ProgressCallback
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            from datasets import Dataset
            from tqdm import tqdm
            import warnings

            warnings.filterwarnings("ignore")


            class QwenFineTuner:
                def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct", hf_token=None):
                    self.model_name = model_name
                    self.hf_token = hf_token
                    self.tokenizer = None
                    self.model = None
                    self.peft_model = None

                def load_model(self):
                    """Загрузка модели с квантизацией INT4 (или FP16 на macOS)"""
                    import platform

                    is_macos = platform.system() == "Darwin"

                    if is_macos:
                        print(f"Загрузка модели {self.model_name} без квантизации (macOS)...", flush=True)
                        print(
                            "⚠️ INT4 квантизация отключена на macOS (bitsandbytes требует CUDA)",
                            flush=True,
                        )
                    else:
                        print(f"Загрузка модели {self.model_name} с INT4 квантизацией...", flush=True)

                    # Токенизатор
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.model_name, token=self.hf_token, trust_remote_code=True
                    )

                    if self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token

                    # Попытка загрузки с квантизацией, с запасным вариантом
                    if is_macos:
                        self.model = AutoModelForCausalLM.from_pretrained(
                            self.model_name,
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
                        try:
                            self.model = AutoModelForCausalLM.from_pretrained(
                                self.model_name,
                                quantization_config=bnb_config,
                                device_map="auto",
                                token=self.hf_token,
                                trust_remote_code=True,
                            )
                        except Exception as e:
                            print("⚠️ Не удалось загрузить модель с INT4 квантизацией:", str(e))
                            print("Попытка загрузки без квантизации (FP16)...", flush=True)
                            self.model = AutoModelForCausalLM.from_pretrained(
                                self.model_name,
                                torch_dtype=torch.float16,
                                device_map="auto",
                                token=self.hf_token,
                                trust_remote_code=True,
                                low_cpu_mem_usage=True,
                            )

                    print("✓ Модель успешно загружена!", flush=True)

                def prepare_lora(self):
                    """Подготовка модели для LoRA обучения"""
                    print("Настройка LoRA...")

                    # Подготовка модели для k-bit обучения
                    self.model = prepare_model_for_kbit_training(self.model)
                    # Отключаем use_cache для надёжности при обучении
                    try:
                        self.model.config.use_cache = False
                    except Exception:
                        pass

                    lora_config = LoraConfig(
                        r=16,
                        lora_alpha=32,
                        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                        lora_dropout=0.05,
                        bias="none",
                        task_type="CAUSAL_LM",
                    )

                    self.peft_model = get_peft_model(self.model, lora_config)
                    self.peft_model.print_trainable_parameters()

                    print("✓ LoRA настроен!")

                def prepare_dataset(self, csv_path, system_instruction, max_samples=None):
                    """Подготовка датасета для обучения"""
                    print(f"Загрузка датасета из {csv_path}...")
                    df = pd.read_csv(csv_path)

                    if max_samples:
                        df = df.head(max_samples)
                        print(f"Используется {max_samples} примеров для ускорения обучения")

                    print(f"Подготовка {len(df)} примеров...")

                    def create_prompt(row):
                        messages = [
                            {"role": "system", "content": system_instruction},
                            {"role": "user", "content": f"Контекст: {row['context']}\n\nВопрос: {row['question']}"},
                            {"role": "assistant", "content": row["answer"]},
                        ]
                        text = self.tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=False
                        )
                        return text

                    texts = []
                    for _, row in tqdm(df.iterrows(), total=len(df), desc="Создание промптов"):
                        texts.append(create_prompt(row))

                    print("Токенизация...")
                    encodings = self.tokenizer(
                        texts,
                        truncation=True,
                        padding=True,
                        max_length=512,
                        return_tensors="pt",
                    )

                    dataset = Dataset.from_dict(
                        {
                            "input_ids": encodings["input_ids"],
                            "attention_mask": encodings["attention_mask"],
                            "labels": encodings["input_ids"].clone(),
                        }
                    )

                    print(f"✓ Датасет подготовлен: {len(dataset)} примеров")
                    return dataset

                def train(self, train_dataset, output_dir="./qwen_finetuned"):
                    """Обучение модели"""
                    print("Начало обучения...")

                    training_args = TrainingArguments(
                        output_dir=output_dir,
                        num_train_epochs=1,
                        per_device_train_batch_size=1,
                        gradient_accumulation_steps=2,
                        learning_rate=5e-4,
                        fp16=True,
                        logging_steps=5,
                        save_strategy="epoch",
                        warmup_steps=20,
                        weight_decay=0.01,
                        report_to="none",
                        disable_tqdm=False,
                        gradient_checkpointing=False,
                    )

                    trainer = Trainer(
                        model=self.peft_model,
                        args=training_args,
                        train_dataset=train_dataset,
                        callbacks=[ProgressCallback()],
                    )

                    trainer.train()

                    print(f"Сохранение модели в {output_dir}...")
                    self.peft_model.save_pretrained(output_dir)
                    self.tokenizer.save_pretrained(output_dir)

                    print("✓ Обучение завершено!")

                def generate(self, prompt, max_new_tokens=256):
                    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.peft_model.device)

                    with torch.no_grad():
                        outputs = self.peft_model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            temperature=0.7,
                            do_sample=True,
                            top_p=0.9,
                            pad_token_id=self.tokenizer.pad_token_id,
                        )

                    response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                    if "assistant" in response:
                        response = response.split("assistant")[-1].strip()

                    return response


            def fine_tune_qwen(csv_path, hf_token, system_instruction, output_dir="./qwen_finetuned", max_samples=1000):
                tuner = QwenFineTuner(hf_token=hf_token)
                tuner.load_model()
                tuner.prepare_lora()

                dataset = tuner.prepare_dataset(csv_path, system_instruction, max_samples=max_samples)
                tuner.train(dataset, output_dir=output_dir)

                return tuner


            if __name__ == "__main__":
                from dotenv import load_dotenv

                load_dotenv()

                csv_path = "selected_qa_full.csv"
                hf_token = os.getenv("HF_TOKEN")

                system_instruction = """Вы профессиональный менеджер компании Университет Искусственного интеллекта.
                Компания занимается продажей курсов по AI.
                Ответьте на вопрос (запрос) так, чтобы человек захотел после ответа купить обучение.
                Если вы не знаете ответа, просто скажите, что не знаете, не пытайтесь выдумывать ответ."""

                tuner = fine_tune_qwen(
                    csv_path=csv_path,
                    hf_token=hf_token,
                    system_instruction=system_instruction,
                    max_samples=100,
                )

                print("Fine-tuning завершен!")
        csv_path, system_instruction, max_samples=max_samples
