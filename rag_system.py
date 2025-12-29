"""
RAG система с использованием Qdrant и SentenceTransformers
"""
import os
import gc
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class RAGSystem:
    def __init__(self, collection_name="uii_knowledge_base", embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Инициализация RAG системы

        Args:
            collection_name: Имя коллекции в Qdrant
            embedding_model: Модель для создания эмбеддингов
        """
        self.collection_name = collection_name
        self.client = QdrantClient(":memory:")  # In-memory для простоты
        self.encoder = SentenceTransformer(embedding_model)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()

    def create_collection(self):
        """Создание коллекции в Qdrant"""
        print(f"Создание коллекции {self.collection_name}...")

        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.embedding_dim,
                distance=Distance.COSINE
            )
        )
        print("Коллекция успешно создана!")

    def load_and_index_data(self, csv_path):
        """
        Загрузка данных из CSV и индексация в Qdrant

        Args:
            csv_path: Путь к CSV файлу с данными
        """
        print(f"Загрузка данных из {csv_path}...", flush=True)
        df = pd.read_csv(csv_path)

        print(f"Загружено {len(df)} записей", flush=True)
        print("Создание эмбеддингов...", flush=True)

        points = []
        batch_size = 50  # Уменьшаем размер батча для экономии памяти
        successful_count = 0

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Индексация"):
            try:
                # Комбинируем вопрос и контекст для более полного поиска
                text_to_embed = f"{row['question']} {row['context']}"
                embedding = self.encoder.encode(text_to_embed, show_progress_bar=False)

                point = PointStruct(
                    id=idx,
                    vector=embedding.tolist(),
                    payload={
                        "question": str(row['question']),
                        "answer": str(row['answer']),
                        "context": str(row['context'])
                    }
                )
                points.append(point)

                # Загружаем батчами для эффективности
                if len(points) >= batch_size:
                    try:
                        self.client.upsert(
                            collection_name=self.collection_name,
                            points=points
                        )
                        successful_count += len(points)
                        points = []
                        # Принудительно освобождаем память
                        gc.collect()
                    except Exception as e:
                        print(f"\n❌ Ошибка при загрузке батча на индексе {idx}: {str(e)}", flush=True)
                        print(f"Размер батча: {len(points)}", flush=True)
                        raise

            except Exception as e:
                print(f"\n❌ Ошибка при обработке строки {idx}: {str(e)}", flush=True)
                print(f"Данные строки: question={row.get('question', 'N/A')[:50]}...", flush=True)
                raise

        # Загружаем оставшиеся точки
        if points:
            try:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                successful_count += len(points)
            except Exception as e:
                print(f"\n❌ Ошибка при загрузке последнего батча: {str(e)}", flush=True)
                raise

        print(f"✓ Индексация завершена! Добавлено {successful_count} документов", flush=True)

    def search(self, query, top_k=3):
        """
        Поиск наиболее релевантных документов

        Args:
            query: Поисковый запрос
            top_k: Количество возвращаемых результатов

        Returns:
            Список найденных документов
        """
        query_embedding = self.encoder.encode(query, show_progress_bar=False)

        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=top_k
        )

        return search_results

    def get_context_for_question(self, question, top_k=3):
        """
        Получение контекста для вопроса

        Args:
            question: Вопрос пользователя
            top_k: Количество документов для контекста

        Returns:
            Строка с объединенным контекстом
        """
        results = self.search(question, top_k=top_k)

        contexts = []
        for result in results:
            contexts.append(result.payload['context'])

        return "\n\n".join(contexts)


def create_rag_database(csv_path):
    """
    Создание RAG базы данных

    Args:
        csv_path: Путь к CSV файлу с данными

    Returns:
        Инициализированная RAGSystem
    """
    rag = RAGSystem()
    rag.create_collection()
    rag.load_and_index_data(csv_path)
    return rag


if __name__ == "__main__":
    # Тест RAG системы
    csv_path = "selected_qa_full.csv"
    rag = create_rag_database(csv_path)

    # Тестовый запрос
    test_query = "Какие курсы есть по Python?"
    context = rag.get_context_for_question(test_query)
    print(f"\nТестовый запрос: {test_query}")
    print(f"Найденный контекст:\n{context[:500]}...")
