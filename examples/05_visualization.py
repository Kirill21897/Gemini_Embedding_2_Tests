"""
Пример визуализации эмбеддингов
Запуск: uv run python examples/05_visualization.py
"""

import numpy as np
import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from src.embedder import GeminiEmbedder
from src.visualizer import EmbeddingVisualizer

def main():
    print("Визуализация эмбеддингов Gemini Embedding 2\n")
    
    # Инициализация
    embedder = GeminiEmbedder(output_dimensionality=768)
    visualizer = EmbeddingVisualizer()
    
    # Набор текстов для тестирования
    texts = [
        "Python — язык программирования для веб-разработки",
        "JavaScript используется для создания интерактивных сайтов",
        "SQL работает с реляционными базами данных",
        "Машинное обучение и нейронные сети",
        "Искусственный интеллект в медицине",
        "Глубокое обучение для компьютерного зрения",
        "Рецепт шоколадного торта",
        "Как ухаживать за комнатными растениями",
        "История Древнего Рима и Греции",
        "Средневековая архитектура Европы"
    ]
    
    # Категории для цветов
    categories = {
        "programming": ["Python", "JavaScript", "SQL"],
        "ai_ml": ["Машинное обучение", "Искусственный интеллект", "Глубокое обучение"],
        "other": ["Рецепт", "Растения", "История", "Архитектура"]
    }
    
    colors = []
    for text in texts:
        if any(kw in text for kw in categories["programming"]):
            colors.append(0)  # Синий
        elif any(kw in text for kw in categories["ai_ml"]):
            colors.append(1)  # Оранжевый
        else:
            colors.append(2)  # Зелёный
    
    print("Генерация эмбеддингов...")
    embeddings = embedder.embed_texts(texts)
    print(f"Создано {len(embeddings)} эмбеддингов размером {embeddings.shape[1]}\n")
    
    # 1. PCA 2D
    print("Строим PCA проекцию...")
    visualizer.plot_pca_2d(
        embeddings, labels=texts, colors=colors,
        title="Тематические кластеры текстов (PCA)"
    )
    
    # 2. t-SNE 2D
    print("Строим t-SNE проекцию...")
    visualizer.plot_tsne_2d(
        embeddings, labels=texts, colors=colors,
        perplexity=5,  # Мало данных -> маленькая perplexity
        title="Тематические кластеры текстов (t-SNE)"
    )
    
    # 3. Heatmap сходства
    print("Строим heatmap сходства...")
    short_labels = [f"{i}: {t[:15]}..." for i, t in enumerate(texts)]
    visualizer.plot_similarity_heatmap(
        embeddings, labels=short_labels,
        title="Косинусное сходство между текстами"
    )
    
    # 4. Распределение значений
    print("Строим распределение...")
    visualizer.plot_embedding_distribution(
        embeddings, title="Распределение значений эмбеддингов"
    )
    
    # 5. Кластеризация
    print("Кластеризуем...")
    visualizer.plot_clusters(
        embeddings, labels=texts, n_clusters=3,
        title="Автоматическая кластеризация текстов"
    )
    
    # 6. Сравнение с запросом
    print("Сравниваем с запросом...")
    query = "Программирование и веб-разработка"
    query_emb = embedder.embed_text(query)
    
    visualizer.plot_comparison(
        query_emb, embeddings, candidate_labels=texts,
        top_k=5, title=f"Похожие на: '{query}'"
    )
    
    print("\nВсе визуализации сохранены в results/plots/")
    print("Готово!")

if __name__ == "__main__":
    main()