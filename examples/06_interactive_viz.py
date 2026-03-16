"""
Интерактивная визуализация с Plotly
Запуск: uv run python examples/06_interactive_viz.py
"""

import numpy as np
import plotly.express as px
import pandas as pd
import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from src.embedder import GeminiEmbedder

def interactive_pca_3d(embeddings: np.ndarray, labels: list, colors: list):
    """Интерактивная 3D визуализация через Plotly"""
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=3)
    reduced = pca.fit_transform(embeddings)
    
    df = pd.DataFrame({
        'PC1': reduced[:, 0],
        'PC2': reduced[:, 1],
        'PC3': reduced[:, 2],
        'label': [l[:30] for l in labels],
        'category': ['Programming' if c == 0 else 'AI/ML' if c == 1 else 'Other' for c in colors]
    })
    
    fig = px.scatter_3d(
        df, x='PC1', y='PC2', z='PC3',
        color='category', text='label',
        title='Интерактивная 3D визуализация эмбеддингов',
        height=700
    )
    
    fig.update_traces(textposition='top center', marker=dict(size=5))
    fig.write_html("results/plots/interactive_3d.html")
    print("Интерактивный график: results/plots/interactive_3d.html")
    fig.show()

def main():
    embedder = GeminiEmbedder()
    
    texts = [
        "Python для анализа данных",
        "React и фронтенд-разработка",
        "PostgreSQL и оптимизация запросов",
        "Трансформеры и BERT модели",
        "Компьютерное зрение с CNN",
        "Рецепт пасты карбонара",
        "Йога для начинающих",
        "Финансовый анализ и инвестиции"
    ]
    
    colors = [0, 0, 0, 1, 1, 2, 2, 2]
    
    print("Генерация эмбеддингов...")
    embeddings = embedder.embed_texts(texts)
    
    print("Строим интерактивную визуализацию...")
    interactive_pca_3d(embeddings, texts, colors)
    
    print("\nОткройте results/plots/interactive_3d.html в браузере!")

if __name__ == "__main__":
    main()