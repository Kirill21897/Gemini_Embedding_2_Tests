"""
Вспомогательные утилиты
"""

import numpy as np
from typing import List
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def visualize_embeddings(
    embeddings: np.ndarray, 
    labels: List[str],
    title: str = "Визуализация эмбеддингов"
):
    """Визуализировать эмбеддинги в 2D с помощью PCA"""
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.6)
    
    for i, label in enumerate(labels):
        plt.annotate(
            label[:20] + "..." if len(label) > 20 else label,
            (reduced[i, 0], reduced[i, 1]),
            fontsize=8,
            alpha=0.7
        )
    
    plt.title(f"{title}\nОбъяснённая дисперсия: {pca.explained_variance_ratio_.sum():.2%}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True, alpha=0.3)
    plt.savefig("results/embeddings_visualization.png", dpi=150)
    plt.show()

def save_embeddings(embeddings: np.ndarray, filepath: str):
    """Сохранить эмбеддинги в файл"""
    np.save(filepath, embeddings)
    print(f"Эмбеддинги сохранены: {filepath}")

def load_embeddings(filepath: str) -> np.ndarray:
    """Загрузить эмбеддинги из файла"""
    return np.load(filepath)

def calculate_stats(embeddings: np.ndarray) -> dict:
    """Рассчитать статистику эмбеддингов"""
    return {
        "shape": embeddings.shape,
        "mean": float(np.mean(embeddings)),
        "std": float(np.std(embeddings)),
        "min": float(np.min(embeddings)),
        "max": float(np.max(embeddings)),
        "dimensionality": embeddings.shape[1] if len(embeddings.shape) > 1 else 0
    }