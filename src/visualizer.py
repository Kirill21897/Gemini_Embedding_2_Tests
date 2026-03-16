"""
Модуль визуализации эмбеддингов
Поддерживает: PCA, t-SNE, heatmaps, распределения, кластеризацию
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Union
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import os

# Настройка стиля
sns.set_style("whitegrid")
plt.rcParams["font.size"] = 10
plt.rcParams["figure.figsize"] = (10, 8)

# Папка для сохранения результатов
OUTPUT_DIR = "results/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)


class EmbeddingVisualizer:
    """Визуализация эмбеддингов Gemini"""
    
    def __init__(self, output_dir: str = OUTPUT_DIR):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_pca_2d(
        self,
        embeddings: np.ndarray,
        labels: Optional[List[str]] = None,
        colors: Optional[List[str]] = None,
        title: str = "PCA: 2D проекция эмбеддингов",
        save: bool = True,
        show: bool = True
    ) -> plt.Figure:
        """2D визуализация через PCA"""
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(embeddings)
        
        fig, ax = plt.subplots(figsize=(12, 9))
        
        if colors is None:
            scatter = ax.scatter(
                reduced[:, 0], reduced[:, 1],
                alpha=0.7, s=50, edgecolors='black', linewidth=0.5
            )
        else:
            scatter = ax.scatter(
                reduced[:, 0], reduced[:, 1],
                c=colors, alpha=0.7, s=50, edgecolors='black', linewidth=0.5,
                cmap='tab10'
            )
        
        # Подписи точек
        if labels:
            for i, label in enumerate(labels):
                ax.annotate(
                    label[:25] + "..." if len(label) > 25 else label,
                    (reduced[i, 0], reduced[i, 1]),
                    fontsize=8, alpha=0.8,
                    xytext=(5, 5), textcoords='offset points'
                )
        
        explained_var = pca.explained_variance_ratio_.sum()
        ax.set_title(f"{title}\nОбъяснённая дисперсия: {explained_var:.2%}")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
        ax.grid(True, alpha=0.3)
        
        if save:
            filepath = os.path.join(self.output_dir, "pca_2d.png")
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"График сохранён: {filepath}")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_pca_3d(
        self,
        embeddings: np.ndarray,
        labels: Optional[List[str]] = None,
        colors: Optional[List[str]] = None,
        title: str = "PCA: 3D проекция эмбеддингов",
        save: bool = True,
        show: bool = True
    ) -> plt.Figure:
        """3D визуализация через PCA"""
        from mpl_toolkits.mplot3d import Axes3D
        
        pca = PCA(n_components=3)
        reduced = pca.fit_transform(embeddings)
        
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        if colors is None:
            ax.scatter(
                reduced[:, 0], reduced[:, 1], reduced[:, 2],
                alpha=0.7, s=50, edgecolors='black', linewidth=0.5
            )
        else:
            ax.scatter(
                reduced[:, 0], reduced[:, 1], reduced[:, 2],
                c=colors, alpha=0.7, s=50, edgecolors='black', linewidth=0.5,
                cmap='tab10'
            )
        
        explained_var = pca.explained_variance_ratio_.sum()
        ax.set_title(f"{title}\nОбъяснённая дисперсия: {explained_var:.2%}")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
        ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]:.2%})")
        
        if save:
            filepath = os.path.join(self.output_dir, "pca_3d.png")
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"3D график сохранён: {filepath}")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_tsne_2d(
        self,
        embeddings: np.ndarray,
        labels: Optional[List[str]] = None,
        colors: Optional[List[str]] = None,
        perplexity: float = 30,
        title: str = "t-SNE: 2D проекция эмбеддингов",
        save: bool = True,
        show: bool = True
    ) -> plt.Figure:
        """2D визуализация через t-SNE"""
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        reduced = tsne.fit_transform(embeddings)
        
        fig, ax = plt.subplots(figsize=(12, 9))
        
        if colors is None:
            ax.scatter(
                reduced[:, 0], reduced[:, 1],
                alpha=0.7, s=50, edgecolors='black', linewidth=0.5
            )
        else:
            ax.scatter(
                reduced[:, 0], reduced[:, 1],
                c=colors, alpha=0.7, s=50, edgecolors='black', linewidth=0.5,
                cmap='tab10'
            )
        
        # Подписи точек
        if labels:
            for i, label in enumerate(labels):
                ax.annotate(
                    label[:25] + "..." if len(label) > 25 else label,
                    (reduced[i, 0], reduced[i, 1]),
                    fontsize=8, alpha=0.8,
                    xytext=(5, 5), textcoords='offset points'
                )
        
        ax.set_title(title)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.grid(True, alpha=0.3)
        
        if save:
            filepath = os.path.join(self.output_dir, "tsne_2d.png")
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"t-SNE график сохранён: {filepath}")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_similarity_heatmap(
        self,
        embeddings: np.ndarray,
        labels: Optional[List[str]] = None,
        title: str = "Матрица косинусного сходства",
        save: bool = True,
        show: bool = True
    ) -> plt.Figure:
        """Тепловая карта сходства между эмбеддингами"""
        # Нормализация и вычисление сходства
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / norms
        similarity = np.dot(normalized, normalized.T)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(
            similarity,
            annot=True, fmt=".2f", cmap="YlOrRd",
            xticklabels=labels, yticklabels=labels,
            ax=ax, square=True, cbar_kws={"shrink": 0.8}
        )
        
        ax.set_title(title)
        ax.set_xlabel("Документы")
        ax.set_ylabel("Документы")
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(fontsize=8)
        
        if save:
            filepath = os.path.join(self.output_dir, "similarity_heatmap.png")
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Heatmap сохранён: {filepath}")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_embedding_distribution(
        self,
        embeddings: np.ndarray,
        title: str = "Распределение значений эмбеддингов",
        save: bool = True,
        show: bool = True
    ) -> plt.Figure:
        """Гистограмма распределения значений в эмбеддингах"""
        flat_values = embeddings.flatten()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Гистограмма
        axes[0].hist(flat_values, bins=50, edgecolor='black', alpha=0.7)
        axes[0].set_title("Гистограмма всех значений")
        axes[0].set_xlabel("Значение")
        axes[0].set_ylabel("Частота")
        axes[0].grid(True, alpha=0.3)
        
        # Boxplot по измерениям (первые 20)
        sample_dims = embeddings[:, :min(20, embeddings.shape[1])]
        axes[1].boxplot(sample_dims, labels=[f"D{i}" for i in range(sample_dims.shape[1])])
        axes[1].set_title("Boxplot первых 20 измерений")
        axes[1].set_xlabel("Измерение")
        axes[1].set_ylabel("Значение")
        axes[1].tick_params(axis='x', rotation=45, labelsize=7)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, "distribution.png")
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Distribution plot сохранён: {filepath}")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_clusters(
        self,
        embeddings: np.ndarray,
        labels: Optional[List[str]] = None,
        n_clusters: int = 3,
        title: str = "Кластеризация эмбеддингов (K-Means)",
        save: bool = True,
        show: bool = True
    ) -> plt.Figure:
        """Визуализация кластеров после K-Means"""
        # Кластеризация
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # PCA для визуализации
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(embeddings)
        
        fig, ax = plt.subplots(figsize=(12, 9))
        
        scatter = ax.scatter(
            reduced[:, 0], reduced[:, 1],
            c=cluster_labels, alpha=0.7, s=50,
            cmap='tab10', edgecolors='black', linewidth=0.5
        )
        
        # Центроиды
        centroids = pca.transform(kmeans.cluster_centers_)
        ax.scatter(
            centroids[:, 0], centroids[:, 1],
            c='red', marker='X', s=200, edgecolors='black',
            label='Центроиды', zorder=5
        )
        
        # Подписи
        if labels:
            for i, label in enumerate(labels):
                ax.annotate(
                    label[:20] + "..." if len(label) > 20 else label,
                    (reduced[i, 0], reduced[i, 1]),
                    fontsize=7, alpha=0.7,
                    xytext=(5, 5), textcoords='offset points'
                )
        
        explained_var = pca.explained_variance_ratio_.sum()
        ax.set_title(f"{title}\nОбъяснённая дисперсия: {explained_var:.2%}")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save:
            filepath = os.path.join(self.output_dir, "clusters.png")
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Кластеры сохранены: {filepath}")
        
        if show:
            plt.show()
        
        return fig, cluster_labels
    
    def plot_comparison(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: np.ndarray,
        candidate_labels: List[str],
        top_k: int = 5,
        title: str = "Сравнение с запросом",
        save: bool = True,
        show: bool = True
    ) -> plt.Figure:
        """Сравнение эмбеддинга запроса с кандидатами"""
        # Вычисление сходства
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        candidate_norms = candidate_embeddings / np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)
        similarities = np.dot(candidate_norms, query_norm)
        
        # Сортировка
        sorted_idx = np.argsort(similarities)[::-1][:top_k]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, top_k))
        bars = ax.barh(
            [candidate_labels[i] for i in sorted_idx],
            similarities[sorted_idx],
            color=colors, edgecolor='black'
        )
        
        ax.set_xlabel("Косинусное сходство")
        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.grid(axis='x', alpha=0.3)
        
        # Значения на барах
        for bar, sim in zip(bars, similarities[sorted_idx]):
            ax.text(sim + 0.02, bar.get_y() + bar.get_height()/2, 
                   f"{sim:.3f}", va='center', fontsize=9)
        
        if save:
            filepath = os.path.join(self.output_dir, "comparison.png")
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Comparison plot сохранён: {filepath}")
        
        if show:
            plt.show()
        
        return fig