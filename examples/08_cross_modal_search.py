"""
Кросс-модальный поиск: Текст <-> Изображения
Продвинутый пример с метриками качества и визуализацией

Запуск: uv run python examples/08_cross_modal_search.py
"""

from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from src.embedder import GeminiEmbedder


class CrossModalSearch:
    """Система кросс-модального поиска"""
    
    def __init__(self, embedder: GeminiEmbedder):
        self.embedder = embedder
        self.texts = {}  # {label: (text, embedding)}
        self.images = {}  # {label: (path, embedding)}
    
    def add_text(self, label: str, text: str):
        """Добавить текстовый документ"""
        emb = self.embedder.embed_text(text)
        self.texts[label] = (text, emb)
        print(f"   [OK] Добавлен текст: {label}")
    
    def add_image(self, label: str, image_path: str, description: str = ""):
        """Добавить изображение"""
        emb = self.embedder.embed_image(image_path, description=description)
        self.images[label] = (image_path, emb)
        print(f"   [OK] Добавлено изображение: {label}")
    
    def search_text_to_image(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Поиск изображений по текстовому запросу"""
        query_emb = self.embedder.embed_text(query)
        
        results = []
        for label, (path, img_emb) in self.images.items():
            similarity = self.embedder.cosine_similarity(query_emb, img_emb)
            results.append((label, similarity, path))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def search_image_to_text(self, image_path: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Поиск текстов по изображению"""
        query_emb = self.embedder.embed_image(image_path)
        
        results = []
        for label, (text, text_emb) in self.texts.items():
            similarity = self.embedder.cosine_similarity(query_emb, text_emb)
            results.append((label, similarity, text))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def evaluate_retrieval(self, top_k: int = 5) -> Dict[str, float]:
        """
        Оценка качества поиска (Recall@K, MRR)
        Предполагает, что label текста и изображения совпадают
        """
        recall_sum = 0
        mrr_sum = 0
        total = 0
        
        for text_label, (text, text_emb) in self.texts.items():
            if text_label not in self.images:
                continue
            
            scores = []
            for img_label, (img_path, img_emb) in self.images.items():
                sim = self.embedder.cosine_similarity(text_emb, img_emb)
                scores.append((img_label, sim))
            
            scores.sort(key=lambda x: x[1], reverse=True)
            top_results = [label for label, _ in scores[:top_k]]
            
            if text_label in top_results:
                recall_sum += 1
            
            try:
                rank = top_results.index(text_label) + 1
                mrr_sum += 1.0 / rank
            except ValueError:
                pass
            
            total += 1
        
        for img_label, (img_path, img_emb) in self.images.items():
            if img_label not in self.texts:
                continue
            
            scores = []
            for text_label, (text, text_emb) in self.texts.items():
                sim = self.embedder.cosine_similarity(img_emb, text_emb)
                scores.append((text_label, sim))
            
            scores.sort(key=lambda x: x[1], reverse=True)
            top_results = [label for label, _ in scores[:top_k]]
            
            if img_label in top_results:
                recall_sum += 1
            
            try:
                rank = top_results.index(img_label) + 1
                mrr_sum += 1.0 / rank
            except ValueError:
                pass
            
            total += 1
        
        return {
            'recall_at_k': recall_sum / total if total > 0 else 0,
            'mrr': mrr_sum / total if total > 0 else 0,
            'total_queries': total
        }


def visualize_search_results(results: List[Tuple], query: str, 
                             modality: str = "text_to_image", save: bool = True):
    """Визуализация результатов поиска"""
    
    labels = [r[0] for r in results]
    scores = [r[1] for r in results]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(scores)))
    bars = ax.barh(labels, scores, color=colors, edgecolor='black')
    
    ax.set_xlabel("Косинусное сходство", fontsize=11)
    ax.set_title(f"Поиск: {modality.replace('_', ' ').title()}\nЗапрос: '{query}'", 
                 fontsize=12, pad=20)
    ax.set_xlim(0, 1)
    ax.grid(axis='x', alpha=0.3)
    
    for bar, score in zip(bars, scores):
        ax.text(score + 0.02, bar.get_y() + bar.get_height()/2, 
               f"{score:.3f}", va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    if save:
        output_path = f"results/plots/search_{modality}.png"
        plt.savefig(output_path, dpi=150)
        print(f"[OK] Визуализация сохранена: {output_path}")
    
    plt.show()


def create_comparison_chart(text_to_image_results, image_to_text_results, save: bool = True):
    """Сравнительная диаграмма обоих направлений поиска"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    labels1 = [r[0] for r in text_to_image_results]
    scores1 = [r[1] for r in text_to_image_results]
    colors1 = plt.cm.Blues(np.linspace(0.4, 0.9, len(scores1)))
    
    ax1.barh(labels1, scores1, color=colors1, edgecolor='black')
    ax1.set_xlabel("Сходство", fontsize=10)
    ax1.set_title("Текст -> Изображение", fontsize=11, fontweight='bold')
    ax1.set_xlim(0, 1)
    ax1.grid(axis='x', alpha=0.3)
    
    for bar, score in zip(ax1.patches, scores1):
        ax1.text(score + 0.02, bar.get_y() + bar.get_height()/2, 
                f"{score:.3f}", va='center', fontsize=9)
    
    labels2 = [r[0] for r in image_to_text_results]
    scores2 = [r[1] for r in image_to_text_results]
    colors2 = plt.cm.Oranges(np.linspace(0.4, 0.9, len(scores2)))
    
    ax2.barh(labels2, scores2, color=colors2, edgecolor='black')
    ax2.set_xlabel("Сходство", fontsize=10)
    ax2.set_title("Изображение -> Текст", fontsize=11, fontweight='bold')
    ax2.set_xlim(0, 1)
    ax2.grid(axis='x', alpha=0.3)
    
    for bar, score in zip(ax2.patches, scores2):
        ax2.text(score + 0.02, bar.get_y() + bar.get_height()/2, 
                f"{score:.3f}", va='center', fontsize=9)
    
    plt.suptitle("Двунаправленный кросс-модальный поиск", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save:
        output_path = "results/plots/bidirectional_search.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Сравнительная диаграмма: {output_path}")
    
    plt.show()


def main():
    print("Кросс-модальный поиск: Текст <-> Изображения")
    print("=" * 70)
    
    embedder = GeminiEmbedder(output_dimensionality=768)
    search_engine = CrossModalSearch(embedder)
    
    print("\nИндексация текстов...")
    texts_db = {
        "cat": "Пушистый рыжий кот с зелёными глазами играет с клубком ниток",
        "dog": "Весёлый золотистый ретривер бежит по зелёному парку",
        "car": "Спортивный красный автомобиль Ferrari на гоночной трассе",
        "nature": "Горный пейзаж с кристально чистым озером и сосновым лесом",
        "food": "Свежая итальянская пицца с моцареллой, томатами и базиликом",
        "city": "Ночной мегаполис с небоскрёбами и огнями",
    }
    
    for label, text in texts_db.items():
        search_engine.add_text(label, text)
    
    print("\nИндексация изображений...")
    images_dir = Path("data/sample_images")
    
    available_images = {
        "cat": images_dir / "cat.png",
        "dog": images_dir / "dog.png",
        "car": images_dir / "car.png",
        "nature": images_dir / "nature.png",
        "food": images_dir / "food.png",
    }
    
    for label, img_path in available_images.items():
        if img_path.exists() and label in texts_db:
            search_engine.add_image(label, str(img_path), description=texts_db[label])
        else:
            print(f"   [WARN] Изображение не найдено: {label}")
    
    print(f"\nСтатистика индекса:")
    print(f"   - Текстов: {len(search_engine.texts)}")
    print(f"   - Изображений: {len(search_engine.images)}")
    
    print("\n" + "=" * 70)
    print("ДЕМО 1: Поиск изображений по тексту")
    print("=" * 70)
    
    queries = [
        "Милый пушистый котёнок играет",
        "Быстрая спортивная машина",
        "Красивая природа горы озеро",
        "Вкусная еда пицца"
    ]
    
    for query in queries:
        print(f"\nЗапрос: '{query}'")
        results = search_engine.search_text_to_image(query, top_k=3)
        
        for rank, (label, score, path) in enumerate(results, 1):
            print(f"   {rank}. {label}: {score:.3f}")
    
    demo_results = search_engine.search_text_to_image(queries[0], top_k=5)
    visualize_search_results(demo_results, queries[0], "text_to_image")
    
    print("\n" + "=" * 70)
    print("ДЕМО 2: Поиск текстов по изображению")
    print("=" * 70)
    
    first_img_label = list(available_images.keys())[0]
    first_img_path = available_images[first_img_label]
    
    if first_img_path.exists():
        print(f"\nЗапрос: изображение '{first_img_label}'")
        results = search_engine.search_image_to_text(str(first_img_path), top_k=3)
        
        for rank, (label, score, text) in enumerate(results, 1):
            print(f"   {rank}. {label}: {score:.3f}")
            print(f"      Текст: {text[:60]}...")
    
    print("\n" + "=" * 70)
    print("ДЕМО 3: Двунаправленный поиск для 'car'")
    print("=" * 70)
    
    text_query = "Спортивный красный автомобиль"
    t2i_results = search_engine.search_text_to_image(text_query, top_k=5)
    
    print(f"\n'{text_query}' -> Изображения:")
    for label, score, _ in t2i_results:
        print(f"   {label}: {score:.3f}")
    
    if (images_dir / "car.png").exists():
        i2t_results = search_engine.search_image_to_text(str(images_dir / "car.png"), top_k=5)
        
        print(f"\ncar.png -> Тексты:")
        for label, score, text in i2t_results:
            print(f"   {label}: {score:.3f}")
        
        create_comparison_chart(t2i_results, i2t_results)
    
    print("\n" + "=" * 70)
    print("ОЦЕНКА КАЧЕСТВА ПОИСКА")
    print("=" * 70)
    
    metrics = search_engine.evaluate_retrieval(top_k=3)
    
    print(f"\nМетрики (top-{3}):")
    print(f"   - Recall@3: {metrics['recall_at_k']:.2%}")
    print(f"   - MRR (Mean Reciprocal Rank): {metrics['mrr']:.3f}")
    print(f"   - Всего запросов: {metrics['total_queries']}")
    
    if metrics['recall_at_k'] > 0.8:
        print(f"\n[OK] Отличное качество! Система находит правильные результаты.")
    elif metrics['recall_at_k'] > 0.5:
        print(f"\n[OK] Хорошее качество для небольшого набора данных.")
    else:
        print(f"\n[INFO] Рекомендуется добавить больше данных и улучшить описания.")
    
    print("\n" + "=" * 70)
    print("ПРАКТИЧЕСКИЕ СЦЕНАРИИ ИСПОЛЬЗОВАНИЯ")
    print("=" * 70)
    
    scenarios = [
        ("Поиск по фото", "Загрузите фото -> найдите похожие описания/товары"),
        ("Поиск иллюстраций", "Введите текст -> найдите подходящие изображения"),
        ("Reverse Image Search", "Найдите текстовое описание по изображению"),
        ("Авто-тегирование", "Автоматическая генерация тегов для изображений"),
        ("Визуальный поиск", "Поиск товаров по фотографии или описанию"),
    ]
    
    for name, desc in scenarios:
        print(f"\n{name}")
        print(f"   {desc}")
    
    print("\n" + "=" * 70)
    print("[OK] Кросс-модальный поиск завершён!")
    print("[OK] Все визуализации сохранены в results/plots/")


if __name__ == "__main__":
    main()