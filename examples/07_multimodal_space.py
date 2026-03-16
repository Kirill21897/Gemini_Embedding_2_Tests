"""
Мультимодальное пространство эмбеддингов Gemini Embedding 2
Исследуем единое пространство для текста и изображений

Запуск: uv run python examples/07_multimodal_space.py
"""

from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from src.embedder import GeminiEmbedder
from src.visualizer import EmbeddingVisualizer

def create_sample_images():
    """Создаёт тестовые изображения (заглушки)"""
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    
    images_dir = Path("data/sample_images")
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Создаём простые цветные изображения с текстом
    samples = {
        "cat": ("#FF6B6B", "🐱 CAT"),
        "dog": ("#4ECDC4", "🐶 DOG"),
        "car": ("#45B7D1", "🚗 CAR"),
        "nature": ("#96CEB4", "🌿 NATURE"),
        "food": ("#FFEAA7", "🍕 FOOD"),
    }
    
    created_paths = []
    
    for name, (color, text) in samples.items():
        # Создаём изображение 400x300
        img = Image.new('RGB', (400, 300), color=color)
        draw = ImageDraw.Draw(img)
        
        # Пытаемся добавить текст (если есть шрифт)
        try:
            # Пробуем стандартные шрифты
            font = ImageFont.truetype("arial.ttf", 40)
        except:
            font = ImageFont.load_default()
        
        # Центрируем текст
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x = (400 - text_width) // 2
        y = (300 - text_height) // 2
        
        draw.text((x, y), text, fill='black', font=font)
        
        # Сохраняем
        path = images_dir / f"{name}.png"
        img.save(path)
        created_paths.append((path, text))
        print(f"Создано изображение: {path}")
    
    return created_paths

def main():
    print("Мультимодальное пространство эмбеддингов\n")
    print("=" * 60)
    
    embedder = GeminiEmbedder(output_dimensionality=768)
    visualizer = EmbeddingVisualizer()
    
    # Текстовые описания
    text_samples = {
        "cat": "Милый пушистый кот играет с клубком",
        "dog": "Весёлая собака бежит по парку",
        "car": "Спортивный красный автомобиль на трассе",
        "nature": "Горный пейзаж с озером и соснами",
        "food": "Свежая пицца с моцареллой и базиликом",
    }
    
    # Создаём тестовые изображения (если их нет)
    images_dir = Path("data/sample_images")
    if not list(images_dir.glob("*.png")):
        print("Создание тестовых изображений...\n")
        image_paths = create_sample_images()
    else:
        print("Используем существующие изображения\n")
        image_paths = [(p, name) for p in images_dir.glob("*.png") 
                       for name in text_samples.keys() if name in p.name]
    
    print("\nГенерация текстовых эмбеддингов...")
    text_labels = list(text_samples.keys())
    text_embeddings = embedder.embed_texts(list(text_samples.values()))
    print(f"Создано {len(text_embeddings)} текстовых эмбеддингов")
    
    print("\nГенерация эмбеддингов изображений...")
    image_embeddings = []
    image_labels = []
    
    for img_path, label in image_paths:
        if label in text_samples:  # Только если есть соответствие
            try:
                emb = embedder.embed_image(
                    str(img_path), 
                    description=text_samples[label]
                )
                image_embeddings.append(emb)
                image_labels.append(label)
                print(f"   ✓ {img_path.name}")
            except Exception as e:
                print(f"   ✗ {img_path.name}: {e}")
    
    image_embeddings = np.array(image_embeddings)
    print(f"Создано {len(image_embeddings)} эмбеддингов изображений")
    
    # Объединяем тексты и изображения
    all_embeddings = np.vstack([text_embeddings, image_embeddings])
    all_labels = [f" {l}" for l in text_labels] + [f"{l}" for l in image_labels]
    all_types = ['text'] * len(text_labels) + ['image'] * len(image_labels)
    
    # Цвета: одинаковые категории = одинаковый цвет
    colors_map = {
        'cat': 0, 'dog': 1, 'car': 2, 'nature': 3, 'food': 4
    }
    all_colors = [colors_map[l.replace(' ', '').replace(' ', '')] for l in all_labels]
    
    print("\n Общая статистика:")
    print(f"   • Всего эмбеддингов: {len(all_embeddings)}")
    print(f"   • Текстов: {len(text_embeddings)}")
    print(f"   • Изображений: {len(image_embeddings)}")
    print(f"   • Размерность: {all_embeddings.shape[1]}")
    
    # 1. Визуализация всего пространства
    print("\n Визуализация мультимодального пространства...")
    visualizer.plot_pca_2d(
        all_embeddings,
        labels=all_labels,
        colors=all_colors,
        title=" Текст + Изображения в едином пространстве (PCA)",
        show=False
    )
    
    # 🔍 2. Кросс-модальный поиск: текст → изображение
    print("\n Кросс-модальный поиск: текст → изображения")
    print("=" * 60)
    
    query_text = "Милый пушистый кот"
    query_emb = embedder.embed_text(query_text)
    
    # Ищем среди изображений
    image_sims = []
    for i, img_emb in enumerate(image_embeddings):
        sim = embedder.cosine_similarity(query_emb, img_emb)
        image_sims.append((image_labels[i], sim))
    
    image_sims.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nЗапрос: '{query_text}'\n")
    print("Наиболее похожие изображения:")
    for label, sim in image_sims[:3]:
        print(f"   {label}: {sim:.3f}")
    
    #  3. Визуализация поиска
    print("\n Визуализация поиска...")
    visualizer.plot_comparison(
        query_emb,
        image_embeddings,
        candidate_labels=[f" {l}" for l in image_labels],
        top_k=5,
        title=f" Поиск по запросу: '{query_text}'",
        show=False
    )
    
    #  4. Heatmap сходства текст-изображение
    print("\n Матрица сходства текст ↔ изображение...")
    
    # Создаём матрицу
    cross_matrix = np.zeros((len(text_labels), len(image_labels)))
    for i, text_emb in enumerate(text_embeddings):
        for j, img_emb in enumerate(image_embeddings):
            cross_matrix[i, j] = embedder.cosine_similarity(text_emb, img_emb)
    
    # Визуализация
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cross_matrix, cmap='YlOrRd', aspect='auto')
    
    ax.set_xticks(range(len(image_labels)))
    ax.set_yticks(range(len(text_labels)))
    ax.set_xticklabels([f" {l}" for l in image_labels], rotation=45, ha='right')
    ax.set_yticklabels([f" {l}" for l in text_labels])
    
    ax.set_title("Сходство между текстами и изображениями")
    
    # Добавляем значения
    for i in range(len(text_labels)):
        for j in range(len(image_labels)):
            ax.text(j, i, f'{cross_matrix[i, j]:.2f}', 
                   ha="center", va="center", color="black", fontsize=8)
    
    plt.colorbar(im, label='Косинусное сходство')
    plt.tight_layout()
    plt.savefig("results/plots/cross_modal_heatmap.png", dpi=150)
    print(" Heatmap сохранён: results/plots/cross_modal_heatmap.png")
    plt.show()
    
    #  5. Проверка семантической близости
    print("\n Семантическая близость одинаковых понятий:")
    print("=" * 60)
    
    for label in text_labels:
        if label in image_labels:
            # Находим индексы
            text_idx = text_labels.index(label)
            if label in image_labels:
                img_idx = image_labels.index(label)
                
                text_emb = text_embeddings[text_idx]
                img_emb = image_embeddings[img_idx]
                
                similarity = embedder.cosine_similarity(text_emb, img_emb)
                
                print(f"\n{label.upper()}:")
                print(f"   Текст ↔ Изображение: {similarity:.3f}")
                
                # Сравниваем с другими
                other_texts = [i for i in range(len(text_labels)) if i != text_idx]
                avg_other = np.mean([
                    embedder.cosine_similarity(text_emb, text_embeddings[i])
                    for i in other_texts
                ])
                
                print(f"   Среднее с другими текстами: {avg_other:.3f}")
                
                if similarity > avg_other:
                    print(f"    Отлично! Текст и изображение близки")
                else:
                    print(f"     Текст и изображение дальше, чем другие тексты")
    
    print("\n" + "=" * 60)
    print(" Все визуализации сохранены в results/plots/")
    print(" Мультимодальный анализ завершён!")

if __name__ == "__main__":
    main()