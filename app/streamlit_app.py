"""
Streamlit приложение для тестирования Gemini Embedding 2
Запуск: streamlit run app/streamlit_app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from typing import List, Tuple
from sklearn.decomposition import PCA

# Добавляем корень проекта в путь
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from src.embedder import GeminiEmbedder
from src.visualizer import EmbeddingVisualizer


# Настройка страницы
st.set_page_config(
    page_title="Gemini Embedding 2 Tests",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Кэширование клиента
@st.cache_resource
def get_embedder():
    """Создаёт и кэширует клиент эмбеддингов"""
    try:
        return GeminiEmbedder(output_dimensionality=768)
    except ValueError as e:
        st.error(f"Ошибка инициализации: {e}")
        st.info("Убедитесь, что API ключ указан в файле .env")
        return None


# Кэширование эмбеддингов
@st.cache_data
def compute_embedding(text: str) -> np.ndarray:
    """Вычисляет эмбеддинг для текста с кэшированием"""
    embedder = get_embedder()
    if embedder:
        return embedder.embed_text(text)
    return None


def main():
    st.title("Gemini Embedding 2 - Тестирование модели")
    st.markdown("---")
    
    # Инициализация
    embedder = get_embedder()
    visualizer = EmbeddingVisualizer(output_dir="results/plots")
    
    if not embedder:
        st.stop()
    
    # Боковая панель
    st.sidebar.title("Навигация")
    page = st.sidebar.radio(
        "Выберите раздел:",
        [
            "Текстовые эмбеддинги",
            "Мультимодальный поиск",
            "Визуализация пространства",
            "Сравнение моделей",
            "Пакетная обработка"
        ]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Параметры")
    
    dimensionality = st.sidebar.selectbox(
        "Размерность вектора:",
        [128, 768, 1536, 3072],
        index=1
    )
    
    task_type = st.sidebar.selectbox(
        "Тип задачи:",
        [
            "SEMANTIC_SIMILARITY",
            "RETRIEVAL_QUERY",
            "RETRIEVAL_DOCUMENT",
            "CLASSIFICATION",
            "QUESTION_ANSWERING"
        ],
        index=0
    )
    
    # Обновляем конфигурацию embedder при изменении параметров
    embedder.output_dimensionality = dimensionality
    embedder.task_type = task_type
    embedder.config = type('obj', (object,), {
        'task_type': task_type,
        'output_dimensionality': dimensionality
    })()
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        f"**Текущие параметры:**\n\n"
        f"- Модель: gemini-embedding-2-preview\n"
        f"- Размерность: {dimensionality}\n"
        f"- Задача: {task_type}"
    )
    
    # Основной контент
    if page == "Текстовые эмбеддинги":
        page_text_embeddings(embedder)
    elif page == "Мультимодальный поиск":
        page_multimodal_search(embedder)
    elif page == "Визуализация пространства":
        page_visualization(embedder, visualizer)
    elif page == "Сравнение моделей":
        page_comparison(embedder)
    elif page == "Пакетная обработка":
        page_batch_processing(embedder)


def page_text_embeddings(embedder: GeminiEmbedder):
    """Страница: Текстовые эмбеддинги"""
    
    st.header("Текстовые эмбеддинги")
    st.write("Генерация эмбеддингов для отдельных текстов и сравнение")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Ввод текста")
        text1 = st.text_area(
            "Текст 1:",
            "Искусственный интеллект меняет мир технологий",
            height=100,
            key="text1"
        )
        
        text2 = st.text_area(
            "Текст 2:",
            "Машинное обучение трансформирует индустрии",
            height=100,
            key="text2"
        )
    
    with col2:
        st.subheader("Результаты")
        
        if st.button("Сгенерировать эмбеддинги", type="primary"):
            if text1 and text2:
                with st.spinner("Генерация эмбеддингов..."):
                    emb1 = embedder.embed_text(text1)
                    emb2 = embedder.embed_text(text2)
                
                st.success("Эмбеддинги созданы!")
                
                # Статистика
                st.metric("Размерность", len(emb1))
                st.metric("Среднее значение", f"{np.mean(emb1):.4f}")
                st.metric("Стандартное отклонение", f"{np.std(emb1):.4f}")
                
                # Сходство
                similarity = embedder.cosine_similarity(emb1, emb2)
                st.metric("Косинусное сходство", f"{similarity:.4f}")
                
                # Интерпретация
                if similarity > 0.8:
                    st.info("Тексты очень похожи семантически")
                elif similarity > 0.5:
                    st.info("Тексты имеют некоторое сходство")
                else:
                    st.info("Тексты семантически различаются")
                
                # Сохранение в сессию
                st.session_state['emb1'] = emb1
                st.session_state['emb2'] = emb2
                st.session_state['text1'] = text1
                st.session_state['text2'] = text2
            else:
                st.warning("Введите оба текста")
    
    # Детальная информация
    st.markdown("---")
    st.subheader("Детальная информация")
    
    if 'emb1' in st.session_state:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Первые 20 значений вектора 1:**")
            st.code(str(st.session_state['emb1'][:20].round(4)))
        
        with col2:
            st.write("**Первые 20 значений вектора 2:**")
            st.code(str(st.session_state['emb2'][:20].round(4)))
        
        # Гистограмма распределения
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(st.session_state['emb1'], bins=50, alpha=0.7, label='Вектор 1')
        ax.hist(st.session_state['emb2'], bins=50, alpha=0.7, label='Вектор 2')
        ax.set_xlabel("Значение")
        ax.set_ylabel("Частота")
        ax.set_title("Распределение значений эмбеддингов")
        ax.legend()
        st.pyplot(fig)


def page_multimodal_search(embedder: GeminiEmbedder):
    """Страница: Мультимодальный поиск"""
    
    st.header("Мультимодальный поиск")
    st.write("Поиск изображений по тексту и наоборот")
    
    tab1, tab2 = st.tabs(["Текст -> Изображение", "Изображение -> Текст"])
    
    with tab1:
        st.subheader("Поиск изображений по текстовому запросу")
        
        query = st.text_input(
            "Введите текстовый запрос:",
            "Милый пушистый кот играет",
            key="search_query"
        )
        
        # Проверка наличия изображений
        images_dir = Path("data/sample_images")
        available_images = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
        
        if available_images:
            st.write(f"Найдено изображений: {len(available_images)}")
            
            if st.button("Выполнить поиск", type="primary"):
                with st.spinner("Поиск..."):
                    query_emb = embedder.embed_text(query)
                    
                    results = []
                    for img_path in available_images:
                        try:
                            img_emb = embedder.embed_image(str(img_path))
                            sim = embedder.cosine_similarity(query_emb, img_emb)
                            results.append((img_path.name, sim, str(img_path)))
                        except Exception as e:
                            st.warning(f"Ошибка обработки {img_path.name}: {e}")
                    
                    results.sort(key=lambda x: x[1], reverse=True)
                    
                    # Отображение результатов
                    st.subheader("Результаты поиска")
                    
                    for rank, (name, score, path) in enumerate(results[:5], 1):
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.metric(f"#{rank}", f"{score:.3f}")
                        with col2:
                            st.write(f"**{name}**")
                            st.progress(score)
                    
                    # Визуализация
                    if results:
                        fig, ax = plt.subplots(figsize=(10, 5))
                        names = [r[0] for r in results[:5]]
                        scores = [r[1] for r in results[:5]]
                        ax.barh(names, scores, color='steelblue')
                        ax.set_xlabel("Косинусное сходство")
                        ax.set_title(f"Результаты поиска для: '{query}'")
                        ax.set_xlim(0, 1)
                        st.pyplot(fig)
        else:
            st.warning(
                "Изображения не найдены. Поместите файлы в папку data/sample_images/"
            )
    
    with tab2:
        st.subheader("Поиск текстов по изображению")
        
        uploaded_file = st.file_uploader(
            "Загрузите изображение",
            type=['png', 'jpg', 'jpeg'],
            key="upload_img"
        )
        
        # База текстов для поиска
        text_database = {
            "cat": "Пушистый кот играет с клубком",
            "dog": "Собака бежит по парку",
            "car": "Спортивный автомобиль на трассе",
            "nature": "Горный пейзаж с озером",
            "food": "Пицца с моцареллой и базиликом"
        }
        
        if uploaded_file:
            st.image(uploaded_file, caption="Загруженное изображение", width=300)
            
            if st.button("Найти похожие тексты", type="primary"):
                with st.spinner("Обработка изображения..."):
                    # Сохраняем временно
                    temp_path = Path("data/sample_images") / "temp_upload.png"
                    temp_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    
                    # Генерируем эмбеддинг
                    img_emb = embedder.embed_image(str(temp_path))
                    
                    # Ищем среди текстов
                    results = []
                    for label, text in text_database.items():
                        text_emb = embedder.embed_text(text)
                        sim = embedder.cosine_similarity(img_emb, text_emb)
                        results.append((label, text, sim))
                    
                    results.sort(key=lambda x: x[1], reverse=True)
                    
                    st.subheader("Наиболее похожие тексты")
                    
                    for rank, (label, text, score) in enumerate(results, 1):
                        st.write(f"**#{rank} {label}** (сходство: {score:.3f})")
                        st.write(text)
                        st.progress(score)
                        st.markdown("---")
                    
                    # Очистка
                    temp_path.unlink()


def page_visualization(embedder: GeminiEmbedder, visualizer: EmbeddingVisualizer):
    """Страница: Визуализация пространства"""
    
    st.header("Визуализация векторного пространства")
    st.write("PCA и t-SNE проекции эмбеддингов")
    
    # Предопределённые тексты для демонстрации
    default_texts = [
        "Python программирование код",
        "JavaScript веб разработка фронтенд",
        "Машинное обучение нейросети AI",
        "Глубокое обучение компьютерное зрение",
        "Рецепт приготовление еда кухня",
        "Спорт фитнес тренировки здоровье",
        "Путешествия туризм отдых природа",
        "Финансы инвестиции деньги банк"
    ]
    
    st.subheader("Набор данных")
    
    col1, col2 = st.columns(2)
    
    with col1:
        use_custom = st.checkbox("Использовать свои тексты", value=False)
        
        if use_custom:
            custom_texts = st.text_area(
                "Введите тексты (каждый с новой строки):",
                value="\n".join(default_texts[:4]),
                height=200
            )
            texts = [t.strip() for t in custom_texts.split("\n") if t.strip()]
        else:
            texts = default_texts
            st.write(f"Используется {len(texts)} демонстрационных текстов")
    
    with col2:
        st.subheader("Параметры визуализации")
        
        method = st.selectbox(
            "Метод проекции:",
            ["PCA", "t-SNE"],
            index=0
        )
        
        n_components = st.selectbox(
            "Количество компонент:",
            [2, 3],
            index=0
        )
    
    if st.button("Создать визуализацию", type="primary"):
        if len(texts) < 3:
            st.error("Нужно минимум 3 текста для визуализации")
        else:
            with st.spinner("Генерация эмбеддингов и визуализация..."):
                embeddings = embedder.embed_texts(texts)
                
                if method == "PCA":
                    pca = PCA(n_components=n_components)
                    reduced = pca.fit_transform(embeddings)
                    explained_var = pca.explained_variance_ratio_.sum()
                    
                    st.success(f"Объяснённая дисперсия: {explained_var:.2%}")
                else:
                    from sklearn.manifold import TSNE
                    tsne = TSNE(n_components=n_components, perplexity=min(5, len(texts)-1))
                    reduced = tsne.fit_transform(embeddings)
                    explained_var = None
                
                # 2D визуализация
                if n_components == 2:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    scatter = ax.scatter(reduced[:, 0], reduced[:, 1], 
                                        c=range(len(texts)), cmap='tab10', 
                                        s=100, alpha=0.7)
                    
                    for i, text in enumerate(texts):
                        ax.annotate(text[:30], (reduced[i, 0], reduced[i, 1]),
                                   fontsize=8, alpha=0.8)
                    
                    title = f"{method} проекция"
                    if explained_var:
                        title += f" (дисперсия: {explained_var:.2%})"
                    
                    ax.set_title(title)
                    ax.set_xlabel("Компонента 1")
                    ax.set_ylabel("Компонента 2")
                    ax.grid(True, alpha=0.3)
                    
                    st.pyplot(fig)
                else:
                    st.info("3D визуализация доступна в сохранённых файлах")
                    visualizer.plot_pca_3d(embeddings, labels=texts, show=False)
                    st.success("3D график сохранён в results/plots/pca_3d.png")
                
                # Матрица сходства
                st.subheader("Матрица сходства")
                
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                normalized = embeddings / norms
                similarity = np.dot(normalized, normalized.T)
                
                fig, ax = plt.subplots(figsize=(10, 8))
                im = ax.imshow(similarity, cmap='YlOrRd')
                
                short_labels = [t[:15] for t in texts]
                ax.set_xticks(range(len(texts)))
                ax.set_yticks(range(len(texts)))
                ax.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=8)
                ax.set_yticklabels(short_labels, fontsize=8)
                
                for i in range(len(texts)):
                    for j in range(len(texts)):
                        ax.text(j, i, f'{similarity[i, j]:.2f}', 
                               ha="center", va="center", fontsize=7)
                
                plt.colorbar(im, label='Сходство')
                plt.title("Матрица косинусного сходства")
                plt.tight_layout()
                
                st.pyplot(fig)


def page_comparison(embedder: GeminiEmbedder):
    """Страница: Сравнение моделей"""
    
    st.header("Сравнение конфигураций модели")
    st.write("Сравнение эмбеддингов с разными параметрами")
    
    test_text = st.text_area(
        "Текст для сравнения:",
        "Искусственный интеллект и машинное обучение меняют мир",
        height=100
    )
    
    dimensions = st.multiselect(
        "Выберите размерности для сравнения:",
        [128, 768, 1536, 3072],
        default=[128, 768]
    )
    
    if st.button("Сравнить", type="primary") and dimensions:
        results = []
        
        for dim in dimensions:
            with st.spinner(f"Генерация эмбеддинга ({dim})..."):
                embedder.output_dimensionality = dim
                embedder.config.output_dimensionality = dim
                
                emb = embedder.embed_text(test_text)
                
                results.append({
                    "Размерность": dim,
                    "Время": "N/A",
                    "Среднее": f"{np.mean(emb):.4f}",
                    "Стд": f"{np.std(emb):.4f}",
                    "Мин": f"{np.min(emb):.4f}",
                    "Макс": f"{np.max(emb):.4f}"
                })
        
        df = pd.DataFrame(results)
        st.table(df)
        
        # Визуализация распределений
        fig, axes = plt.subplots(1, len(dimensions), figsize=(5*len(dimensions), 4))
        
        if len(dimensions) == 1:
            axes = [axes]
        
        for idx, dim in enumerate(dimensions):
            embedder.output_dimensionality = dim
            embedder.config.output_dimensionality = dim
            emb = embedder.embed_text(test_text)
            
            axes[idx].hist(emb, bins=30, edgecolor='black', alpha=0.7)
            axes[idx].set_title(f"Размерность: {dim}")
            axes[idx].set_xlabel("Значение")
            axes[idx].set_ylabel("Частота")
            axes[idx].grid(True, alpha=0.3)
        
        plt.suptitle("Распределение значений по размерностям")
        plt.tight_layout()
        st.pyplot(fig)


def page_batch_processing(embedder: GeminiEmbedder):
    """Страница: Пакетная обработка"""
    
    st.header("Пакетная обработка текстов")
    st.write("Генерация эмбеддингов для множества текстов")
    
    # Загрузка файлов
    uploaded_file = st.file_uploader(
        "Загрузите TXT файл с текстами (каждый с новой строки)",
        type=['txt'],
        key="batch_upload"
    )
    
    if uploaded_file:
        texts = uploaded_file.read().decode('utf-8').splitlines()
        texts = [t.strip() for t in texts if t.strip()]
        st.write(f"Загружено текстов: {len(texts)}")
    else:
        # Демонстрационные данные
        texts = st.text_area(
            "Или введите тексты вручную:",
            "Текст 1\nТекст 2\nТекст 3\nТекст 4\nТекст 5",
            height=150
        )
        texts = [t.strip() for t in texts.split("\n") if t.strip()]
    
    if texts:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Статистика")
            st.metric("Количество текстов", len(texts))
            st.metric("Средняя длина", f"{np.mean([len(t) for t in texts]):.0f} символов")
        
        with col2:
            st.subheader("Параметры")
            batch_size = st.slider("Размер пакета:", 1, 10, 5)
        
        if st.button("Обработать", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                
                with st.spinner(f"Обработка текстов {i+1}-{min(i+batch_size, len(texts))}..."):
                    batch_embs = embedder.embed_texts(batch)
                    all_embeddings.append(batch_embs)
                
                progress = (i + len(batch)) / len(texts)
                progress_bar.progress(progress)
                status_text.text(f"Обработано {i + len(batch)} из {len(texts)}")
            
            progress_bar.progress(1.0)
            status_text.text("Готово!")
            
            all_embeddings = np.vstack(all_embeddings)
            
            st.success(f"Создано {len(all_embeddings)} эмбеддингов")
            
            # Сохранение
            st.subheader("Сохранение результатов")
            
            np.save("results/embeddings_batch.npy", all_embeddings)
            st.download_button(
                label="Скачать эмбеддинги (.npy)",
                data=open("results/embeddings_batch.npy", "rb"),
                file_name="embeddings.npy",
                mime="application/octet-stream"
            )
            
            # Визуализация
            if len(texts) >= 3:
                st.subheader("Визуализация")
                
                pca = PCA(n_components=2)
                reduced = pca.fit_transform(all_embeddings)
                
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.scatter(reduced[:, 0], reduced[:, 1], alpha=0.6)
                
                for i, text in enumerate(texts[:20]):
                    ax.annotate(text[:20], (reduced[i, 0], reduced[i, 1]), fontsize=7)
                
                ax.set_title(f"PCA проекция ({len(texts)} текстов)")
                ax.set_xlabel("PC1")
                ax.set_ylabel("PC2")
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)


if __name__ == "__main__":
    main()