# 🧬 Gemini Embedding 2 Tests

Проект для тестирования и экспериментов с мультимодальными эмбеддингами на базе **Google Gemini Embedding 2**.

##  Описание

В этом репозитории реализован простой и расширяемый модуль на Python, позволяющий:

- генерировать эмбеддинги текста и изображений
- сравнивать документы по косинусному сходству
- искать семантически схожие фрагменты
- строить визуализации PCA/t-SNE/heatmap/кластеризацию
- запускать web-интерфейс на Streamlit
- прогонять автоматические тесты через pytest

##  Структура проекта

- `src/embedder.py` — клиент для Gemini Embedding 2.
- `src/utils.py` — утилиты загрузки/сохранения и простая визуализация.
- `src/visualizer.py` — класс `EmbeddingVisualizer` с PCA/t-SNE/k-means/heatmap.
- `app/streamlit_app.py` — Streamlit UI для интерактивных тестов.
- `examples/` — набор сценариев использования (визуализации, мультимодальный поиск и т.д.).
- `data/sample_images/` — примеры изображений для мультимодального поиска.
- `tests/` — модульные тесты.

##  Установка

```bash
# Клонировать репозиторий
git clone https://github.com/yourusername/gemini-embedding2-tests.git
cd gemini-embedding2-tests

# Создать и активировать виртуальное окружение
python -m venv venv
# Linux/macOS
source venv/bin/activate
# Windows PowerShell
venv\Scripts\Activate.ps1

# Установить зависимости
pip install -r requirements.txt
```

##  Конфигурация API

Скопируйте файл:

```bash
cp .env.example .env
```

Отредактируйте `.env`:

```ini
GEMINI_API_KEY=ваш_ключ
EMBEDDING_MODEL=gemini-embedding-2-preview
OUTPUT_DIMENSIONALITY=768
TASK_TYPE=SEMANTIC_SIMILARITY
```

> Если ключ не указан, `GeminiEmbedder` выбросит `ValueError`.

## ▶ Быстрый старт

```python
from src.embedder import GeminiEmbedder

embedder = GeminiEmbedder()
emb = embedder.embed_text('Привет, мир!')
print(emb.shape)

img_emb = embedder.embed_image('data/sample_images/example.png')
print(img_emb.shape)

sim = embedder.cosine_similarity(emb, img_emb)
print(sim)
```

##  Запуск Streamlit интерфейса

```bash
streamlit run app/streamlit_app.py
```

Интерфейс содержит:
- текстовые эмбеддинги и сравнение 2-х фраз
- мультимодальный поиск (текст→изображение, изображение→текст)
- визуализация пространства эмбеддингов (PCA, t-SNE, heatmap, кластеры)
- сравнение моделей (разные task_type и размерность)
- пакетную обработку

##  Примеры

- `examples/05_visualization.py`: генерация PCA/t-SNE/heatmap/кластеров + сравнение.
- `examples/06_interactive_viz.py`: интерактивная работа с точками и подсказками.
- `examples/07_multimodal_space.py`: анализ текста и изображений в общем пространстве.
- `examples/08_cross_modal_search.py`: кросс-модальный поиск.

Запуск любого примера:

```bash
python examples/05_visualization.py
```

##  Тестирование

```bash
pytest -q
```

Важно:
- для `tests/test_multimodal.py` требуется `data/sample_images/test.png`.
- если файла нет, тест будет пропущен (`@pytest.mark.skipif`).

##  Параметры модели

`GeminiEmbedder` принимает:
- `api_key` (необязательно, из `.env` берётся автоматически)
- `model` (по умолчанию `gemini-embedding-2-preview`)
- `output_dimensionality` (128/768/1536/3072)
- `task_type` (`SEMANTIC_SIMILARITY` и др.)

##  Полезные советы

- Для масштабных сценариев кэшируйте эмбеддинги (`save_embeddings/load_embeddings`).
- Проверяйте Hit/Miss на `cosine_similarity` при подборе порога.
- Для визуализации данных из `visualizer` смотрите сохранённые PNG в `results/plots`.

##  Зависимости

Описание в `requirements.txt`:
- `google-genai`, `python-dotenv`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `plotly`, `Pillow`, `tqdm`, `streamlit`, `pytest`, `pytest-asyncio`.
