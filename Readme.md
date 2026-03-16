# 🧬 Gemini Embedding 2 Tests

Тестирование и эксперименты с мультимодальной моделью эмбеддингов **Google Gemini Embedding 2**.

## Возможности

- Текстовые эмбеддинги
- Мультимодальные эмбеддинги (текст + изображение)
- Поиск семантически похожих документов
- Визуализация векторного пространства
- Сравнение разных конфигураций модели

## Установка

```bash
# Клонировать репозиторий
git clone https://github.com/yourusername/gemini-embedding2-tests.git
cd gemini-embedding2-tests

# Создать виртуальное окружение
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Установить зависимости
pip install -r requirements.txt

# Настроить API ключ
cp .env.example .env
# Отредактируйте .env и добавьте ваш API ключ