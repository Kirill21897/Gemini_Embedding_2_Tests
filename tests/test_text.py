"""
Тесты для текстовых эмбеддингов
"""

import pytest
import numpy as np
import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from src.embedder import GeminiEmbedder

@pytest.fixture
def embedder():
    return GeminiEmbedder(
        output_dimensionality=768,
        task_type="SEMANTIC_SIMILARITY"
    )

def test_single_embedding(embedder):
    """Тест одного эмбеддинга"""
    text = "Искусственный интеллект меняет мир"
    embedding = embedder.embed_text(text)
    
    assert embedding is not None
    assert len(embedding) == 768
    assert isinstance(embedding, np.ndarray)
    assert not np.all(embedding == 0)

def test_batch_embeddings(embedder):
    """Тест пакетной обработки"""
    texts = [
        "Привет, мир!",
        "Как дела?",
        "Что нового?"
    ]
    embeddings = embedder.embed_texts(texts)
    
    assert embeddings.shape == (3, 768)

def test_similarity_same_text(embedder):
    """Одинаковый текст должен иметь высокое сходство"""
    text = "Тестовый текст для проверки"
    emb1 = embedder.embed_text(text)
    emb2 = embedder.embed_text(text)
    
    similarity = embedder.cosine_similarity(emb1, emb2)
    assert similarity > 0.99

def test_similarity_different_text(embedder):
    """Разный текст должен иметь меньшее сходство"""
    text1 = "Кот сидит на диване"
    text2 = "Программирование на Python"
    
    emb1 = embedder.embed_text(text1)
    emb2 = embedder.embed_text(text2)
    
    similarity = embedder.cosine_similarity(emb1, emb2)
    assert similarity < 0.8

def test_find_similar(embedder):
    """Тест поиска похожих документов"""
    documents = [
        "Машинное обучение и нейросети",
        "Рецепт борща",
        "Искусственный интеллект в медицине",
        "Как приготовить пельмени",
        "Глубокое обучение для начинающих"
    ]
    
    query = "Нейронные сети и ML"
    results = embedder.find_similar(query, documents, top_k=3)
    
    assert len(results) == 3
    assert "Машинное обучение" in results[0][0] or "Искусственный интеллект" in results[0][0]