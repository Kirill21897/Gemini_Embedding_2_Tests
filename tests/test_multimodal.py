"""
Тесты для мультимодальных эмбеддингов
"""

import pytest
import os
import sys
import pathlib as Path

root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from src.embedder import GeminiEmbedder

@pytest.fixture
def embedder():
    return GeminiEmbedder(output_dimensionality=768)

@pytest.mark.skipif(
    not os.path.exists("data/sample_images/test.png"),
    reason="Тестовое изображение не найдено"
)
def test_image_embedding(embedder):
    """Тест эмбеддинга изображения"""
    image_path = "data/sample_images/test.png"
    embedding = embedder.embed_image(
        image_path, 
        description="Тестовое изображение"
    )
    
    assert embedding is not None
    assert len(embedding) == 768

def test_different_dimensions():
    """Тест разных размерностей векторов"""
    embedder_128 = GeminiEmbedder(output_dimensionality=128)
    embedder_768 = GeminiEmbedder(output_dimensionality=768)
    embedder_3072 = GeminiEmbedder(output_dimensionality=3072)
    
    text = "Тестовый текст"
    
    emb_128 = embedder_128.embed_text(text)
    emb_768 = embedder_768.embed_text(text)
    emb_3072 = embedder_3072.embed_text(text)
    
    assert len(emb_128) == 128
    assert len(emb_768) == 768
    assert len(emb_3072) == 3072