"""
Gemini Embedding 2 Client
Основной класс для работы с эмбеддингами
"""

import os
from typing import List, Union, Optional
from google import genai
from google.genai import types
from dotenv import load_dotenv
import numpy as np

load_dotenv()

class GeminiEmbedder:
    """Клиент для Gemini Embedding 2"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-embedding-2-preview",
        output_dimensionality: int = 768,
        task_type: str = "SEMANTIC_SIMILARITY"
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("API ключ не найден. Установите GEMINI_API_KEY")
        
        self.model = model
        self.output_dimensionality = output_dimensionality
        self.task_type = task_type
        
        self.client = genai.Client(api_key=self.api_key)
        self.config = types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=output_dimensionality
        )
    
    def embed_text(self, text: str) -> np.ndarray:
        """Создать эмбеддинг для текста"""
        result = self.client.models.embed_content(
            model=self.model,
            contents=text,
            config=self.config
        )
        return np.array(result.embeddings[0].values)
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Создать эмбеддинги для списка текстов"""
        embeddings = []
        for text in texts:
            emb = self.embed_text(text)
            embeddings.append(emb)
        return np.array(embeddings)
    
    def embed_image(self, image_path: str, description: str = "") -> np.ndarray:
        """Создать эмбеддинг для изображения"""
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        
        parts = [types.Part.from_bytes(
            data=image_bytes, 
            mime_type='image/png'
        )]
        
        if description:
            parts.insert(0, types.Part(text=description))
        
        result = self.client.models.embed_content(
            model=self.model,
            contents=[types.Content(parts=parts)],
            config=self.config
        )
        return np.array(result.embeddings[0].values)
    
    def cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Вычислить косинусное сходство между двумя векторами"""
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    
    def find_similar(
        self, 
        query: str, 
        documents: List[str], 
        top_k: int = 5
    ) -> List[tuple]:
        """Найти наиболее похожие документы"""
        query_emb = self.embed_text(query)
        doc_embs = self.embed_texts(documents)
        
        similarities = [
            self.cosine_similarity(query_emb, doc_emb) 
            for doc_emb in doc_embs
        ]
        
        indexed_sims = list(enumerate(similarities))
        sorted_sims = sorted(indexed_sims, key=lambda x: x[1], reverse=True)
        
        return [
            (documents[idx], float(sim)) 
            for idx, sim in sorted_sims[:top_k]
        ]