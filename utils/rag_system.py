import json
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

class RAGSystem:
    def __init__(self, knowledge_base: Optional[List[Dict]] = None):
        self.encoder = SentenceTransformer('jhgan/ko-sroberta-multitask')
        self.knowledge_base = knowledge_base if knowledge_base else []
        self.embeddings = None
        self.index = None
        self.id_to_doc = {}  # FAISS ID → 문서 매핑
        self._build_index()

    @classmethod
    def from_json(cls, json_path: str) -> 'RAGSystem':
        with open(json_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        knowledge_base = []

        if isinstance(raw, dict) and "evidence" in raw:
            knowledge_base = [
                {
                    "content": sentence,
                    "source": raw.get("source"),
                    "stance": raw.get("stance"),
                    "date": raw.get("date"),
                    "title": raw.get("title"),
                    "url": raw.get("url")
                }
                for sentence in raw["evidence"]
            ]

        elif isinstance(raw, list):
            for doc in raw:
                if "evidence" in doc:
                    knowledge_base.extend([
                        {
                            "content": sentence,
                            "source": doc.get("source"),
                            "stance": doc.get("stance"),
                            "date": doc.get("date"),
                            "title": doc.get("title"),
                            "url": doc.get("url")
                        }
                        for sentence in doc["evidence"]
                    ])
        else:
            raise ValueError("지원되지 않는 JSON 형식입니다.")

        return cls(knowledge_base)

    def _build_index(self):
        if not self.knowledge_base:
            return
        
        texts = [item["content"] for item in self.knowledge_base]
        self.embeddings = self.encoder.encode(texts)
        
        dimension = self.embeddings.shape[1]
        base_index = faiss.IndexFlatIP(dimension)
        self.index = faiss.IndexIDMap2(base_index)

        normalized_embeddings = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        
        ids = np.arange(len(self.knowledge_base))
        self.index.add_with_ids(normalized_embeddings.astype('float32'), ids)

        self.id_to_doc = {i: doc for i, doc in enumerate(self.knowledge_base)}

    def search_relevant_info(self, query: str, top_k: int = 3) -> List[Dict]:
        if not self.index or not self.knowledge_base:
            return []
        
        query_embedding = self.encoder.encode([query])
        query_normalized = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        scores, indices = self.index.search(query_normalized.astype('float32'), top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            if score > 0.3:
                doc = self.id_to_doc.get(idx, {}).copy()
                doc['relevance_score'] = float(score)
                results.append(doc)
        
        return results

    def add_document(self, content: str, source: str, topic: str):
        new_doc = {
            "content": content,
            "source": source,
            "topic": topic
        }
        new_id = len(self.knowledge_base)

        self.knowledge_base.append(new_doc)
        self.id_to_doc[new_id] = new_doc

        new_embedding = self.encoder.encode([content])
        new_embedding = new_embedding / np.linalg.norm(new_embedding, axis=1, keepdims=True)

        if self.index is None:
            dimension = new_embedding.shape[1]
            base_index = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIDMap2(base_index)
            self.embeddings = new_embedding
            self.index.add_with_ids(new_embedding.astype('float32'), np.array([new_id]))
        else:
            self.embeddings = np.vstack([self.embeddings, new_embedding])
            self.index.add_with_ids(new_embedding.astype('float32'), np.array([new_id]))
