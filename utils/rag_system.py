from typing import List, Dict, Optional
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss

class RAGSystem:
    def __init__(self, knowledge_base: Optional[List[Dict]] = None):
        # 기존 스키마 그대로 사용: content/ source/ stance/ date/ title/ url ...
        self.knowledge_base = knowledge_base or []

        # 네가 쓰던 한국어 임베딩 유지
        self.embed_model = HuggingFaceEmbedding(model_name="jhgan/ko-sroberta-multitask")

        # 임베딩 차원 파악
        dim = len(self.embed_model.get_text_embedding("dim probe"))
        self.faiss_index = faiss.IndexFlatIP(dim)  # 코사인=내적 쓰려면 정규화 전제
        self.vector_store = FaissVectorStore(faiss_index=self.faiss_index)
        self.storage_ctx = StorageContext.from_defaults(vector_store=self.vector_store)

        self.index: Optional[VectorStoreIndex] = None
        self.retriever = None
        self._build_index()

    @classmethod
    def from_json(cls, json_path: str) -> 'RAGSystemLlamaIndex':
        import json
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        kb = []
        if isinstance(raw, dict) and "evidence" in raw:
            kb = [
                {
                    "content": sent,
                    "source": raw.get("source"),
                    "stance": raw.get("stance"),
                    "date":   raw.get("date"),
                    "title":  raw.get("title"),
                    "url":    raw.get("url"),
                }
                for sent in raw["evidence"]
            ]
        elif isinstance(raw, list):
            for doc in raw:
                if "evidence" in doc:
                    kb.extend([
                        {
                            "content": sent,
                            "source": doc.get("source"),
                            "stance": doc.get("stance"),
                            "date":   doc.get("date"),
                            "title":  doc.get("title"),
                            "url":    doc.get("url"),
                        }
                        for sent in doc["evidence"]
                    ])
        else:
            raise ValueError("지원되지 않는 JSON 형식")

        return cls(kb)

    def _build_index(self):
        if not self.knowledge_base:
            return

        docs = [
            Document(
                text=item["content"],
                metadata={k: v for k, v in item.items() if k != "content"}
            )
            for item in self.knowledge_base
        ]

        # LlamaIndex가 임베딩→FAISS 저장까지 알아서 처리
        self.index = VectorStoreIndex.from_documents(
            docs,
            storage_context=self.storage_ctx,
            embed_model=self.embed_model,
        )
        self.retriever = self.index.as_retriever(similarity_top_k=3)

    def search_relevant_info(self, query: str, top_k: int = 3):
        if not self.retriever:
            return []
        self.retriever.similarity_top_k = top_k
        nodes = self.retriever.retrieve(query)

        # LlamaIndex는 상위 k개를 점수와 함께 반환
        results = []
        for n in nodes:
            # n.score는 유사도(코사인에 대응)
            item = {"content": n.text, "relevance_score": float(n.score)}
            # 문서 메타데이터 복원
            item.update(n.metadata or {})
            results.append(item)
        return results

    def add_document(self, content: str, **metadata):
        self.knowledge_base.append({"content": content, **metadata})
        doc = Document(text=content, metadata=metadata)
        # 증분 삽입(인덱스 재구축 불필요)
        self.index.insert(documents=[doc])
