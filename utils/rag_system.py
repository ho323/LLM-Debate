from llama_index.core import VectorStoreIndex, Document, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from typing import List, Dict, Optional
import json
import faiss

class RAGSystem:
    def __init__(self, progressive_path: str, conservative_path: str):
        self.progressive_path = progressive_path
        self.conservative_path = conservative_path

        # 1. 임베딩 모델 불러오기
        self.embed_model = HuggingFaceEmbedding(model_name="jhgan/ko-sroberta-multitask")

        # 2. FAISS 인덱스 초기화
        faiss_index = faiss.IndexFlatL2(768)
        self.vector_store = FaissVectorStore(faiss_index=faiss_index)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

        self.index = None
        self.documents = []

        self._load_documents()

    def _load_documents(self):
        """진보 및 보수 문서 JSON을 로드하여 벡터 인덱스 생성"""
        all_docs = []

        for path, stance in [
            (self.progressive_path, "진보"),
            (self.conservative_path, "보수")
        ]:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for article in data:
                    # 🔁 evidence 리스트를 하나의 텍스트로 병합
                    full_text = "\n".join(article.get("evidence", []))

                    metadata = {
                        "title": article.get("title", ""),
                        "source": article.get("source", ""),
                        "url": article.get("url", ""),
                        "date": article.get("date", ""),
                        "stance": stance,
                    }

                    doc = Document(text=full_text, metadata=metadata)
                    all_docs.append(doc)

        self.documents = all_docs

        self.index = VectorStoreIndex.from_documents(
            documents=self.documents,
            embed_model=self.embed_model,
            storage_context=self.storage_context
        )

    def search(self, query: str, stance_filter: Optional[str] = None, top_k: int = 5) -> List[Dict]:
        """질의어(query)를 바탕으로 관련 문단을 벡터 검색"""
        retriever = self.index.as_retriever(similarity_top_k=top_k)
        retrieved_nodes = retriever.retrieve(query)

        results = []
        for node in retrieved_nodes:
            meta = node.node.metadata
            if stance_filter and meta.get("stance") != stance_filter:
                continue
            results.append({
                "text": node.node.text,
                "score": node.score,
                "title": meta.get("title"),
                "source": meta.get("source"),
                "url": meta.get("url"),
                "date": meta.get("date"),
                "stance": meta.get("stance")
            })
        return results
