import requests
import re
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

class RAGSystem:
    def __init__(self):
        self.encoder = SentenceTransformer('jhgan/ko-sroberta-multitask')
        self.knowledge_base = []
        self.embeddings = None
        self.index = None
        self._initialize_knowledge_base()
    
    def _initialize_knowledge_base(self):
        """기본 한국 정치 관련 지식베이스를 초기화합니다."""
        # 예시 정치 뉴스/보고서 데이터
        sample_data = [
            {
                "content": "한국의 복지 지출은 OECD 평균보다 낮은 수준으로, GDP 대비 사회보장 지출이 12.2%로 OECD 평균 20.1%에 못 미친다.",
                "source": "OECD 사회보장 통계 2023",
                "topic": "사회복지"
            },
            {
                "content": "최저임금 인상이 소상공인과 중소기업에 부담을 주지만, 저소득층의 소득 증대와 소비 진작 효과도 있다는 연구 결과가 발표되었다.",
                "source": "한국노동연구원 2023",
                "topic": "최저임금"
            },
            {
                "content": "정부의 시장 개입이 과도할 경우 민간 투자 위축과 비효율성을 초래할 수 있으나, 적절한 개입은 시장 실패를 보완할 수 있다.",
                "source": "경제학회 정책 연구 2023",
                "topic": "정부개입"
            },
            {
                "content": "부동산 가격 안정화를 위한 정부 규제가 시장 자율성을 해치지만, 주거 안정성 확보를 위해서는 필요하다는 의견이 대립하고 있다.",
                "source": "부동산 정책 연구소 2023",
                "topic": "부동산정책"
            },
            {
                "content": "탄소중립 달성을 위한 정부 주도 정책이 산업계 부담을 가중시키지만, 장기적 지속가능성을 위해서는 불가피하다는 분석이다.",
                "source": "환경정책평가연구원 2023",
                "topic": "환경정책"
            }
        ]
        
        self.knowledge_base = sample_data
        self._build_index()
    
    def _build_index(self):
        """벡터 인덱스를 구축합니다."""
        if not self.knowledge_base:
            return
        
        texts = [item["content"] for item in self.knowledge_base]
        self.embeddings = self.encoder.encode(texts)
        
        # FAISS 인덱스 구축
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product (cosine similarity)
        
        # 정규화 후 인덱스에 추가
        normalized_embeddings = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.index.add(normalized_embeddings.astype('float32'))
    
    def extract_keywords(self, text: str) -> List[str]:
        """텍스트에서 키워드를 추출합니다."""
        # 한국어 정치 관련 키워드 패턴
        political_keywords = [
            r'복지', r'최저임금', r'세금', r'정부', r'시장', r'규제', r'자유',
            r'평등', r'공정', r'경제', r'사회', r'정책', r'개입', r'민간',
            r'공공', r'보장', r'지원', r'투자', r'성장', r'안정', r'개혁'
        ]
        
        keywords = []
        for pattern in political_keywords:
            if re.search(pattern, text):
                keywords.append(pattern.replace('r\'', '').replace('\'', ''))
        
        return list(set(keywords))
    
    def search_relevant_info(self, query: str, top_k: int = 3) -> List[Dict]:
        """쿼리와 관련된 정보를 검색합니다."""
        if not self.index or not self.knowledge_base:
            return []
        
        # 쿼리 임베딩
        query_embedding = self.encoder.encode([query])
        query_normalized = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        # 유사도 검색
        scores, indices = self.index.search(query_normalized.astype('float32'), min(top_k, len(self.knowledge_base)))
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if score > 0.3:  # 유사도 임계값
                result = self.knowledge_base[idx].copy()
                result['relevance_score'] = float(score)
                results.append(result)
        
        return results
    
    def add_document(self, content: str, source: str, topic: str):
        """새로운 문서를 지식베이스에 추가합니다."""
        new_doc = {
            "content": content,
            "source": source,
            "topic": topic
        }
        self.knowledge_base.append(new_doc)
        self._build_index()  # 인덱스 재구축 