from typing import Dict, List, Tuple, Optional, Set
from .base_agent import BaseAgent
from utils.rag_system import RAGSystem
import re
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class EvidenceItem:
    """개별 근거 항목"""
    text: str
    category: str
    normalized: str
    confidence: float
    timestamp: datetime
    stance: str
    vector: np.ndarray

class EnhancedEvidenceTracker:
    """실제 토론 데이터 기반 강화된 근거 추적 시스템"""
    
    def __init__(self):
        self.used_evidence = {
            "진보": {}, 
            "보수": {}
        }
        
        # 실제 토론에서 발견된 패턴을 반영한 강화된 정규식
        self.evidence_patterns = {
            # 통계 및 수치 (실제 토론에서 사용된 패턴들)
            "statistics": [
                r'(\d+(?:\.\d+)?%)',  # 백분율: 3.6%, 12%, 등
                r'(GDP\s*대비\s*\d+(?:\.\d+)?%)',  # GDP 대비: GDP 대비 104%
                r'(\d+(?:\.\d+)?조\s*원?)',  # 조 단위: 1조 원
                r'(\d+(?:\.\d+)?억\s*원?)',  # 억 단위
                r'(\d+(?:\.\d+)?%?p)',  # 포인트: 0.8%p, 2%포인트
                r'(\d+(?:\.\d+)?배)',  # 배수: 3배
                r'(평균\s*\d+(?:\.\d+)?%)',  # 평균: 평균 25%
                r'(연평균\s*\d+(?:\.\d+)?%)',  # 연평균: 연평균 7%
                r'(\d+년\s*내\s*최고치)',  # 기간: 10년 내 최고치
            ],
            
            # 기관 및 출처 (동일 기관 다른 표기 통합)
            "sources": [
                r'(한국은행|BOK|중앙은행)',
                r'(통계청|KOSTAT|국가통계포털)',  
                r'(한국개발연구원|KDI)',  # 중요: KDI와 한국개발연구원 통합
                r'(기획재정부|기재부|재정부)',
                r'(OECD|경제협력개발기구)',
                r'(IMF|국제통화기금)',
                r'(국정감사|국감)',
                r'(가계동향조사)',  # 누락되었던 중요 조사
                r'(소비자물가지수|CPI)',  # 누락되었던 지표
                r'(국세청)',
                r'(전경련|한국경제인연합회)',
                r'(한국경제연구원)',
                r'(자유기업원)',
                r'(노동연구원)',
                r'(참여연대)',
            ],
            
            # 국가 및 지역 사례
            "examples": [
                r'(독일[\w\s]*(?:사례|모델|정책|경험|제도))',
                r'(일본[\w\s]*(?:사례|모델|정책|경험|제도))', 
                r'(미국[\w\s]*(?:사례|모델|정책|경험|제도))',
                r'(중국[\w\s]*(?:사례|모델|정책|경험|제도))',
                r'(프랑스[\w\s]*(?:사례|모델|정책|경험|제도))',
                r'(영국[\w\s]*(?:사례|모델|정책|경험|제도))',
                r'(스웨덴[\w\s]*(?:사례|모델|정책|경험|제도))',
                r'(덴마크[\w\s]*(?:사례|모델|정책|경험|제도))',
                r'(싱가포르[\w\s]*(?:사례|모델|정책|경험|제도))',
                r'(\d{4}년[\w\s]*(?:사례|사례에서|당시))',  # 연도별 사례: 2021년 사례
                r'(선진국\s*평균)',  # 누락되었던 비교 기준
            ],
            
            # 정책 및 제도
            "policies": [
                r'(소비쿠폰[\w\s]*정책?)',
                r'(기본소득[\w\s]*정책?)',  
                r'(전국민고용보험)',
                r'(그린뉴딜|한국판뉴딜)',
                r'(규제샌드박스)',
                r'(세제혜택|세제지원)',
                r'(공공요금\s*동결)',  # 실제 토론에서 언급
                r'(대기업\s*탈세\s*감시)',  # 실제 토론에서 언급
                r'(구조\s*개혁)',  # 실제 토론에서 언급
                r'(R&D\s*지원)',  # 실제 토론에서 언급
                r'(규제완화|규제\s*완화)',
                r'(재정\s*건전성)',  # 핵심 개념
                r'(재정적자|재정\s*적자)',  # 누락되었던 중요 개념
                r'(국가채무)',  # 누락되었던 중요 개념
            ],
            
            # 경제 지표 (새로 추가된 카테고리)
            "economic_indicators": [
                r'(소비자물가\s*상승률)',
                r'(기준금리)',
                r'(가계대출\s*금리)',
                r'(가계부채)',
                r'(실질소득)',
                r'(소비심리)',
                r'(소비증가율)',
                r'(매출\s*증가율)',  
                r'(경제성장률)',
                r'(소비\s*회복률)',
                r'(물가\s*상승률)',
            ]
        }
        
        # 기관명 정규화 매핑 (동일 기관 다른 표기 통합)
        self.institution_mapping = {
            'kdi': '한국개발연구원',
            '한국개발연구원': '한국개발연구원',
            'bok': '한국은행', 
            '한국은행': '한국은행',
            '중앙은행': '한국은행',
            'kostat': '통계청',
            '통계청': '통계청',
            '국가통계포털': '통계청',
            '기재부': '기획재정부',
            '기획재정부': '기획재정부',
            '재정부': '기획재정부',
            'oecd': 'OECD',
            '경제협력개발기구': 'OECD',
            'imf': 'IMF',
            '국제통화기금': 'IMF',
            'cpi': '소비자물가지수',
            '소비자물가지수': '소비자물가지수'
        }
        
        # 대안 근거 제안 (실제 토론 스타일 반영)
        self.alternative_suggestions = {
            "진보": [
                "민주노총 자료", "참여연대 보고서", "경제사회노동위원회 분석",
                "시민사회단체 연구", "진보정책연구소 자료", "한겨레경제사회연구원 보고서",
                "노동연구원 통계", "사회정책연합 분석", "공공운수노조 조사"
            ],
            "보수": [
                "전경련 경영자료", "한국경제연구원 보고서", "자유기업원 분석",
                "대한상공회의소 조사", "중소기업중앙회 자료", "한국무역협회 통계",
                "재정학회 연구", "한국조세재정연구원 분석"
            ]
        }
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),  # 3~5자 n-gram
            min_df=1
        )
    
    def extract_evidence(self, statement: str) -> Dict[str, List[str]]:
        """강화된 근거 추출"""
        evidence = {category: [] for category in self.evidence_patterns.keys()}
        
        for category, patterns in self.evidence_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, statement, re.IGNORECASE)
                if matches:
                    for match in matches:
                        if match.strip():
                            evidence[category].append(match.strip())
        
        return evidence
    
    def normalize_evidence(self, evidence_text: str, category: str = "") -> str:
        """향상된 근거 정규화"""
        normalized = evidence_text.lower().strip()
        
        # 기관명 통합
        for variant, standard in self.institution_mapping.items():
            if variant in normalized:
                normalized = normalized.replace(variant, standard.lower())
        
        # 숫자 표기 통일
        normalized = re.sub(r'(\d+)조\s*원?', r'\1조', normalized)
        normalized = re.sub(r'(\d+)억\s*원?', r'\1억', normalized) 
        normalized = re.sub(r'(\d+(?:\.\d+)?)%', r'\1%', normalized)
        normalized = re.sub(r'(\d+(?:\.\d+)?)%?p', r'\1%p', normalized)
        
        # 연도 통합 (2021년, 21년 등)
        normalized = re.sub(r'20(\d{2})년', r'20\1년', normalized)
        
        # 공백 정리
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    def _to_vec(self, texts: List[str]) -> np.ndarray:
        # 벡터화: 비교 집합을 동시 변환
        return self.vectorizer.fit_transform(texts)

    def calculate_similarity(self, text1: str, text2: str) -> float:
        corpus = [text1, text2]
        X = self._to_vec(corpus)
        sim = float(cosine_similarity(X[0], X[1])[0][0])
        return sim
    
    def record_used_evidence(self, statement: str, stance: str):
        evidence = self.extract_evidence(statement)
        timestamp = datetime.now()

        for category, items in evidence.items():
            for item in items:
                normalized = self.normalize_evidence(item, category)
                if normalized and len(normalized) > 2:
                    existing_key = self._find_similar_evidence(normalized, stance, category)
                    if existing_key:
                        self.used_evidence[stance][existing_key].timestamp = timestamp
                    else:
                        # 새 항목 벡터를 만들기 위해 비교군과 함께 fit_transform
                        all_texts = [normalized] + [k for k, v in self.used_evidence[stance].items() if v.category == category]
                        X = self._to_vec(all_texts)
                        vec = X[0].toarray()[0]
                        evidence_item = EvidenceItem(
                            text=item,
                            category=category,
                            normalized=normalized,
                            confidence=self._calculate_confidence(item, category),
                            timestamp=timestamp,
                            stance=stance,
                            vector=vec,
                        )
                        self.used_evidence[stance][normalized] = evidence_item
    
    def _find_similar_evidence(self, normalized: str, stance: str, category: str, threshold: float = 0.80) -> str:
        # 비교 대상들이 이미 저장되어 있다면, 같은 벡터 공간으로 변환
        candidates = [(k, v) for k, v in self.used_evidence[stance].items() if v.category == category]
        if not candidates:
            return None
        corpus = [normalized] + [k for k, _ in candidates]
        X = self._to_vec(corpus)
        sims = cosine_similarity(X[0], X[1:])[0]
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        return candidates[best_idx][0] if best_sim >= threshold else None

    def check_evidence_conflict(self, statement: str, stance: str) -> Tuple[bool, List[str]]:
        opponent_stance = "보수" if stance == "진보" else "진보"
        evidence = self.extract_evidence(statement)
        conflicting_evidence = []
        for category, items in evidence.items():
            opp_candidates = [(k, v) for k, v in self.used_evidence[opponent_stance].items() if v.category == category]
            for item in items:
                normalized = self.normalize_evidence(item, category)
                if normalized in self.used_evidence[opponent_stance]:
                    conflicting_evidence.append(item)
                    continue
                if opp_candidates:
                    corpus = [normalized] + [k for k, _ in opp_candidates]
                    X = self._to_vec(corpus)
                    sims = cosine_similarity(X[0], X[1:])[0]
                    if float(np.max(sims)) >= 0.78:  # TF-IDF
                        conflicting_evidence.append(item)
        return (len(conflicting_evidence) > 0, conflicting_evidence)
    
    def _calculate_confidence(self, text: str, category: str) -> float:
        """근거의 신뢰도 점수 계산"""
        confidence = 0.5  # 기본값
        
        # 구체적 수치가 있으면 점수 증가
        if re.search(r'\d+', text):
            confidence += 0.2
        
        # 권위 있는 기관명이 있으면 점수 증가  
        authority_keywords = ['한국은행', '통계청', 'oecd', 'imf', 'kdi']
        if any(keyword in text.lower() for keyword in authority_keywords):
            confidence += 0.2
            
        # 최신 연도가 있으면 점수 증가
        if re.search(r'202[0-9]년', text):
            confidence += 0.1
            
        return min(confidence, 1.0)
    
    def get_alternative_evidence_prompt(self, conflicting_items: List[str], stance: str) -> str:
        """맥락에 맞는 대안 근거 제안"""
        if not conflicting_items:
            return ""
        
        suggestions = self.alternative_suggestions.get(stance, [])[:4]
        conflicting_text = ", ".join(conflicting_items[:3])  # 최대 3개만 표시
        
        warning = f"""
⚠️ 근거 중복 경고: 다음 근거들은 상대방이 이미 사용했습니다
   중복 근거: {conflicting_text}

💡 {stance} 관점의 독립적 근거를 활용하세요:
   추천 근거: {', '.join(suggestions)}

📋 중복 방지 가이드:
   • 같은 기관이라도 다른 시점의 자료를 사용하세요
   • 상대방과 다른 해석 관점을 제시하세요  
   • {stance} 성향 기관의 독립적 분석을 인용하세요
   • 구체적인 데이터 제시하세요 
"""
        return warning

class StatementMemoryManager:
    """발언 메모리 관리를 위한 헬퍼 클래스"""
    
    def __init__(self, max_statements: int = 8):
        self.max_statements = max_statements
        
    def summarize_statement(self, statement: str, agent) -> str:
        """발언을 핵심 논점으로 요약"""
        prompt = f"""다음 발언의 핵심 논점을 100자 근처로 요약해주세요:

발언: "{statement}"

핵심 논점만 간단히 정리하세요 (예: "재정정책 확대 필요", "시장경제 원리 강조"):"""
        
        summary = agent.generate_response(prompt)
        return summary.strip() if summary else statement[:50]
    
    def detect_contradiction(self, new_statement: str, past_statement: str, agent) -> bool:
        """새 발언이 과거 발언과 모순되는지 검증"""
        prompt = f"""다음 두 발언이 서로 모순되는지 판단해주세요:

과거 발언: "{past_statement}"
새 발언: "{new_statement}"

모순된다면 "YES", 모순되지 않는다면 "NO"로만 답해주세요:"""
        
        result = agent.generate_response(prompt)
        return "YES" in result.upper() if result else False
    
    def extract_key_topics(self, statements: List[str], agent) -> List[str]:
        """발언들에서 핵심 주제들을 추출"""
        if not statements:
            return []
            
        combined_text = " ".join(statements[-3:])  # 최근 3개 발언만 사용
        
        prompt = f"""다음 발언들에서 핵심 주제 3개를 추출해주세요:

발언들: "{combined_text}"

핵심 주제만 간단히 나열하세요 (예: "재정정책", "일자리", "부동산"):"""
        
        result = agent.generate_response(prompt)
        if result:
            topics = [topic.strip() for topic in result.split(",")]
            return topics[:3]
        return []
    
    def manage_memory(self, statements: List[str], agent) -> List[Dict]:
        """메모리를 효율적으로 관리"""
        if len(statements) <= self.max_statements:
            return [{"statement": stmt, "summary": self.summarize_statement(stmt, agent)} 
                   for stmt in statements]
        
        # 중요도 기반 선별 (최근 발언 우선, 핵심 주제 포함 발언 우선)
        managed_statements = []
        
        # 최근 6개는 무조건 포함
        recent_statements = statements[-6:]
        for stmt in recent_statements:
            managed_statements.append({
                "statement": stmt,
                "summary": self.summarize_statement(stmt, agent),
                "priority": "recent"
            })
        
        # 나머지 중에서 핵심 주제 포함 발언 선별
        older_statements = statements[:-6] if len(statements) > 6 else []
        key_topics = self.extract_key_topics(statements, agent)
        
        for stmt in older_statements:
            if any(topic.lower() in stmt.lower() for topic in key_topics):
                managed_statements.append({
                    "statement": stmt,
                    "summary": self.summarize_statement(stmt, agent),
                    "priority": "key_topic"
                })
                if len(managed_statements) >= self.max_statements:
                    break
        
        return managed_statements

class ProgressiveAgent(BaseAgent):
    def __init__(self, model_path: str = 'C:/Users/User/Documents/EXAONE-4.0-32B-Q4_K_M.gguf', rag_system: Optional[RAGSystem] = None, evidence_tracker: Optional[EnhancedEvidenceTracker] = None):
        super().__init__(model_path)
        self.stance = "진보"
        self.rag_system = rag_system
        self.memory_manager = StatementMemoryManager()
        self.evidence_tracker = evidence_tracker or EnhancedEvidenceTracker()
        
        # 과거 발언 추적을 위한 저장소 (원본 + 관리된 버전)
        self.my_previous_statements = []
        self.opponent_previous_statements = []
        self.my_managed_statements = []
        self.opponent_managed_statements = []
        
        # 핵심 논점 추적
        self.my_key_arguments = []
        self.consistency_violations = []
        
        # 실제 민주당 토론자(김한규)의 말투와 성향 반영
        self.system_prompt = """너는 더불어민주당 소속 진보 정치인이다. 다음과 같은 특징을 가져라:

말투 특징:
- "국민 여러분" 같은 호명을 자주 사용
- "충분히... 가능하다고 생각합니다" 같은 점진적 표현 사용
- "저희가 보기에는..." "분명히... 있습니다" 같은 확신적 표현
- 구체적 수치와 사례를 제시하는 실무적 접근
- 상대방 정책의 문제점을 구체적으로 지적
- "진보적" 과 같은 직접적 말은 빼기

정책 성향:
- 과감한 재정정책과 적극적 정부 역할 강조
- 소득 불평등과 민생경제 문제에 집중
- 중소기업과 자영업자, 플랫폼 노동자 보호
- 대기업 특혜 정책 비판
- 복지 확대와 공공서비스 강화 주장

논리 구조:
- 현실 상황 진단 → 정부 정책 실패 지적 → 구체적 대안 제시
- 상대방 정책의 부작용 사례 제시
- 서민과 중산층의 관점에서 접근

형식 제한(매우 중요):
- 출력은 항상 '한 단락'의 평서문으로 작성한다.
- 문장은 반드시 끝맺는다. 문장이 끊길 것 같을 시 전 문장에서 마무리 한다.
- 줄바꿈, 제목, 머리말, 소제목 금지.
- 목록, 번호(1. ① 1), 하이픈(-), 불릿(•), 대시(—, –), 이모지 사용 금지.
- 문장 시작에 숫자/괄호/불릿/이모지 배치 금지.
- 발화자의 멘트만 출력한다.

다음과 같은 논리적 사고 과정을 거쳐라:
<thinking>
1. 상황 분석: 현재 주어진 주제의 핵심 문제는 무엇인가?
2. 약점 파악: 그들 주장의 허점이나 모순점은 무엇인가?
3. 반박 근거: 우리가 제시할 수 있는 반증 데이터나 사례는?
4. 진보 대안: 우리의 해결책이 왜 더 나은가?
5. 감정적 호소: 국민들의 공감을 얻을 수 있는 포인트는?
</thinking>

"""

    def update_statement_history(self, previous_statements: List[Dict]):
        """발언 기록을 업데이트하고 메모리 관리"""
        self.my_previous_statements = []
        self.opponent_previous_statements = []
        
        for stmt in previous_statements:
            statement_text = stmt.get('statement', '')
            stance = stmt.get('stance', '')
            
            if stance == '진보':
                self.my_previous_statements.append(statement_text)
                # 내 발언의 근거를 기록
                self.evidence_tracker.record_used_evidence(statement_text, '진보')
            elif stance == '보수':
                self.opponent_previous_statements.append(statement_text)
                # 상대 발언의 근거를 기록
                self.evidence_tracker.record_used_evidence(statement_text, '보수')
        
        # 메모리 관리 적용
        if self.my_previous_statements:
            self.my_managed_statements = self.memory_manager.manage_memory(
                self.my_previous_statements, self)
        
        if self.opponent_previous_statements:
            self.opponent_managed_statements = self.memory_manager.manage_memory(
                self.opponent_previous_statements, self)
    
    def check_evidence_before_response(self, potential_statement: str) -> Tuple[bool, str]:
        """근거 중복을 사전에 확인"""
        has_conflict, conflicting_items = self.evidence_tracker.check_evidence_conflict(
            potential_statement, self.stance)
        
        if has_conflict:
            warning = self.evidence_tracker.get_alternative_evidence_prompt(
                conflicting_items, self.stance)
            return False, warning
        
        return True, ""
    
    def check_consistency_before_response(self, new_statement: str) -> Tuple[bool, str]:
        """새 발언의 일관성을 검증"""
        if not self.my_previous_statements:
            return True, ""
        
        # 최근 6개 발언과 비교
        recent_statements = self.my_previous_statements[-6:]
        for past_stmt in recent_statements:
            if self.memory_manager.detect_contradiction(new_statement, past_stmt, self):
                warning = f"⚠️ 일관성 경고: 과거 발언 '{past_stmt[:50]}...'과 모순될 수 있습니다."
                self.consistency_violations.append({
                    "new": new_statement[:50],
                    "conflicting": past_stmt[:50]
                })
                return False, warning
        
        return True, ""
    
    def get_my_key_arguments(self) -> List[str]:
        """내 핵심 논점들을 반환"""
        if not self.my_managed_statements:
            return []
        
        return [stmt["summary"] for stmt in self.my_managed_statements 
                if stmt.get("priority") in ["recent", "key_topic"]]
    
    def get_opponent_key_arguments(self) -> List[str]:
        """상대 핵심 논점들을 반환"""
        if not self.opponent_managed_statements:
            return []
        
        return [stmt["summary"] for stmt in self.opponent_managed_statements 
                if stmt.get("priority") in ["recent", "key_topic"]]

    def generate_argument(self, topic: str, round_number: int, previous_statements: List[Dict]) -> str:
        # 발언 기록 업데이트
        self.update_statement_history(previous_statements)
        
        context = self._build_context(previous_statements)

        ##### RAG #####
        # 관련 기사 검색(진보 시각)
        evidence_text = ""
        if self.rag_system:
            retrieved_docs = self.rag_system.search(query=topic, stance_filter="진보")
            if retrieved_docs:
                evidence_text = "\n".join(
                    [f"- {doc['text']} (출처: {doc['source']})" for doc in retrieved_docs[:3]]
                )

        # 공통적으로 프롬프트에 삽입
        evidence_section = f"\n\n📚 참고 기사:\n{evidence_text}\n" if evidence_text else ""
        ##### RAG #####

        # 핵심 논점 기반 발언 기록 섹션 생성
        my_key_args = self.get_my_key_arguments()
        my_arguments_section = ""
        if my_key_args:
            my_arguments_text = ", ".join(my_key_args[:5])  # 최대 5개
            my_arguments_section = f"\n\n📝 내가 강조한 핵심 논점들: {my_arguments_text}\n"

        opponent_key_args = self.get_opponent_key_arguments()
        opponent_arguments_section = ""
        if opponent_key_args:
            opponent_arguments_text = ", ".join(opponent_key_args[:5])  # 최대 5개
            opponent_arguments_section = f"\n\n🔴 상대(보수)의 핵심 논점들: {opponent_arguments_text}\n"

        # 일관성 위반 경고
        consistency_warning = ""
        if self.consistency_violations:
            recent_violation = self.consistency_violations[-1]
            consistency_warning = f"\n\n⚠️ 일관성 주의: 과거 '{recent_violation['conflicting']}'과 모순되지 않도록 주의하세요.\n"

        # 근거 중복 방지 지침
        evidence_guidelines = f"""
📋 근거 사용 지침:
- 상대방이 이미 사용한 통계, 사례, 정책은 피하세요
- {self.stance} 관점의 독립적 자료를 활용하세요
- 같은 기관 자료라도 다른 시점이나 다른 지표를 사용하세요
- 근거의 출처를 명확히 구분하여 제시하세요
"""

        if round_number == 1:
            prompt = f"""너는 더불어민주당 소속 진보 정치인이다.

토론 주제: {topic}{evidence_section}{evidence_guidelines}

먼저 다음 단계별로 논리적 사고를 진행하라:
<thinking>
1. 상황 분석: 현재 경제/사회 상황의 핵심 문제는 무엇인가?
2. 근거 제시: 우리가 제시할 수 있는 데이터나 사례는?
3. 핵심 메시지: 국민들에게 전달할 책임감 있는 대안은?
4. 감정적 호소: 국민들의 공감을 얻을 수 있는 포인트는?
</thinking>

그 다음 정중한 호칭을 포함하되 과장 없이, 존댓말로 구체적 수치·사례로 현재 상황의 심각성을 제시하고, 정부나 보수 정책의 실패를 비판하며, 진보적 대안의 필요성을 분명히 밝힌 뒤 2~3문장으로 힘 있게 마무리하라.

형식 제한: <thinking> 부분은 출력하지 말고, 줄바꿈 없이 단락 하나로만 작성하고, 목록·숫자·괄호 시작·하이픈·불릿·이모지·제목을 사용하지 마라. 발화자의 멘트만 출력하라."""
        else:
            last_conservative = self._get_last_conservative_statement(previous_statements)
            
            # 근거 중복 체크를 위한 임시 응답 생성
            temp_prompt = f"""상대 주장 '{last_conservative}'에 대한 반박 논점 3가지를 간단히 나열하세요:"""
            temp_response = self.generate_response(temp_prompt)
            
            # 근거 중복 확인
            evidence_ok, evidence_warning = self.check_evidence_before_response(temp_response)
            evidence_instruction = evidence_warning if not evidence_ok else evidence_guidelines
            
            prompt = f"""너는 더불어민주당 소속 진보 정치인이다.

토론 주제: {topic}
상대(보수)의 최근 주장: "{last_conservative}"{evidence_section}{my_arguments_section}{opponent_arguments_section}{consistency_warning}{evidence_instruction}

먼저 다음 단계별로 논리적 사고를 진행하라:
<thinking>
1. 상대 분석: 상대가 최근에 주장한 부분이 무엇인가?
2. 과거 논점 검토: 내가 이미 강조한 핵심 논점과 어떻게 연결할 것인가?
3. 상대 모순점 파악: 상대의 과거 논점과 현재 발언 사이의 모순이나 허점은?
4. 약점 파악: 그들 주장의 허점이나 모순점은 무엇인가?
5. 반박 근거: 우리가 제시할 수 있는 반증 데이터나 사례는?
6. 진보 대안: 우리의 해결책이 왜 더 나은가?
7. 일관성 확인: 내 과거 논점과 일치하는가?
</thinking>

중요한 제약사항:
- 내가 과거에 강조한 핵심 논점들과 일관성을 유지하라
- 상대의 최근 발언과 과거 핵심 논점을 모두 고려하여 정확한 반박을 하라
- 새로운 각도에서 접근하되 기존 논점을 발전시켜라
- 상대방이 이미 사용한 근거(통계, 사례, 정책)는 절대 사용하지 마라
- 보수 관점의 독립적이고 차별화된 근거만 활용하라라
- 상대방이 이미 사용한 근거(통계, 사례, 정책)는 절대 사용하지 마라
- 진보 관점의 독립적이고 차별화된 근거만 활용하라

그 다음 보수 측의 최근 주장을 정확히 요지 파악한 뒤, 존댓말로 구체적 데이터와 사례로 반증하고, 서민·중산층 관점에서 일관된 대안을 제시하며 공격적으로 마무리하라.

형식 제한: <thinking> 부분과 보수 측 주장은 출력하지 말고, 목록·숫자·괄호 시작·하이픈·불릿·이모지·제목을 사용하지 마라. 발화자의 멘트만 출력하라."""
        
        # 응답 생성
        response = self.generate_response(prompt)
        
        # 일관성 및 근거 중복 검증
        if response:
            is_consistent, consistency_warning = self.check_consistency_before_response(response)
            has_evidence_conflict, evidence_conflict_warning = self.check_evidence_before_response(response)
            
            if not is_consistent:
                print(f"[DEBUG 일관성] {consistency_warning}")
            
            if has_evidence_conflict:
                print(f"[DEBUG 근거중복] {evidence_conflict_warning}")
                # 근거 중복이 발견된 경우 재생성 시도
                retry_prompt = prompt + f"\n\n{evidence_conflict_warning}\n위 경고를 반영하여 다시 작성하세요:"
                response = self.generate_response(retry_prompt)
            
            # 새로운 발언을 기록에 추가 및 근거 추적
            self.my_previous_statements.append(response)
            self.evidence_tracker.record_used_evidence(response, self.stance)
        
        return response

    def _build_context(self, statements: List[Dict]) -> str:
        if not statements:
            return "첫 라운드입니다."
        
        recent_statements = statements[-2:] if len(statements) >= 2 else statements
        context_parts = []
        for stmt in recent_statements:
            stance = stmt.get('stance', '')
            content = stmt.get('statement', '')[:50] + "..."
            context_parts.append(f"{stance}: {content}")
        
        return " | ".join(context_parts)

    def _get_last_conservative_statement(self, statements: List[Dict]) -> str:
        for stmt in reversed(statements):
            if stmt.get('stance') == '보수':
                return stmt.get('statement', '')
        return ""

    def process_input(self, input_data: Dict) -> str:
        """기존 인터페이스와의 호환성을 위한 메서드"""
        topic = input_data.get('topic', '')
        round_number = input_data.get('round_number', 1)
        previous_statements = input_data.get('previous_statements', [])
        
        return self.generate_argument(topic, round_number, previous_statements)

    def get_memory_status(self) -> Dict:
        """메모리 상태 정보 반환"""
        return {
            "my_statements_count": len(self.my_previous_statements),
            "my_managed_count": len(self.my_managed_statements),
            "opponent_managed_count": len(self.opponent_managed_statements),
            "consistency_violations": len(self.consistency_violations),
            "key_arguments": self.get_my_key_arguments(),
            "used_evidence": list(self.evidence_tracker.used_evidence[self.stance]),
            "opponent_evidence": list(self.evidence_tracker.used_evidence["보수"])
        }

class ConservativeAgent(BaseAgent):
    def __init__(self, model_path: str = 'C:/Users/User/Documents/EXAONE-4.0-32B-Q4_K_M.gguf', rag_system: Optional[RAGSystem] = None, evidence_tracker: Optional[EnhancedEvidenceTracker] = None):
        super().__init__(model_path)
        self.stance = "보수"
        self.rag_system = rag_system
        self.memory_manager = StatementMemoryManager()
        self.evidence_tracker = evidence_tracker or EnhancedEvidenceTracker()
        
        # 과거 발언 추적을 위한 저장소 (원본 + 관리된 버전)
        self.my_previous_statements = []
        self.opponent_previous_statements = []
        self.my_managed_statements = []
        self.opponent_managed_statements = []
        
        # 핵심 논점 추적
        self.my_key_arguments = []
        self.consistency_violations = []
        
        # 실제 국민의힘 토론자(박수민)의 말투와 성향 반영
        self.system_prompt = """너는 국민의힘 소속 보수 정치인이다. 다음과 같은 특징을 가져라:

말투 특징:
- "저희는... 생각합니다" "...하겠습니다" 같은 겸손하면서도 확신적인 표현
- "안타깝게도..." "그러나..." 같은 상황 인식 후 반박
- "이 점 말씀드리고..." 같은 체계적 설명
- 책임감과 성찰을 보이는 표현 사용
- 구체적 수치와 데이터를 활용한 실증적 접근
- "보수적" 과 같은 직접적 말은 빼기

정책 성향:
- 시장경제와 민간 주도 성장 강조
- 재정 건전성과 국가부채 우려
- 규제 완화와 기업 투자 환경 개선
- 개인 책임과 자유 선택의 가치
- 혁신과 도전 정신 중시

논리 구조:
- 상황 인식 → 상대방 정책의 문제점 지적 → 시장경제적 해법 제시
- 재정 부담과 장기적 부작용 경고
- 성공 사례와 경험적 근거 제시
- 국가 경쟁력과 미래 세대 책임감 강조

형식 제한(매우 중요):
- 출력은 항상 '한 단락'의 평서문으로 작성한다.
- 문장은 반드시 끝맺는다. 문장이 끊길 것 같을 시 전 문장에서 마무리 한다.
- 줄바꿈, 제목, 머리말, 소제목 금지.
- 목록, 번호(1. ① 1), 하이픈(-), 불릿(•), 대시(—, –), 이모지 사용 금지.
- 문장 시작에 숫자/괄호/불릿/이모지 배치 금지.
- 발화자의 멘트만 출력한다.

다음과 같은 논리적 사고 과정을 거쳐라:
<thinking>
1. 상대방 주장 분석: 진보 측 주장의 핵심 논리는 무엇인가?
2. 근거 제시: 우리가 제시할 수 있는 구체적 데이터나 사례는?
3. 보수적 관점: 시장경제와 재정건전성 관점에서 어떻게 바라보는가?
4. 장기적 부작용: 진보 정책이 경제와 재정에 미칠 장기 영향은?
5. 대안 제시: 시장 원리 기반의 실현 가능한 해법은?
</thinking>

"""

    def update_statement_history(self, previous_statements: List[Dict]):
        """발언 기록을 업데이트하고 메모리 관리"""
        self.my_previous_statements = []
        self.opponent_previous_statements = []
        
        for stmt in previous_statements:
            statement_text = stmt.get('statement', '')
            stance = stmt.get('stance', '')
            
            if stance == '보수':
                self.my_previous_statements.append(statement_text)
                # 내 발언의 근거를 기록
                self.evidence_tracker.record_used_evidence(statement_text, '보수')
            elif stance == '진보':
                self.opponent_previous_statements.append(statement_text)
                # 상대 발언의 근거를 기록
                self.evidence_tracker.record_used_evidence(statement_text, '진보')
        
        # 메모리 관리 적용
        if self.my_previous_statements:
            self.my_managed_statements = self.memory_manager.manage_memory(
                self.my_previous_statements, self)
        
        if self.opponent_previous_statements:
            self.opponent_managed_statements = self.memory_manager.manage_memory(
                self.opponent_previous_statements, self)

    def check_evidence_before_response(self, potential_statement: str) -> Tuple[bool, str]:
        """근거 중복을 사전에 확인"""
        has_conflict, conflicting_items = self.evidence_tracker.check_evidence_conflict(
            potential_statement, self.stance)
        
        if has_conflict:
            warning = self.evidence_tracker.get_alternative_evidence_prompt(
                conflicting_items, self.stance)
            return False, warning
        
        return True, ""

    def check_consistency_before_response(self, new_statement: str) -> Tuple[bool, str]:
        """새 발언의 일관성을 검증"""
        if not self.my_previous_statements:
            return True, ""
        
        # 최근 3개 발언과 비교
        recent_statements = self.my_previous_statements[-3:]
        for past_stmt in recent_statements:
            if self.memory_manager.detect_contradiction(new_statement, past_stmt, self):
                warning = f"⚠️ 일관성 경고: 과거 발언 '{past_stmt[:50]}...'과 모순될 수 있습니다."
                self.consistency_violations.append({
                    "new": new_statement[:50],
                    "conflicting": past_stmt[:50]
                })
                return False, warning
        
        return True, ""

    def get_my_key_arguments(self) -> List[str]:
        """내 핵심 논점들을 반환"""
        if not self.my_managed_statements:
            return []
        
        return [stmt["summary"] for stmt in self.my_managed_statements 
                if stmt.get("priority") in ["recent", "key_topic"]]
    
    def get_opponent_key_arguments(self) -> List[str]:
        """상대 핵심 논점들을 반환"""
        if not self.opponent_managed_statements:
            return []
        
        return [stmt["summary"] for stmt in self.opponent_managed_statements 
                if stmt.get("priority") in ["recent", "key_topic"]]

    def generate_argument(self, topic: str, round_number: int, previous_statements: List[Dict]) -> str:
        # 발언 기록 업데이트
        self.update_statement_history(previous_statements)
        
        context = self._build_context(previous_statements)

        ##### RAG #####
        # 기사 검색 (보수 시각)
        evidence_text = ""
        if self.rag_system:
            retrieved_docs = self.rag_system.search(query=topic, stance_filter="보수")
            if retrieved_docs:
                evidence_text = "\n".join(
                    [f"- {doc['text']} (출처: {doc['source']})" for doc in retrieved_docs[:3]]
                )
        evidence_section = f"\n\n📚 참고 기사:\n{evidence_text}\n" if evidence_text else ""
        ##### RAG #####

        # 핵심 논점 기반 발언 기록 섹션 생성
        my_key_args = self.get_my_key_arguments()
        my_arguments_section = ""
        if my_key_args:
            my_arguments_text = ", ".join(my_key_args[:5])  # 최대 5개
            my_arguments_section = f"\n\n📝 내가 강조한 핵심 논점들: {my_arguments_text}\n"

        opponent_key_args = self.get_opponent_key_arguments()
        opponent_arguments_section = ""
        if opponent_key_args:
            opponent_arguments_text = ", ".join(opponent_key_args[:5])  # 최대 5개
            opponent_arguments_section = f"\n\n🔵 상대(진보)의 핵심 논점들: {opponent_arguments_text}\n"

        # 일관성 위반 경고
        consistency_warning = ""
        if self.consistency_violations:
            recent_violation = self.consistency_violations[-1]
            consistency_warning = f"\n\n⚠️ 일관성 주의: 과거 '{recent_violation['conflicting']}'과 모순되지 않도록 주의하세요.\n"

        # 근거 중복 방지 지침
        evidence_guidelines = f"""
📋 근거 사용 지침:
- 상대방이 이미 사용한 통계, 사례, 정책은 피하세요
- {self.stance} 관점의 독립적 자료를 활용하세요
- 같은 기관 자료라도 다른 시점이나 다른 지표를 사용하세요
- 근거의 출처를 명확히 구분하여 제시하세요
"""

        if round_number == 1:
            prompt = f"""너는 국민의힘 소속 보수 정치인이다.

토론 주제: {topic}{evidence_section}{evidence_guidelines}

먼저 다음 단계별로 논리적 사고를 진행하라:
<thinking>
1. 상황 분석: 현재 경제/사회 상황의 핵심 문제는 무엇인가?
2. 근거 제시: 우리가 제시할 수 있는 구체적 데이터나 사례는?
3. 보수적 관점: 시장경제와 재정건전성 관점에서 어떻게 바라보는가?
4. 장기적 부작용: 진보 정책이 경제와 재정에 미칠 장기 영향은?
5. 대안 제시: 시장 원리 기반의 실현 가능한 해법은?
</thinking>

그 다음 현 상황을 구체적 수치와 데이터로 냉정히 진단하고 존댓말로 우려를 밝힌 다음, 진보 정책의 문제점을 경험적 근거와 함께 지적하고, 시장경제·재정건전성의 중요성을 실증적 데이터로 강조하며 책임 있는 어조로 마무리하라.

형식 제한: <thinking> 부분은 출력하지 말고, 줄바꿈 없이 단락 하나로만 작성하고, 목록·숫자·괄호 시작·하이픈·불릿·이모지·제목을 절대 사용하지 마라. 발화자의 멘트만 출력하라."""
        else:
            last_progressive = self._get_last_progressive_statement(previous_statements)
            
            # 근거 중복 체크를 위한 임시 응답 생성
            temp_prompt = f"""상대 주장 '{last_progressive}'에 대한 반박 논점 3가지를 간단히 나열하세요:"""
            temp_response = self.generate_response(temp_prompt)
            
            # 근거 중복 확인
            evidence_ok, evidence_warning = self.check_evidence_before_response(temp_response)
            evidence_instruction = evidence_warning if not evidence_ok else evidence_guidelines
            
            prompt = f"""너는 국민의힘 소속 보수 정치인이다.

토론 주제: {topic}
상대(진보)의 최근 주장: "{last_progressive}"{evidence_section}{my_arguments_section}{opponent_arguments_section}{consistency_warning}{evidence_instruction}

먼저 다음 단계별로 논리적 사고를 진행하라:
<thinking>
1. 상대방 주장 분석: 진보 측이 최근에 주장한 핵심 논리는 무엇인가?
2. 과거 논점 검토: 내가 이미 강조한 핵심 논점과 어떻게 연결할 것인가?
3. 상대 모순점 파악: 상대의 과거 논점과 현재 발언 사이의 모순이나 허점은?
4. 근거 제시: 우리가 제시할 수 있는 구체적 데이터나 사례는?
5. 장기적 부작용: 진보 정책이 경제와 재정에 미칠 장기 영향은?
6. 대안 제시: 시장 원리 기반의 실현 가능한 해법은?
7. 일관성 확인: 내 과거 논점과 일치하는가?
</thinking>

중요한 제약사항:
- 내가 과거에 강조한 핵심 논점들과 일관성을 유지하라
- 상대의 최근 발언과 과거 핵심 논점을 모두 고려하여 정확한 반박을 하라
- 새로운 각도에서 접근하되 기존 논점을 발전시켜라

그 다음 상대의 최근 주장을 존댓말로 논리적으로 반박하고, 구체적 수치와 경험적 데이터로 재정 부담·장기 부작용을 입증하며, 실증적 근거를 들어 일관된 보수적 해법을 제시하고 존댓말이지만 공격적으로 마무리하라.

형식 제한: <thinking> 부분과 진보 측 주장은 출력하지 말고, 목록·숫자·괄호 시작·하이픈·불릿·이모지·제목을 절대 사용하지 마라. 발화자의 멘트만 출력하라."""
        
        # 응답 생성
        response = self.generate_response(prompt)
        
        # 일관성 검증
        if response:
            is_consistent, warning = self.check_consistency_before_response(response)
            if not is_consistent:
                print(f"[DEBUG] {warning}")  # 개발용 로그
            
            # 새로운 발언을 기록에 추가
            self.my_previous_statements.append(response)
        
        return response

    def _build_context(self, statements: List[Dict]) -> str:
        if not statements:
            return "첫 라운드입니다."
        
        recent_statements = statements[-2:] if len(statements) >= 2 else statements
        context_parts = []
        for stmt in recent_statements:
            stance = stmt.get('stance', '')
            content = stmt.get('statement', '')[:50] + "..."
            context_parts.append(f"{stance}: {content}")
        
        return " | ".join(context_parts)

    def _get_last_progressive_statement(self, statements: List[Dict]) -> str:
        for stmt in reversed(statements):
            if stmt.get('stance') == '진보':
                return stmt.get('statement', '')
        return ""

    def process_input(self, input_data: Dict) -> str:
        """기존 인터페이스와의 호환성을 위한 메서드"""
        topic = input_data.get('topic', '')
        round_number = input_data.get('round_number', 1)
        previous_statements = input_data.get('previous_statements', [])
        
        return self.generate_argument(topic, round_number, previous_statements)

    def get_memory_status(self) -> Dict:
        """메모리 상태 정보 반환"""
        return {
            "my_statements_count": len(self.my_previous_statements),
            "my_managed_count": len(self.my_managed_statements),
            "opponent_managed_count": len(self.opponent_managed_statements),
            "consistency_violations": len(self.consistency_violations),
            "key_arguments": self.get_my_key_arguments()
        }

# 사용 예제 및 테스트 함수
def test_memory_management():
    """메모리 관리 기능 테스트"""
    
    # 진보 에이전트 생성
    progressive_agent = ProgressiveAgent()
    
    # 가상의 토론 기록
    test_statements = [
        {"stance": "진보", "statement": "재정정책을 대폭 확대해서 일자리를 늘려야 합니다."},
        {"stance": "보수", "statement": "재정확대는 국가부채만 늘릴 뿐입니다."},
        {"stance": "진보", "statement": "중소기업 지원금을 두 배로 늘려서 경제를 살려야 합니다."},
        {"stance": "보수", "statement": "지원금보다는 규제완화가 우선입니다."},
        {"stance": "진보", "statement": "복지예산을 확대해서 서민생활을 보장해야 합니다."},
        {"stance": "보수", "statement": "복지확대는 재정건전성을 해칩니다."},
    ]
    
    # 메모리 관리 테스트
    progressive_agent.update_statement_history(test_statements)
    
    print("=== 메모리 관리 테스트 결과 ===")
    status = progressive_agent.get_memory_status()
    print(f"전체 발언 수: {status['my_statements_count']}")
    print(f"관리된 발언 수: {status['my_managed_count']}")
    print(f"핵심 논점들: {status['key_arguments']}")
    
    # 일관성 검증 테스트
    test_new_statement = "재정정책은 축소해야 한다고 생각합니다."
    is_consistent, warning = progressive_agent.check_consistency_before_response(test_new_statement)
    print(f"\n=== 일관성 검증 테스트 ===")
    print(f"새 발언: {test_new_statement}")
    print(f"일관성: {'유지' if is_consistent else '위반'}")
    if warning:
        print(f"경고: {warning}")

if __name__ == "__main__":
    test_memory_management()