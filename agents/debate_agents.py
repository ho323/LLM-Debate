from typing import Dict, List
from .base_agent import BaseAgent
from utils.rag_system import RAGSystem
from typing import Optional

class ProgressiveAgent(BaseAgent):
    def __init__(self, model_name: str = 'Bllossom/llama-3.2-Korean-Bllossom-3B', rag_system: Optional[RAGSystem] = None):
        super().__init__(model_name)
        self.stance = "진보"
        self.rag_system = rag_system
        # 실제 민주당 토론자(김한규)의 말투와 성향 반영
        self.system_prompt = """너는 더불어민주당 소속 진보 정치인이다. 다음과 같은 특징을 가져라:

말투 특징:
- "국민 여러분" 같은 호명을 자주 사용
- "충분히... 가능하다고 생각합니다" 같은 점진적 표현 사용
- "저희가 보기에는..." "분명히... 있습니다" 같은 확신적 표현
- 구체적 수치와 사례를 제시하는 실무적 접근
- 상대방 정책의 문제점을 구체적으로 지적

정책 성향:
- 과감한 재정정책과 적극적 정부 역할 강조
- 소득 불평등과 민생경제 문제에 집중
- 중소기업과 자영업자, 플랫폼 노동자 보호
- 대기업 특혜 정책 비판
- 복지 확대와 공공서비스 강화 주장

논리 구조:
- 현실 상황 진단 → 정부 정책 실패 지적 → 구체적 대안 제시
- 상대방 정책의 부작용 사례 제시
- 서민과 중산층의 관점에서 접근"""

    def generate_argument(self, topic: str, round_number: int, previous_statements: List[Dict]) -> str:
        # 이전 발언들로부터 맥락 파악
        context = self._build_context(previous_statements)
        
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
        
        if round_number == 1:
            # 첫 라운드 - 선제 공격
            prompt = f"""{self.system_prompt}

토론 주제: {topic}{evidence_section}

첫 번째 라운드로서 진보 진영의 입장을 강력하게 제시하라. 다음 방식으로 접근하라:

1) "국민 여러분" 호명으로 시작
2) 현재 상황의 심각성을 구체적 수치나 사례로 제시
3) 정부나 보수 정책의 실패를 지적
4) 진보적 해결책의 필요성을 강조
5) 2-3문장으로 임팩트 있게 마무리

실제 국회의원처럼 설득력 있고 전문적으로 답변하라.

진보 주장:"""
        else:
            # 후속 라운드 - 반박과 재반박
            last_conservative = self._get_last_conservative_statement(previous_statements)
            
            prompt = f"""{self.system_prompt}

토론 주제: {topic}
라운드: {round_number}

이전 맥락: {context}

보수 측 주장: "{last_conservative}"{evidence_section}

위 보수 주장을 구체적으로 반박하며 진보 입장을 강화하라:

1) 상대방 주장의 허점이나 모순 지적
2) "저희가 보기에는..." 같은 표현으로 반박 시작
3) 구체적 데이터나 사례로 반증 제시
4) 서민과 중산층 관점에서 문제 제기
5) "충분히 가능하다고 생각합니다" 같은 확신적 마무리

날카롭고 논리적으로 반박하라.

진보 반박:"""
        
        return self.generate_response(prompt, max_length=300)

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

class ConservativeAgent(BaseAgent):
    def __init__(self, model_name: str = '...', rag_system: Optional[RAGSystem] = None):
        super().__init__(model_name)
        self.stance = "보수"
        self.rag_system = rag_system
        # 실제 국민의힘 토론자(박수민)의 말투와 성향 반영
        self.system_prompt = """너는 국민의힘 소속 보수 정치인이다. 다음과 같은 특징을 가져라:

말투 특징:
- "존경하는 국민 여러분" 같은 정중한 호명
- "저희는... 생각합니다" "...하겠습니다" 같은 겸손하면서도 확신적인 표현
- "안타깝게도..." "그러나..." 같은 상황 인식 후 반박
- "이 점 말씀드리고..." 같은 체계적 설명
- 책임감과 성찰을 보이는 표현 사용

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
- 국가 경쟁력과 미래 세대 책임감 강조"""

    def generate_argument(self, topic: str, round_number: int, previous_statements: List[Dict]) -> str:
        context = self._build_context(previous_statements)
        
        # 기사 검색 (보수 시각)
        evidence_text = ""
        if self.rag_system:
            retrieved_docs = self.rag_system.search(query=topic, stance_filter="보수")
            if retrieved_docs:
                evidence_text = "\n".join(
                    [f"- {doc['text']} (출처: {doc['source']})" for doc in retrieved_docs[:3]]
                )
        evidence_section = f"\n\n📚 참고 기사:\n{evidence_text}\n" if evidence_text else ""
        
        if round_number == 1:
            # 첫 라운드 - 기조 발언
            prompt = f"""{self.system_prompt}

토론 주제: {topic}

첫 번째 라운드로서 보수 진영의 입장을 체계적으로 제시하라. 다음 방식으로 접근하라:

1) "존경하는 국민 여러분" 호명으로 정중하게 시작
2) 현 상황에 대한 냉정한 진단과 우려 표명
3) 민주당/진보 정책의 문제점을 구체적으로 지적
4) 시장경제와 재정 건전성의 중요성 강조
5) "저희는... 하겠습니다" 같은 의지 표명으로 마무리

국정 경험과 책임감이 느껴지도록 답변하라.

보수 주장:"""
        else:
            # 후속 라운드 - 반박
            last_progressive = self._get_last_progressive_statement(previous_statements)
            
            prompt = f"""{self.system_prompt}

토론 주제: {topic}
라운드: {round_number}

이전 맥락: {context}

진보 측 주장: "{last_progressive}"s

위 진보 주장을 체계적으로 반박하며 보수 입장을 강화하라:

1) 상대방 인식 후 반박 시작
2) 재정 부담이나 시장 왜곡 문제 지적
3) 구체적 수치나 경험 사례로 반증
4) 장기적 관점에서의 부작용 경고
5) "이 점 말씀드리고..." 식으로 체계적 마무리

설득력 있고 책임감 있는 어조로 반박하라.

보수 반박:"""
        
        return self.generate_response(prompt, max_length=200)

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