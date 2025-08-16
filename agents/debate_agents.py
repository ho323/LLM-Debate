from typing import Dict, List
from .base_agent import BaseAgent
from utils.rag_system import RAGSystem
from typing import Optional

class ProgressiveAgent(BaseAgent):
    def __init__(self, model_path: str = 'C:/Users/User/Documents/EXAONE-4.0-32B-Q4_K_M.gguf', rag_system: Optional[RAGSystem] = None):
        super().__init__(model_path)
        self.stance = "진보"
        self.rag_system = rag_system
        # 과거 발언 추적을 위한 저장소
        self.my_previous_statements = []
        self.opponent_previous_statements = []
        
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
        """발언 기록을 업데이트합니다."""
        self.my_previous_statements = []
        self.opponent_previous_statements = []
        
        for stmt in previous_statements:
            if stmt.get('stance') == '진보':
                self.my_previous_statements.append(stmt.get('statement', ''))
            elif stmt.get('stance') == '보수':
                self.opponent_previous_statements.append(stmt.get('statement', ''))

    def get_my_previous_statements(self) -> List[str]:
        """내가 과거에 한 발언들을 반환합니다."""
        return self.my_previous_statements.copy()

    def get_opponent_previous_statements(self) -> List[str]:
        """상대가 과거에 한 발언들을 반환합니다."""
        return self.opponent_previous_statements.copy()

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

        # 과거 발언 기록 섹션 생성
        my_statements_section = ""
        if self.my_previous_statements:
            my_statements_text = "\n".join([f"- {stmt[:100]}..." for stmt in self.my_previous_statements])
            my_statements_section = f"\n\n📝 내가 과거에 한 주요 발언들:\n{my_statements_text}\n"

        opponent_statements_section = ""
        if self.opponent_previous_statements:
            opponent_statements_text = "\n".join([f"- {stmt[:100]}..." for stmt in self.opponent_previous_statements])
            opponent_statements_section = f"\n\n🔴 상대(보수)가 과거에 한 주요 발언들:\n{opponent_statements_text}\n"

        if round_number == 1:
            prompt = f"""너는 더불어민주당 소속 진보 정치인이다.

토론 주제: {topic}{evidence_section}

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
            prompt = f"""너는 더불어민주당 소속 진보 정치인이다.

토론 주제: {topic}
상대(보수)의 최근 주장: "{last_conservative}"{evidence_section}{my_statements_section}{opponent_statements_section}

먼저 다음 단계별로 논리적 사고를 진행하라:
<thinking>
1. 상대 분석: 상대가 최근에 주장한 부분이 무엇인가?
2. 과거 발언 검토: 내가 이미 한 발언과 중복되지 않는 새로운 논점은?
3. 상대 모순점 파악: 상대의 과거 발언과 현재 발언 사이의 모순이나 허점은?
4. 약점 파악: 그들 주장의 허점이나 모순점은 무엇인가?
5. 반박 근거: 우리가 제시할 수 있는 반증 데이터나 사례는?
6. 진보 대안: 우리의 해결책이 왜 더 나은가?
</thinking>

중요한 제약사항:
- 내가 과거에 한 발언과 유사한 내용은 반복하지 마라
- 상대의 최근 발언과 과거 발언을 모두 고려하여 정확한 반박을 하라
- 새로운 논점과 근거를 제시하라

그 다음 보수 측의 최근 주장을 정확히 요지 파악한 뒤, 존댓말로 구체적 데이터와 사례로 반증하고, 서민·중산층 관점에서 새로운 대안을 제시하며 공격적으로 마무리하라.

형식 제한: <thinking> 부분과 보수 측 주장은 출력하지 말고, 목록·숫자·괄호 시작·하이픈·불릿·이모지·제목을 사용하지 마라. 발화자의 멘트만 출력하라."""
        
        response = self.generate_response(prompt)
        
        # 새로운 발언을 기록에 추가
        if response:
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
    def __init__(self, model_path: str = 'C:/Users/User/Documents/EXAONE-4.0-32B-Q4_K_M.gguf', rag_system: Optional[RAGSystem] = None):
        super().__init__(model_path)
        self.stance = "보수"
        self.rag_system = rag_system
        # 과거 발언 추적을 위한 저장소
        self.my_previous_statements = []
        self.opponent_previous_statements = []
        
        # 실제 국민의힘 토론자(박수민)의 말투와 성향 반영
        self.system_prompt = """너는 국민의힘 소속 보수 정치인이다. 다음과 같은 특징을 가져라:

말투 특징:
- "저희는... 생각합니다" "...하겠습니다" 같은 겸손하면서도 확신적인 표현
- "안타깝게도..." "그러나..." 같은 상황 인식 후 반박
- "이 점 말씀드리고..." 같은 체계적 설명
- 책임감과 성찰을 보이는 표현 사용
- 구체적 수치와 데이터를 활용한 실증적 접근

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
        """발언 기록을 업데이트합니다."""
        self.my_previous_statements = []
        self.opponent_previous_statements = []
        
        for stmt in previous_statements:
            if stmt.get('stance') == '보수':
                self.my_previous_statements.append(stmt.get('statement', ''))
            elif stmt.get('stance') == '진보':
                self.opponent_previous_statements.append(stmt.get('statement', ''))

    def get_my_previous_statements(self) -> List[str]:
        """내가 과거에 한 발언들을 반환합니다."""
        return self.my_previous_statements.copy()

    def get_opponent_previous_statements(self) -> List[str]:
        """상대가 과거에 한 발언들을 반환합니다."""
        return self.opponent_previous_statements.copy()

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

        # 과거 발언 기록 섹션 생성
        my_statements_section = ""
        if self.my_previous_statements:
            my_statements_text = "\n".join([f"- {stmt[:100]}..." for stmt in self.my_previous_statements])
            my_statements_section = f"\n\n📝 내가 과거에 한 주요 발언들:\n{my_statements_text}\n"

        opponent_statements_section = ""
        if self.opponent_previous_statements:
            opponent_statements_text = "\n".join([f"- {stmt[:100]}..." for stmt in self.opponent_previous_statements])
            opponent_statements_section = f"\n\n🔵 상대(진보)가 과거에 한 주요 발언들:\n{opponent_statements_text}\n"

        if round_number == 1:
            prompt = f"""너는 국민의힘 소속 보수 정치인이다.

토론 주제: {topic}{evidence_section}

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
            prompt = f"""너는 국민의힘 소속 보수 정치인이다.

토론 주제: {topic}
상대(진보)의 최근 주장: "{last_progressive}"{evidence_section}{my_statements_section}{opponent_statements_section}

먼저 다음 단계별로 논리적 사고를 진행하라:
<thinking>
1. 상대방 주장 분석: 진보 측이 최근에 주장한 핵심 논리는 무엇인가?
2. 과거 발언 검토: 내가 이미 한 발언과 중복되지 않는 새로운 논점은?
3. 상대 모순점 파악: 상대의 과거 발언과 현재 발언 사이의 모순이나 허점은?
4. 근거 제시: 우리가 제시할 수 있는 구체적 데이터나 사례는?
5. 장기적 부작용: 진보 정책이 경제와 재정에 미칠 장기 영향은?
6. 대안 제시: 시장 원리 기반의 실현 가능한 해법은?
</thinking>

중요한 제약사항:
- 내가 과거에 한 발언과 유사한 내용은 반복하지 마라
- 상대의 최근 발언과 과거 발언을 모두 고려하여 정확한 반박을 하라
- 새로운 논점과 구체적 데이터, 근거를 제시하라

그 다음 상대의 최근 주장을 존댓말로 논리적으로 반박하고, 구체적 수치와 경험적 데이터로 재정 부담·장기 부작용을 입증하며, 실증적 근거를 들어 새로운 보수적 해법을 제시하고 존댓말이지만 공격적으로 마무리하라.

형식 제한: <thinking> 부분과 진보 측 주장은 출력하지 말고, 목록·숫자·괄호 시작·하이픈·불릿·이모지·제목을 절대 사용하지 마라. 발화자의 멘트만 출력하라."""
        
        response = self.generate_response(prompt)
        
        # 새로운 발언을 기록에 추가
        if response:
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