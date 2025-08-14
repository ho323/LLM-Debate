from typing import Dict, List
from .base_agent import BaseAgent

class ProgressiveAgent(BaseAgent):
    def __init__(self, model_path: str = '/home/ho/Documents/금융ai/models/EXAONE-4.0-32B-Q4_K_M.gguf'):
        super().__init__(model_path)
        self.stance = "진보"
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
- 줄바꿈, 제목, 머리말, 소제목 금지.
- 목록, 번호(1. ① 1), 하이픈(-), 불릿(•), 대시(—, –), 이모지 사용 금지.
- 문장 시작에 숫자/괄호/불릿/이모지 배치 금지.
- 발화자의 멘트만 출력한다.

"""

    def generate_argument(self, topic: str, round_number: int, previous_statements: List[Dict]) -> str:
        context = self._build_context(previous_statements)

        if round_number == 1:
            prompt = f"""너는 더불어민주당 소속 진보 정치인이다.
    토론 주제: {topic}
    정중한 호칭을 포함하되 과장 없이, 존댓말로 구체적 수치·사례로 현재 상황의 심각성을 제시하고, 정부나 보수 정책의 실패를 비판하며, 진보적 대안의 필요성을 분명히 밝힌 뒤 2~3문장으로 힘 있게 마무리하라.
    형식 제한: 줄바꿈 없이 단락 하나로만 작성하고, 목록·숫자·괄호 시작·하이픈·불릿·이모지·제목을 사용하지 마라. 발화자의 멘트만 출력하라."""
        else:
            last_conservative = self._get_last_conservative_statement(previous_statements)
            prompt = f"""너는 더불어민주당 소속 진보 정치인이다.
    토론 주제: {topic}
    보수 측 주장: "{last_conservative}"
    보수 측 주장을 정확히 요지 파악한 뒤, 존댓말로 구체적 데이터와 사례로 반증하고, 서민·중산층 관점에서 대안을 제시하며 공격적으로 마무리하라.
    형식 제한: 보수 측 주장은 출력하지 말고, 목록·숫자·괄호 시작·하이픈·불릿·이모지·제목을 사용하지 마라. 발화자의 멘트만 출력하라."""
        return self.generate_response(prompt)

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
    def __init__(self, model_path: str = '/home/ho/Documents/금융ai/models/EXAONE-4.0-32B-Q4_K_M.gguf'):
        super().__init__(model_path)
        self.stance = "보수"
        # 실제 국민의힘 토론자(박수민)의 말투와 성향 반영
        self.system_prompt = """너는 국민의힘 소속 보수 정치인이다. 다음과 같은 특징을 가져라:

말투 특징:
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
- 국가 경쟁력과 미래 세대 책임감 강조

형식 제한(매우 중요):
- 출력은 항상 '한 단락'의 평서문으로 작성한다.
- 줄바꿈, 제목, 머리말, 소제목 금지.
- 목록, 번호(1. ① 1), 하이픈(-), 불릿(•), 대시(—, –), 이모지 사용 금지.
- 문장 시작에 숫자/괄호/불릿/이모지 배치 금지.
- 발화자의 멘트만 출력한다.

"""

    def generate_argument(self, topic: str, round_number: int, previous_statements: List[Dict]) -> str:
        context = self._build_context(previous_statements)

        if round_number == 1:
            prompt = f"""너는 국민의힘 소속 보수 정치인이다.
    토론 주제: {topic}
    현 상황을 냉정히 진단하고 존댓말로 우려를 밝힌 다음, 진보 정책의 문제점을 구체적으로 지적하고, 시장경제·재정건전성의 중요성을 근거와 함께 강조하며 책임 있는 어조로 마무리하라.
    형식 제한: 줄바꿈 없이 단락 하나로만 작성하고, 목록·숫자·괄호 시작·하이픈·불릿·이모지·제목을 절대 사용하지 마라. 발화자의 멘트만 출력하라."""
        else:
            last_progressive = self._get_last_progressive_statement(previous_statements)
            prompt = f"""너는 국민의힘 소속 보수 정치인이다.
    토론 주제: {topic}
    진보 측 주장: "{last_progressive}"
    상대 주장을 존댓말로 논리적으로 반박하고, 재정 부담·장기 부작용을 짚으며, 구체적 수치나 경험적 근거를 들어 보수적 해법을 제시하고 존댓말이지만 공격적으로 마무리하라.
    형식 제한: 진보 측 주장은 출력하지 말고, 목록·숫자·괄호 시작·하이픈·불릿·이모지·제목을 절대 사용하지 마라. 발화자의 멘트만 출력하라."""
        return self.generate_response(prompt)

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