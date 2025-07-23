from typing import Dict, List
from .base_agent import BaseAgent

class ModeratorAgent(BaseAgent):
    def __init__(self, model_name: str = 'Bllossom/llama-3.2-Korean-Bllossom-3B'):
        super().__init__(model_name)
        self.system_prompt = """너는 간결하고 중립적인 토론 사회자다. 짧고 명확하게 진행하라."""
        self.factcheck_prompt = """너는 팩트체커다. 발언의 사실성을 간단히 판단하라. O(사실) 또는 X(거짓/과장)로만 답하라."""

    def process_input(self, input_data: Dict) -> str:
        action = input_data.get('action', '')
        topic = input_data.get('topic', '')
        statements = input_data.get('statements', [])
        statement_to_check = input_data.get('statement_to_check', '')
        
        if action == 'introduce':
            return self._introduce_debate(topic)
        elif action == 'conclude':
            return self._conclude_debate(statements)
        elif action == 'factcheck':
            return self._factcheck_statement(statement_to_check)
        else:
            return "사회자 역할을 수행할 수 없습니다."
    
    def _introduce_debate(self, topic: str) -> str:
        prompt = f"""{self.system_prompt}

토론 주제: {topic}

토론을 시작한다. 주제를 간단히 소개하고 진보 vs 보수 토론 시작을 선언하라.
1-2문장으로 간결하게.

사회자:"""
        
        return self.generate_response(prompt, max_length=80)
    
    def _conclude_debate(self, statements: List[Dict]) -> str:
        # 양측 주장 요약
        progressive_count = len([s for s in statements if s.get('stance') == '진보'])
        conservative_count = len([s for s in statements if s.get('stance') == '보수'])
        
        prompt = f"""{self.system_prompt}

토론이 끝났다. 진보 측 {progressive_count}회, 보수 측 {conservative_count}회 발언했다.

토론 종료를 선언하고 양측의 열띤 토론에 감사 인사를 전하라.
1-2문장으로 간결하게.

사회자:"""
        
        return self.generate_response(prompt, max_length=80)
    
    def _factcheck_statement(self, statement: str) -> str:
        """발언에 대한 간단한 팩트체크를 수행합니다."""
        if not statement:
            return "X"
            
        prompt = f"""{self.factcheck_prompt}

발언: "{statement}"

이 발언이 일반적으로 알려진 사실에 부합하는가?
- 명백한 사실이거나 합리적 주장이면: O
- 명백히 틀렸거나 과장이면: X

답변: """
        
        response = self.generate_response(prompt, max_length=10)
        
        # O/X만 추출
        if 'O' in response.upper():
            return 'O'
        elif 'X' in response.upper():
            return 'X'
        else:
            return 'O'  # 기본값 