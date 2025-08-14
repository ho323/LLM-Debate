from typing import Dict, List
from .base_agent import BaseAgent

class ModeratorAgent(BaseAgent):
    def __init__(self, model_path: str = '/home/ho/Documents/금융ai/models/EXAONE-4.0-32B-Q4_K_M.gguf'):
        super().__init__(model_path)
        self.system_prompt = """너는 중립적 토론 사회자다. 다음을 철저히 지켜라.

        말투/진행:
        - 정중하고 격식 있는 진행, 완전한 중립성
        - 규칙과 시간 관리, 쟁점 정리, 품격 있는 분위기 유지
        """

    def process_input(self, input_data: Dict) -> str:
        action = input_data.get('action', '')
        topic = input_data.get('topic', '')
        statements = input_data.get('statements', [])
        
        if action == 'introduce':
            return self._introduce_debate(topic)
        elif action == 'conclude':
            return self._conclude_debate(statements)
        else:
            return "사회자 역할을 수행할 수 없습니다."
    
    def _introduce_debate(self, topic: str) -> str:
        prompt = f"""너는 중립적 사회자다. 품격 있고 공정하게 방송 토론을 시작하라.
    다음 내용을 자연스러운 '하나의 단락'으로 풀어써라: 정중한 인사, 오늘 토론의 중요성, 주제 제시({topic}), 양측에 대한 공정한 진행과 독려, 시작 선언.
    형식 제한: 줄바꿈 없이 단락 하나로만 작성하고, 목록·숫자·괄호 시작·하이픈·불릿·이모지·제목을 절대 사용하지 마라. 발화자의 멘트만 출력하라."""
        return self.generate_response(prompt)
    
    def _conclude_debate(self, statements: List[Dict]) -> str:
        # 양측 주장 요약
        progressive_count = len([s for s in statements if s.get('stance') == '진보'])
        conservative_count = len([s for s in statements if s.get('stance') == '보수'])
        
        prompt = f"""너는 중립적 토론 사회자다. 다음 특징을 가져라:
- "잘 지켜주셨고요" "수고 많으셨고요" 같은 격려 표현
- 완전한 중립성 유지, 어떤 편도 들지 않음
- 건설적이고 품격있는 토론 분위기 조성
- 목록·숫자·괄호 시작·하이픈·불릿·이모지·제목을 사용하여 정리

토론이 마무리되었다. 진보 측 {progressive_count}회, 보수 측 {conservative_count}회 발언했다.

토론 종료를 선언하고 마무리하라:
1) 양측의 열띤 토론에 대한 감사 인사
2) 토론 과정의 성과와 의미 정리
3) 국민들의 현명한 선택 당부
4) "수고 많으셨습니다" 같은 격려와 마무리 인사

품격있고 중립적으로 토론회를 마무리하라. 발화자의 발언만 출력하라."""
        
        return self.generate_response(prompt) 