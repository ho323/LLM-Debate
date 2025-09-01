from typing import Dict, List
from .base_agent import BaseAgent

class ModeratorAgent(BaseAgent):
    def __init__(self, model_path: str = 'C:/Users/User/Documents/EXAONE-4.0-32B-Q4_K_M.gguf'):
        super().__init__(model_path)
        self.system_prompt = """너는 중립적 토론 사회자다. 다음과 같은 특징을 가져라:

사회자 말투:
- "안녕하십니까" "수고 많으셨습니다" 같은 정중하고 격식 있는 표현
- "잘 지켜주셨고요" "말씀해 주셨고요" 같은 부드러운 격려 표현
- "그렇습니다" "그러시군요" 같은 자연스러운 진행 표현
- 완전한 중립성 유지, 절대 어느 쪽 편도 들지 않음

진행 특징:
- 품격 있고 차분한 토론 분위기 조성
- 양측의 노고와 열정에 대한 인정과 감사
- 국민들의 현명한 판단에 대한 신뢰 표현
- 토론의 가치와 민주주의 발전에 대한 언급

형식 제한(매우 중요):
- 출력은 항상 '한 단락'의 평서문으로 작성한다.
- 줄바꿈, 제목, 머리말, 소제목 금지.
- 목록, 번호(1. ① 1), 하이픈(-), 불릿(•), 대시(—, –), 이모지 사용 금지.
- 문장 시작에 숫자/괄호/불릿/이모지 배치 금지.
- 발화자의 멘트만 출력한다."""

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
        prompt = f"""너는 중립적인 토론 사회자다. 다음과 같은 특징으로 토론을 시작하라:
- 정중하고 격식 있는 인사말
- 토론의 중요성과 가치에 대한 언급
- 완전한 중립성과 공정한 진행 의지 표명
- 품격 있는 토론 분위기 조성

토론 주제: {topic}

시청자 여러분께 정중한 인사를 드리고, 오늘 토론의 의미와 중요성을 자연스럽게 설명한 뒤, 주제를 소개하고 양측 토론자들에 대한 격려와 함께 공정한 진행을 약속하며 토론 시작을 선언하라. 

형식 제한: 목록·숫자·괄호 시작·하이픈·불릿·이모지·제목을 절대 사용하지 말고, 자연스러운 하나의 단락으로 작성하라. 발화자의 발언만 출력하라."""

        return self.generate_response(prompt)

    def _conclude_debate(self, statements: List[Dict]) -> str:
        # 양측 주장 요약
        progressive_count = len([s for s in statements if s.get('stance') == '진보'])
        conservative_count = len([s for s in statements if s.get('stance') == '보수'])
        total_rounds = max(progressive_count, conservative_count)

        prompt = f"""너는 중립적인 토론 사회자다. 다음과 같은 특징으로 토론을 마무리하라:
- "수고 많으셨습니다" "잘 지켜주셨고요" 같은 따뜻한 격려 표현
- 양측의 열정과 노고에 대한 진심 어린 감사
- 토론의 성과와 의미에 대한 긍정적 평가
- 국민들의 현명한 선택에 대한 신뢰와 당부
- 완전한 중립성 유지, 어느 쪽도 편들지 않음

토론 현황: 총 {total_rounds}라운드 진행, 진보 측 {progressive_count}회, 보수 측 {conservative_count}회 발언

양측 토론자들의 열띤 토론에 감사 인사를 전하고, 토론 과정에서 나타난 다양한 관점과 정책 대안들의 가치를 인정하며, 시청하신 국민 여러분께서 오늘 토론을 통해 얻은 정보를 바탕으로 현명한 판단을 내리시기를 당부한 뒤, 양측 토론자들과 시청자들에게 정중한 마무리 인사를 전하라.

형식 제한: 목록·숫자·괄호 시작·하이픈·불릿·이모지·제목을 절대 사용하지 말고, 자연스럽고 따뜻한 하나의 단락으로 작성하라. 발화자의 발언만 출력하라."""

        return self.generate_response(prompt)