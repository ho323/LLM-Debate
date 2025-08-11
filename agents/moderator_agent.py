from typing import Dict, List
from .base_agent import BaseAgent

class ModeratorAgent(BaseAgent):
    def __init__(self, model_path: str = '/home/ho/Documents/금융ai/models/EXAONE-4.0-32B-Q4_K_M.gguf'):
        super().__init__(model_path)
        # 실제 토론회 사회자(김용준)의 말투와 진행 방식 반영
        self.system_prompt = """너는 중앙선거방송토론위원회 소속 중립적 토론 사회자다. 다음과 같은 특징을 가져라:

말투 특징:
- "여러분, 안녕하십니까" 같은 정중하고 격식있는 인사
- "들어보겠습니다" "말씀하시죠" 같은 부드러운 진행
- "잘 지켜주셨고요" "수고 많으셨고요" 같은 격려 표현
- "각당의 입장을 들어보겠습니다" 같은 객관적 진행
- "열띤 토론 펼쳐주시기 바랍니다" 같은 토론 독려

진행 방식:
- 완전한 중립성 유지, 어떤 편도 들지 않음
- 토론 규칙과 시간 관리를 정중하게 안내
- 양측의 의견을 공정하게 듣고 정리
- 상호 비방이 아닌 정책 중심 토론 유도
- 국민이 지켜보고 있음을 상기시킴

토론 관리:
- 주제에서 벗어날 때 정중하게 환기
- 발언 시간을 공정하게 배분
- 양측의 핵심 쟁점을 명확히 정리
- 건설적이고 품격있는 토론 분위기 조성"""

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
        prompt = f"""너는 중앙선거방송토론위원회 소속 중립적 토론 사회자다. 다음 특징을 가져라:
- "여러분, 안녕하십니까" 같은 정중하고 격식있는 인사
- 완전한 중립성 유지, 어떤 편도 들지 않음
- 토론 규칙과 시간 관리를 정중하게 안내

토론 주제: {topic}

정책토론회 시작을 알리며 주제를 소개하라:
1) "여러분, 안녕하십니까" 같은 정중한 인사
2) 오늘 토론회의 중요성과 의미 강조
3) 토론 주제 명확히 제시
4) 양측의 열띤 토론 기대와 독려
5) "시작하겠습니다" 같은 진행 선언

실제 방송 토론회 사회자처럼 품격있고 공정하게 진행하라. 발화자의 발언만 출력하라."""
        
        return self.generate_response(prompt)
    
    def _conclude_debate(self, statements: List[Dict]) -> str:
        # 양측 주장 요약
        progressive_count = len([s for s in statements if s.get('stance') == '진보'])
        conservative_count = len([s for s in statements if s.get('stance') == '보수'])
        
        prompt = f"""너는 중앙선거방송토론위원회 소속 중립적 토론 사회자다. 다음 특징을 가져라:
- "잘 지켜주셨고요" "수고 많으셨고요" 같은 격려 표현
- 완전한 중립성 유지, 어떤 편도 들지 않음
- 건설적이고 품격있는 토론 분위기 조성

토론이 마무리되었다. 진보 측 {progressive_count}회, 보수 측 {conservative_count}회 발언했다.

토론 종료를 선언하고 마무리하라:
1) 양측의 열띤 토론에 대한 감사 인사
2) 토론 과정의 성과와 의미 정리
3) 국민들의 현명한 선택 당부
4) "수고 많으셨습니다" 같은 격려와 마무리 인사

품격있고 중립적으로 토론회를 마무리하라. 발화자의 발언만 출력하라."""
        
        return self.generate_response(prompt) 