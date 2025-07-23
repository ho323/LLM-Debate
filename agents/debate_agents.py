from typing import Dict, List
from .base_agent import BaseAgent

class ProgressiveAgent(BaseAgent):
    def __init__(self, model_name: str = 'Bllossom/llama-3.2-Korean-Bllossom-3B'):
        super().__init__(model_name)
        self.stance = "진보"
        self.system_prompt = """너는 진보 정치인이다. 상대방 주장을 구체적으로 반박하며 정부 개입과 사회복지를 강력히 지지한다. 새로운 각도에서 논증하고 반복을 피하라."""

    def process_input(self, input_data: Dict) -> str:
        topic = input_data.get('topic', '')
        opponent_statement = input_data.get('opponent_statement', '')
        
        if opponent_statement:
            # 상대방 발언이 있을 때 - 반박 모드
            prompt = f"""{self.system_prompt}

토론 주제: {topic}

보수 측 주장: "{opponent_statement}"

위 보수 주장을 구체적으로 반박하라. 다음 중 하나의 방식으로 접근하라:
1) 경제적 불평등 확대 문제 지적
2) 사회적 약자 보호 필요성 강조  
3) 정부 역할의 현실적 필요성 제시
4) 시장 실패 사례를 통한 반박

2-3문장으로 간결하되 임팩트 있게 반박하라.

진보 반박:"""
        else:
            # 첫 발언 - 입장 제시 모드
            prompt = f"""{self.system_prompt}

토론 주제: {topic}

이 주제에 대한 진보적 입장을 강력히 제시하라. 다음 중 하나에 집중하라:
1) 정부 개입의 필요성과 정당성
2) 사회복지 확대의 경제적 효과
3) 평등과 공정성의 중요성

2-3문장으로 간결하게 핵심만 답변하라.

진보 주장:"""
        
        return self.generate_response(prompt, max_length=200, target_length="간결")

class ConservativeAgent(BaseAgent):
    def __init__(self, model_name: str = 'Bllossom/llama-3.2-Korean-Bllossom-3B'):
        super().__init__(model_name)
        self.stance = "보수"
        self.system_prompt = """너는 보수 정치인이다. 상대방 주장을 구체적으로 반박하며 시장 자유와 개인 책임을 강력히 지지한다. 새로운 각도에서 논증하고 반복을 피하라."""

    def process_input(self, input_data: Dict) -> str:
        topic = input_data.get('topic', '')
        opponent_statement = input_data.get('opponent_statement', '')
        
        if opponent_statement:
            # 상대방 발언이 있을 때 - 반박 모드
            prompt = f"""{self.system_prompt}

토론 주제: {topic}

진보 측 주장: "{opponent_statement}"

위 진보 주장을 구체적으로 반박하라. 다음 중 하나의 방식으로 접근하라:
1) 경제 효율성과 성장 저해 문제 지적
2) 개인 자유와 선택권 침해 우려 제기
3) 시장 메커니즘의 우수성 강조
4) 정부 개입의 부작용 사례 제시

2-3문장으로 간결하되 임팩트 있게 반박하라.

보수 반박:"""
        else:
            # 첫 발언 - 입장 제시 모드
            prompt = f"""{self.system_prompt}

토론 주제: {topic}

이 주제에 대한 보수적 입장을 강력히 제시하라. 다음 중 하나에 집중하라:
1) 시장 자유의 중요성과 효율성
2) 개인 책임과 자유 선택의 가치
3) 정부 개입의 부작용과 한계

2-3문장으로 간결하게 핵심만 답변하라.

보수 주장:"""
        
        return self.generate_response(prompt, max_length=200, target_length="간결") 