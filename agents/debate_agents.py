from typing import Dict, List
from .base_agent import BaseAgent

class ProgressiveAgent(BaseAgent):
    def __init__(self, model_name: str = 'Bllossom/llama-3.2-Korean-Bllossom-3B'):
        super().__init__(model_name)
        self.stance = "진보"
        self.system_prompt = """
너는 진보적 정치 성향의 토론자입니다.
- 정부의 적극적 개입과 사회 복지 확대를 지지합니다
- 사회적 평등과 공정성을 중시합니다
- 시장 실패를 보완하는 정부 역할을 강조합니다
- 소외계층 보호와 사회 안전망 강화를 주장합니다
- 논리적이고 설득력 있는 근거를 제시하며 토론합니다
"""

    def process_input(self, input_data: Dict) -> str:
        topic = input_data.get('topic', '')
        opponent_statement = input_data.get('opponent_statement', '')
        
        if opponent_statement:
            prompt = f"""{self.system_prompt}

토론 주제: {topic}

상대방(보수) 의견: {opponent_statement}

위 상대방 의견에 대해 진보적 관점에서 반박하고 당신의 입장을 명확히 제시하세요. 
구체적인 정책이나 사례를 들어 설득력 있게 논증하세요.

진보 입장:"""
        else:
            prompt = f"""{self.system_prompt}

토론 주제: {topic}

위 주제에 대해 진보적 관점에서 당신의 입장을 제시하세요.
정부 개입의 필요성과 사회 복지 확대의 중요성을 강조하며 논거를 제시하세요.

진보 입장:"""
        
        return self.generate_response(prompt)

class ConservativeAgent(BaseAgent):
    def __init__(self, model_name: str = 'Bllossom/llama-3.2-Korean-Bllossom-3B'):
        super().__init__(model_name)
        self.stance = "보수"
        self.system_prompt = """
너는 보수적 정치 성향의 토론자입니다.
- 자유시장 경제와 개인의 책임을 중시합니다
- 정부 개입을 최소화하고 민간 자율성을 선호합니다
- 전통적 가치와 질서 유지를 중요하게 생각합니다
- 경제 효율성과 성장을 우선시합니다
- 논리적이고 설득력 있는 근거를 제시하며 토론합니다
"""

    def process_input(self, input_data: Dict) -> str:
        topic = input_data.get('topic', '')
        opponent_statement = input_data.get('opponent_statement', '')
        
        if opponent_statement:
            prompt = f"""{self.system_prompt}

토론 주제: {topic}

상대방(진보) 의견: {opponent_statement}

위 상대방 의견에 대해 보수적 관점에서 반박하고 당신의 입장을 명확히 제시하세요.
시장 자유와 개인 책임의 중요성을 강조하며 논증하세요.

보수 입장:"""
        else:
            prompt = f"""{self.system_prompt}

토론 주제: {topic}

위 주제에 대해 보수적 관점에서 당신의 입장을 제시하세요.
시장 자유와 개인 책임의 중요성을 강조하며 논거를 제시하세요.

보수 입장:"""
        
        return self.generate_response(prompt) 