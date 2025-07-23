from typing import Dict, List
from .base_agent import BaseAgent

class ModeratorAgent(BaseAgent):
    def __init__(self, model_name: str = 'Bllossom/llama-3.2-Korean-Bllossom-3B'):
        super().__init__(model_name)
        self.system_prompt = """
너는 공정하고 중립적인 토론 사회자입니다.
- 양측의 의견을 객관적으로 정리합니다
- 토론의 흐름을 관리하고 다음 논점을 제시합니다
- 감정적인 표현은 피하고 사실에 기반해 진행합니다
- 양측 모두에게 발언 기회를 공평하게 제공합니다
"""

    def process_input(self, input_data: Dict) -> str:
        action = input_data.get('action', '')
        topic = input_data.get('topic', '')
        statements = input_data.get('statements', [])
        
        if action == 'introduce':
            return self._introduce_debate(topic)
        elif action == 'moderate':
            return self._moderate_discussion(topic, statements)
        elif action == 'conclude':
            return self._conclude_round(statements)
        else:
            return "사회자 역할을 수행할 수 없습니다."
    
    def _introduce_debate(self, topic: str) -> str:
        prompt = f"""{self.system_prompt}

토론 주제: {topic}

위 주제에 대한 토론을 시작합니다. 주제를 소개하고 토론의 진행 방식을 안내해주세요.
양측이 공정하게 의견을 개진할 수 있도록 중립적으로 진행하겠습니다.

사회자:"""
        
        return self.generate_response(prompt)
    
    def _moderate_discussion(self, topic: str, statements: List[Dict]) -> str:
        statements_text = ""
        for i, stmt in enumerate(statements[-2:]):  # 최근 2개 발언만
            statements_text += f"{stmt['stance']}: {stmt['content']}\n\n"
        
        prompt = f"""{self.system_prompt}

토론 주제: {topic}

최근 발언들:
{statements_text}

위 발언들을 바탕으로 토론의 흐름을 정리하고, 다음 논점이나 질문을 제시해주세요.
양측의 핵심 주장을 간단히 요약하고 토론을 이어나가도록 유도해주세요.

사회자:"""
        
        return self.generate_response(prompt)
    
    def _conclude_round(self, statements: List[Dict]) -> str:
        statements_text = ""
        for stmt in statements:
            statements_text += f"{stmt['stance']}: {stmt['content']}\n\n"
        
        prompt = f"""{self.system_prompt}

이번 라운드의 주요 발언들:
{statements_text}

이번 토론 라운드를 마무리하며 양측의 핵심 주장을 간단히 정리해주세요.
다음 토론 주제나 마무리 멘트를 제시해주세요.

사회자:"""
        
        return self.generate_response(prompt) 