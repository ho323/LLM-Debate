from typing import Dict, List
from .base_agent import BaseAgent

class SummaryAgent(BaseAgent):
    def __init__(self, model_name: str = 'Bllossom/llama-3.2-Korean-Bllossom-3B'):
        super().__init__(model_name)
        self.system_prompt = """너는 토론 요약 전문가다. 양측 주장을 간결하고 공정하게 정리하라."""

    def process_input(self, input_data: Dict) -> str:
        action = input_data.get('action', '')
        topic = input_data.get('topic', '')
        statements = input_data.get('statements', [])
        
        if action == 'summarize_debate':
            return self._summarize_debate(topic, statements)
        else:
            return "요약 작업을 수행할 수 없습니다."
    
    def _summarize_debate(self, topic: str, statements: List[Dict]) -> str:
        """전체 토론을 간결하게 요약합니다."""
        # 발언 정리
        progressive_statements = [s for s in statements if s.get('stance') == '진보']
        conservative_statements = [s for s in statements if s.get('stance') == '보수']
        
        statements_text = self._format_statements(progressive_statements, conservative_statements)
        
        prompt = f"""{self.system_prompt}

토론 주제: {topic}

토론 내용:
{statements_text}

위 토론을 다음 형식으로 간결하게 요약하라:

## 토론 요약
### 핵심 쟁점 (1줄)
[가장 중요한 대립점]

### 진보 측 핵심 주장 (2개)
1. [주장 1]
2. [주장 2]

### 보수 측 핵심 주장 (2개)
1. [주장 1]
2. [주장 2]

### 결론
[토론의 핵심을 1-2문장으로]

요약:"""
        
        return self.generate_response(prompt, max_length=300)
    
    def _format_statements(self, progressive_statements: List[Dict], conservative_statements: List[Dict]) -> str:
        """발언을 정리된 형태로 포맷팅합니다."""
        text = "진보 측:\n"
        for i, stmt in enumerate(progressive_statements, 1):
            text += f"{i}. {stmt.get('content', '')}\n"
        
        text += "\n보수 측:\n"
        for i, stmt in enumerate(conservative_statements, 1):
            text += f"{i}. {stmt.get('content', '')}\n"
        
        return text 