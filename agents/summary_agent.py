from typing import Dict, List
from .base_agent import BaseAgent

class SummaryAgent(BaseAgent):
    def __init__(self, model_name: str = 'Bllossom/llama-3.2-Korean-Bllossom-3B'):
        super().__init__(model_name)
        # 실제 방송사나 언론기관의 정책토론회 요약 스타일 반영
        self.system_prompt = """너는 중립적 정책 분석 전문가다. 다음과 같은 특징으로 토론을 요약하라:

분석 스타일:
- 완전한 중립성 유지, 어느 쪽도 편들지 않음
- 정책적 관점에서 객관적 분석
- 양측 주장의 핵심 논리와 근거 정리
- 국민들이 판단할 수 있도록 쟁점 명확화
- 전문적이면서도 이해하기 쉬운 언어 사용

요약 구조:
- 토론의 핵심 쟁점과 대립점 정리
- 각 진영의 주요 논거와 정책 방향
- 양측이 제시한 구체적 데이터나 사례
- 찬반 논리의 강점과 한계
- 국민들이 고려해야 할 주요 판단 기준

중립적 표현:
- "...라고 주장했습니다" "...라는 입장을 밝혔습니다"
- "양측은 서로 다른 관점에서..."
- "이에 대해 찬성 측은... 반대 측은..."
- 판단은 독자에게 맡기는 결론"""

    def summarize_debate(self, topic: str, statements: List[Dict]) -> str:
        """전체 토론을 전문적으로 요약합니다."""
        # 발언 정리
        progressive_statements = [s for s in statements if s.get('stance') == '진보']
        conservative_statements = [s for s in statements if s.get('stance') == '보수']
        
        statements_text = self._format_statements(progressive_statements, conservative_statements)
        
        prompt = f"""{self.system_prompt}

토론 주제: {topic}

토론 발언 내용:
{statements_text}

위 정책토론을 다음 형식으로 전문적이고 중립적으로 요약하라:

## 정책토론 요약

### 📋 핵심 쟁점
이번 토론의 가장 중요한 대립점과 쟁점을 1-2문장으로 정리

### 🔵 진보 진영 주요 주장
- **핵심 논거 1**: [주요 정책 방향과 근거]
- **핵심 논거 2**: [구체적 해결책이나 비판점]
- **제시 데이터**: [언급된 구체적 수치나 사례]

### 🔴 보수 진영 주요 주장  
- **핵심 논거 1**: [주요 정책 방향과 근거]
- **핵심 논거 2**: [구체적 해결책이나 비판점]
- **제시 데이터**: [언급된 구체적 수치나 사례]

### ⚖️ 토론 평가
양측이 제시한 논리의 특징과 국민들이 고려해야 할 판단 기준을 2-3문장으로 중립적 정리

실제 언론사 정책 분석처럼 전문적이고 객관적으로 작성하라.

요약:"""
        
        return self.generate_response(prompt, max_length=1000)
    
    def _format_statements(self, progressive_statements: List[Dict], conservative_statements: List[Dict]) -> str:
        """발언을 정리된 형태로 포맷팅합니다."""
        text = "🔵 진보 측 발언:\n"
        for i, stmt in enumerate(progressive_statements, 1):
            content = stmt.get('statement', '')[:200] + "..." if len(stmt.get('statement', '')) > 200 else stmt.get('statement', '')
            text += f"[라운드 {stmt.get('round', i)}] {content}\n\n"
        
        text += "🔴 보수 측 발언:\n"
        for i, stmt in enumerate(conservative_statements, 1):
            content = stmt.get('statement', '')[:200] + "..." if len(stmt.get('statement', '')) > 200 else stmt.get('statement', '')
            text += f"[라운드 {stmt.get('round', i)}] {content}\n\n"
        
        return text

    def process_input(self, input_data: Dict) -> str:
        """기존 인터페이스와의 호환성을 위한 메서드"""
        action = input_data.get('action', '')
        topic = input_data.get('topic', '')
        statements = input_data.get('statements', [])
        
        if action == 'summarize_debate':
            return self.summarize_debate(topic, statements)
        else:
            return "요약 작업을 수행할 수 없습니다." 