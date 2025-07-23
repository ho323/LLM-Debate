from typing import Dict, List
from .base_agent import BaseAgent

class SummaryAgent(BaseAgent):
    def __init__(self, model_name: str = 'Bllossom/llama-3.2-Korean-Bllossom-3B'):
        super().__init__(model_name)
        self.system_prompt = """
너는 토론 내용을 객관적으로 요약하는 전문가입니다.
- 양측의 핵심 주장을 공정하게 정리합니다
- 주요 논점과 근거를 명확히 구분해서 제시합니다
- 감정적 표현은 배제하고 사실에 기반해 요약합니다
- 가능한 경우 중립적 타협안을 제안합니다
"""

    def process_input(self, input_data: Dict) -> str:
        action = input_data.get('action', '')
        topic = input_data.get('topic', '')
        statements = input_data.get('statements', [])
        factcheck_results = input_data.get('factcheck_results', [])
        
        if action == 'summarize_debate':
            return self._summarize_debate(topic, statements, factcheck_results)
        elif action == 'extract_key_points':
            return self._extract_key_points(statements)
        elif action == 'suggest_compromise':
            return self._suggest_compromise(topic, statements)
        else:
            return "요약 작업을 수행할 수 없습니다."
    
    def _summarize_debate(self, topic: str, statements: List[Dict], factcheck_results: List[Dict]) -> str:
        """전체 토론을 요약합니다."""
        # 발언 정리
        progressive_statements = [s for s in statements if s.get('stance') == '진보']
        conservative_statements = [s for s in statements if s.get('stance') == '보수']
        
        statements_text = self._format_statements(progressive_statements, conservative_statements)
        factcheck_text = self._format_factcheck_results(factcheck_results)
        
        prompt = f"""{self.system_prompt}

토론 주제: {topic}

토론 내용:
{statements_text}

팩트체크 결과:
{factcheck_text}

위 토론 내용을 다음 형식으로 요약해주세요:

## 토론 요약
### 핵심 논점 (3줄 요약)
1. [첫 번째 핵심 논점]
2. [두 번째 핵심 논점]  
3. [세 번째 핵심 논점]

### 진보 측 주요 논거 (3개)
1. [논거 1]
2. [논거 2]
3. [논거 3]

### 보수 측 주요 논거 (3개)
1. [논거 1]
2. [논거 2]
3. [논거 3]

### 타협 가능한 중립안
[양측이 수용할 수 있는 절충 방안]

### 마무리 멘트
[토론에 대한 종합적 평가]

요약:"""
        
        return self.generate_response(prompt, max_length=1024)
    
    def _extract_key_points(self, statements: List[Dict]) -> str:
        """핵심 논점을 추출합니다."""
        statements_text = ""
        for stmt in statements:
            statements_text += f"{stmt.get('stance', '')}: {stmt.get('content', '')}\n\n"
        
        prompt = f"""{self.system_prompt}

토론 발언들:
{statements_text}

위 발언들에서 핵심 논점 5개를 추출해주세요:

핵심 논점:"""
        
        return self.generate_response(prompt)
    
    def _suggest_compromise(self, topic: str, statements: List[Dict]) -> str:
        """타협안을 제안합니다."""
        statements_text = ""
        for stmt in statements:
            statements_text += f"{stmt.get('stance', '')}: {stmt.get('content', '')}\n\n"
        
        prompt = f"""{self.system_prompt}

토론 주제: {topic}

양측 주장:
{statements_text}

위 주제에 대해 양측이 모두 수용할 수 있는 현실적인 타협안을 제안해주세요:

타협안:"""
        
        return self.generate_response(prompt)
    
    def _format_statements(self, progressive_statements: List[Dict], conservative_statements: List[Dict]) -> str:
        """발언을 정리된 형태로 포맷팅합니다."""
        text = "진보 측 주장:\n"
        for i, stmt in enumerate(progressive_statements, 1):
            text += f"{i}. {stmt.get('content', '')}\n"
        
        text += "\n보수 측 주장:\n"
        for i, stmt in enumerate(conservative_statements, 1):
            text += f"{i}. {stmt.get('content', '')}\n"
        
        return text
    
    def _format_factcheck_results(self, factcheck_results: List[Dict]) -> str:
        """팩트체크 결과를 포맷팅합니다."""
        if not factcheck_results:
            return "팩트체크 결과 없음"
        
        text = ""
        for i, result in enumerate(factcheck_results, 1):
            status = result.get('verification_status', 'unclear')
            status_icon = {'verified': '✅', 'false': '❌', 'partial': '⚠️', 'unclear': '❓'}.get(status, '❓')
            text += f"{i}. {status_icon} {result.get('original_statement', '')[:50]}...\n"
        
        return text 