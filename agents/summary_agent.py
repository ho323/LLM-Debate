from typing import Dict, List
from .base_agent import BaseAgent

class SummaryAgent(BaseAgent):
    def __init__(self, model_path: str = 'C:/Users/User/Documents/EXAONE-4.0-32B-Q4_K_M.gguf'):
        super().__init__(model_path)
        
        # 실제 정치 전문지나 정책연구소의 토론 분석 스타일 반영
        self.system_prompt = """너는 정책 분석 전문가로서 정치토론을 객관적으로 분석하는 역할을 한다. 다음과 같은 특징을 가져라:

전문성과 중립성:
- 정치적 편향 없이 양측 논리를 균형 있게 분석
- 정책의 실현가능성과 효과성을 중심으로 평가
- 감정적 표현보다는 논리적 근거와 데이터에 집중
- 국민들이 합리적 판단을 내릴 수 있도록 쟁점 명확화

분석 관점:
- 각 진영의 정책철학과 접근방식의 차이점 부각
- 제시된 정책의 장단점을 객관적으로 검토
- 구체적 근거와 데이터의 신뢰성 평가
- 정책 실현을 위한 현실적 고려사항 제시
- 장기적 파급효과와 부작용 가능성 분석

언어 스타일:
- 전문적이면서도 일반 국민이 이해하기 쉬운 표현
- "주장했다", "제시했다", "강조했다" 등 중립적 서술
- 양측의 입장을 동등하게 다루는 균형감
- 결론에서는 판단을 독자에게 맡기는 열린 접근

형식 제한:
- 자연스러운 문체로 작성하되 구조화된 정보 제공
- 과도한 기호나 이모지 사용 자제
- 읽기 쉬운 단락 구성과 명확한 제목 활용"""

    def summarize_debate(self, topic: str, statements: List[Dict]) -> str:
        """토론을 간단히 요약합니다."""
        return self.generate_brief_summary(topic, statements)
    
    def _analyze_debate_flow(self, statements: List[Dict]) -> str:
        """토론의 전체적인 흐름을 간단히 분석합니다."""
        if not statements:
            return "토론 발언이 없습니다."
        
        total_rounds = len(statements) // 2 if len(statements) % 2 == 0 else (len(statements) + 1) // 2
        return f"총 {total_rounds}라운드의 토론이 진행되었습니다."

    def _extract_key_arguments(self, progressive_statements: List[Dict], conservative_statements: List[Dict]) -> str:
        """양 진영의 발언 수를 간단히 정리합니다."""
        prog_count = len(progressive_statements)
        cons_count = len(conservative_statements)
        return f"진보 측 {prog_count}회, 보수 측 {cons_count}회 발언"

    def generate_brief_summary(self, topic: str, statements: List[Dict]) -> str:
        """간단한 토론 요약을 생성합니다."""
        progressive_count = len([s for s in statements if s.get('stance') == '진보'])
        conservative_count = len([s for s in statements if s.get('stance') == '보수'])
        
        prompt = f"""다음 정치토론을 3-4문장으로 간단히 요약하라:

주제: {topic}
진보 측 발언 수: {progressive_count}회
보수 측 발언 수: {conservative_count}회

최근 주요 발언들:
{self._get_recent_statements(statements, 2)}

토론의 핵심 쟁점과 양측의 기본 입장을 간결하게 정리하되, 어느 쪽으로도 치우치지 않는 중립적 톤으로 작성하라."""
        
        return self.generate_response(prompt)

    def _get_recent_statements(self, statements: List[Dict], count: int) -> str:
        """최근 발언들을 가져옵니다."""
        recent = statements[-count*2:] if len(statements) >= count*2 else statements
        result = ""
        for stmt in recent:
            stance = stmt.get('stance', '')
            content = stmt.get('statement', '')[:100]
            result += f"{stance}: {content}...\n"
        return result

    def analyze_debate_quality(self, topic: str, statements: List[Dict]) -> str:
        """토론의 질적 수준을 평가합니다."""
        prompt = f"""다음 정치토론의 질적 수준을 전문가 관점에서 평가하라:

주제: {topic}
총 발언 수: {len(statements)}회

평가 기준:
1. 논리적 일관성과 근거의 구체성
2. 상대방 주장에 대한 직접적 반박의 적절성  
3. 정책 대안의 현실성과 구체성
4. 데이터와 사례 활용의 적절성
5. 토론 매너와 품격

각 기준별로 간단히 평가하고, 전반적인 토론의 수준과 아쉬운 점, 잘된 점을 객관적으로 분석하라."""
        
        return self.generate_response(prompt)

    def process_input(self, input_data: Dict) -> str:
        """간단 요약과 품질 평가를 처리하는 통합 메서드"""
        action = input_data.get('action', '')
        topic = input_data.get('topic', '')
        statements = input_data.get('statements', [])
        
        if action == 'summarize_debate' or action == 'brief_summary':
            return self.generate_brief_summary(topic, statements)
        elif action == 'analyze_quality':
            return self.analyze_debate_quality(topic, statements)
        else:
            return "지원하지 않는 요약 작업입니다. 'summarize_debate' 또는 'analyze_quality' 중 하나를 선택해주세요."