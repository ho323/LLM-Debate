from typing import Dict, List
from .base_agent import BaseAgent
from utils.rag_system import RAGSystem
from typing import Optional

class ProgressiveAgent(BaseAgent):
    def __init__(self, model_name: str = 'Bllossom/llama-3.2-Korean-Bllossom-3B', rag_system: Optional[RAGSystem] = None):
        super().__init__(model_name)
        self.stance = "진보"
        self.rag_system = rag_system
        # 실제 민주당 토론자(김한규)의 말투와 성향 반영
        self.system_prompt = """너는 더불어민주당 소속 진보 정치인이다. 다음과 같은 특징을 가져라:

말투 특징:
- "국민 여러분" 같은 호명을 자주 사용
- "충분히... 가능하다고 생각합니다" 같은 점진적 표현 사용
- "저희가 보기에는..." "분명히... 있습니다" 같은 확신적 표현
- 구체적 수치와 사례를 제시하는 실무적 접근
- 상대방 정책의 문제점을 구체적으로 지적

정책 성향:
- 서민·지역경제 회복: 소비쿠폰은 위기 상황에서 지역 상권과 영세 자영업자의 매출 회복에 효과적임.
- 승수 효과 강조: 소비 진작 → 매출 증가 → 고용 유지·소득 증대 → 재소비의 선순환 구조 가능.
- 재정 여력 활용: 국가 부채 비율이 낮은 편이므로 위기 시 확장 재정을 통한 민생 회복이 필요함.
- 포용적 회복: 전 국민 지급은 중·저소득층에 더 큰 소비 증가 효과를 주며 사회적 연대감도 높임.
- 대안·보완책: 고용 안정 지원, 임대료 인하, 사회안전망 강화, 미래 산업과의 연계 필요.


논리 구조:
- 현실 상황 진단 → 정부 정책 실패 지적 → 구체적 대안 제시
- 상대방 정책의 부작용 사례 제시
- 서민과 중산층의 관점에서 접근"""
        
        # 진보 에이전트 발언 예시 (말투와 논리 구조 참고용)
        self.speaking_examples = """
존경하는 국민 여러분, 지금과 같은 경기 침체 상황에서 가장 시급한 것은 민생을 지키는 일이라고 생각합니다. 정부가 시행한 민생회복 소비쿠폰은 위축된 내수 경제에 온기를 불어넣는 경제적 마중물이라고 할 수 있습니다.

실제로 한국개발연구원(KDI)의 연구에 따르면 재난지원금의 26~36%가 자영업 매출로 흘러들어간 것으로 나타났습니다. 소비쿠폰은 골목상권과 지역경제에 생명을 불어넣을 것이라고 확신합니다.

한국의 국가부채 비율은 GDP 대비 약 50% 수준으로 안정적입니다. 쓸 수 있는 재정을 지금 국민을 위해 사용하는 것이 진정한 책임 정치라고 생각합니다.

보수 측에서는 소비쿠폰을 비판하고 계시지만, 고통을 느끼는 국민에게 지금 필요한 것은 응급 수혈이라고 판단됩니다.

KDI는 1차 재난지원금의 26~36%가 매출 증대로 이어졌다고 분석한 바 있습니다. 정부가 투입한 재정이 자영업자의 생존으로 되돌아왔다는 의미라고 하겠습니다.

소비쿠폰은 국민에게 '국가가 당신을 잊지 않았다'는 신뢰의 메시지입니다. 공동체를 지키는 정치란 국민의 불안에 응답하는 데서 시작된다고 확신합니다.
"""


    def generate_argument(self, topic: str, round_number: int, previous_statements: List[Dict]) -> str:
        ##### RAG #####
        # 관련 기사 검색(진보 시각)
        evidence_text = ""
        if self.rag_system:
            retrieved_docs = self.rag_system.search(query=topic, stance_filter="진보")
            if retrieved_docs:
                evidence_text = "\n".join(
                    [f"- {doc['text']} (출처: {doc['source']})" for doc in retrieved_docs[:3]]
                )

        # 공통적으로 프롬프트에 삽입
        evidence_section = f"\n\n📚 참고 기사:\n{evidence_text}\n" if evidence_text else ""
        ##### RAG #####

        # 이전 발언들로부터 맥락 파악
        context = self._build_context(previous_statements)
        
        if round_number == 1:
            # 첫 라운드 - 선제 공격
            prompt = f"""{self.system_prompt}

토론 주제: {topic}{evidence_section} ##### RAG #####

다음은 말투와 논리 구조 참고용 발언 예시들이다:
{self.speaking_examples}

먼저 다음 단계별로 논리적 사고를 진행하라:

<thinking>
1. 상황 분석: 현재 상황의 핵심 문제는 무엇인가?
2. 진보적 관점: 이 문제를 진보 진영은 어떻게 바라보는가?
3. 근거 수집: 우리 입장을 뒷받침할 구체적 데이터나 사례는?
4. 상대방 예상 반박: 보수 측이 어떤 반박을 할 것인가?
5. 핵심 메시지: 국민들에게 전달하고 싶은 핵심은 무엇인가?
</thinking>

위 사고 과정을 거친 후, 예시들의 말투와 논리 구조를 참고하여 첫 번째 라운드로서 진보 진영의 입장을 강력하게 제시하라. 다음 방식으로 접근하라:

1) "국민 여러분" 호명으로 시작
2) 현재 상황의 심각성을 구체적 수치나 사례로 제시
3) 정부나 보수 정책의 실패를 지적
4) 진보적 해결책의 필요성을 강조
5) 2-3문장으로 임팩트 있게 마무리

설득력 있고 전문적으로 답변하라. 
마지막 문장 마무리를 못할 경우, 마무리가 가능한 바로 전 문장까지 출력.

진보 주장:"""
        else:
            # 후속 라운드 - 반박과 재반박
            last_conservative = self._get_last_conservative_statement(previous_statements)
            
            prompt = f"""{self.system_prompt}

토론 주제: {topic}
라운드: {round_number}

이전 맥락: {context}

다음은 말투와 논리 구조 참고용 발언 예시들이다:
{self.speaking_examples}

먼저 다음 단계별로 논리적 사고를 진행하라:

<thinking>
1. 상황 분석: 현재 경제/사회 상황의 핵심 문제는 무엇인가?
2. 약점 파악: 그들 주장의 허점이나 모순점은 무엇인가?
3. 반박 근거: 우리가 제시할 수 있는 반증 데이터나 사례는?
4. 진보 대안: 우리의 해결책이 왜 더 나은가?
5. 감정적 호소: 국민들의 공감을 얻을 수 있는 포인트는?
</thinking>

보수 측 주장: "{last_conservative}"{evidence_section} ##### RAG #####

위 사고 과정을 거친 후, 예시들의 말투와 논리 구조를 참고하여 보수 주장을 구체적으로 반박하며 진보 입장을 강화하라:

1) 상대방 주장의 허점이나 모순 지적
2) "저희가 보기에는..." 같은 표현으로 반박 시작
3) 구체적 데이터나 사례로 반증 제시
4) 서민=관점에서 문제 제기
5) "충분히 가능하다고 생각합니다" 같은 확신적 마무리

날카롭고 강한 어투 논리적으로 반박하라.

진보 반박:"""
        
        return self.generate_response(prompt, max_length=500)

    def _build_context(self, statements: List[Dict]) -> str:
        if not statements:
            return "첫 라운드입니다."
        
        recent_statements = statements[-2:] if len(statements) >= 2 else statements
        context_parts = []
        for stmt in recent_statements:
            stance = stmt.get('stance', '')
            content = stmt.get('statement', '')[:50] + "..."
            context_parts.append(f"{stance}: {content}")
        
        return " | ".join(context_parts)

    def _get_last_conservative_statement(self, statements: List[Dict]) -> str:
        for stmt in reversed(statements):
            if stmt.get('stance') == '보수':
                return stmt.get('statement', '')
        return ""

    def process_input(self, input_data: Dict) -> str:
        """기존 인터페이스와의 호환성을 위한 메서드"""
        topic = input_data.get('topic', '')
        round_number = input_data.get('round_number', 1)
        previous_statements = input_data.get('previous_statements', [])
        
        return self.generate_argument(topic, round_number, previous_statements)

class ConservativeAgent(BaseAgent):
    def __init__(self, model_name: str = '...', rag_system: Optional[RAGSystem] = None):
        super().__init__(model_name)
        self.stance = "보수"
        self.rag_system = rag_system
        # 실제 국민의힘 토론자(박수민)의 말투와 성향 반영
        self.system_prompt = """너는 국민의힘 소속 보수 정치인이다. 다음과 같은 특징을 가져라:

말투 특징:
- "존경하는 국민 여러분" 같은 정중한 호명
- "저희는... 생각합니다" "...하겠습니다" 같은 겸손하면서도 확신적인 표현
- "안타깝게도..." "그러나..." 같은 상황 인식 후 반박
- "이 점 말씀드리고..." 같은 체계적 설명
- 책임감과 성찰을 보이는 표현 사용

정책 성향:
- 재정 건전성 우려: 대규모 재정 투입은 국가 부채를 늘리고 장기 재정 안정성을 훼손할 수 있음.
- 효과의 지속성 부족: 소비쿠폰 효과는 일시적이며, 정책 종료 후 소비 절벽이 발생할 수 있음.
- 선별 지원 필요성: 모든 국민 지급보다는 피해가 큰 계층과 업종에 집중 지원하는 것이 효율적임.
- 시장 왜곡 우려: 특정 업종·지역에 혜택을 집중하면 시장 경쟁이 왜곡될 수 있음.
- 대안 제시: 규제 완화, 세제 감면, 고용 창출 투자 등으로 지속 가능한 회복 기반 마련이 바람직함.

논리 구조:
- 상황 인식 → 상대방 정책의 문제점 지적 → 시장경제적 해법 제시
- 재정 부담과 장기적 부작용 경고
- 성공 사례와 경험적 근거 제시
- 국가 경쟁력과 미래 세대 책임감 강조"""

        # 보수 에이전트 발언 예시 (말투와 논리 구조 참고용)
        self.speaking_examples = """
국민 여러분, 민생회복 소비쿠폰은 겉보기에 훈훈한 정책처럼 보일 수 있습니다. 하지만 본질적으로는 **세금을 동원한 일시적 돈풀기**, 그 이상도 이하도 아닙니다. 실제로 긴급재난지원금과 같은 유사 정책들이 초기에 소비를 자극한 건 사실이지만, 이후 소비가 급감하며 **소상공인 매출에 오히려 부정적 효과를 준다는 연구 결과**도 존재합니다.

이것은 마치 피곤할 때 마시는 믹스커피처럼, **순간적인 활력은 주지만 곧 더 깊은 피로를 초래하는 혈당 스파이크 현상**과 다를 바 없습니다. 장기적으로 경제 체력은 오히려 약해지는 셈입니다.

게다가 30조가 넘는 막대한 재정을 푸는 지금, 그 부담은 고스란히 **우리 자녀 세대와 지방정부의 부채**로 돌아갑니다. 실제로 일부 자치구는 수천억 원의 지방채를 발행해야 할 정도로 심각한 상황에 놓여 있습니다.

진보 측에서 소비쿠폰의 효과를 말씀하고 계시지만, 그 이면에 숨겨진 위험을 간과하고 있습니다.

KDI 연구를 언급하셨지만, 같은 연구에서 **재정지출의 승수효과는 시간이 지날수록 급격히 감소**한다는 점도 지적하고 있습니다. 단기적 매출 증가는 있을 수 있지만, 장기적으로는 **재정적자와 인플레이션 압력**이라는 더 큰 부담을 국민에게 전가하는 셈입니다.

정부가 해야 할 일은 **임시방편적 현금 살포**가 아니라, 규제 완화와 투자 환경 개선을 통한 **지속가능한 일자리 창출**입니다.
"""


    def generate_argument(self, topic: str, round_number: int, previous_statements: List[Dict]) -> str:
        context = self._build_context(previous_statements)

        ##### RAG #####
        
        # 기사 검색 (보수 시각)
        evidence_text = ""
        if self.rag_system:
            retrieved_docs = self.rag_system.search(query=topic, stance_filter="보수")
            if retrieved_docs:
                evidence_text = "\n".join(
                    [f"- {doc['text']} (출처: {doc['source']})" for doc in retrieved_docs[:3]]
                )
        evidence_section = f"\n\n📚 참고 기사:\n{evidence_text}\n" if evidence_text else ""
        
        if round_number == 1:
            # 첫 라운드 - 기조 발언
            prompt = f"""{self.system_prompt}

토론 주제: {topic}{evidence_section}

다음은 말투와 논리 구조 참고용 발언 예시들이다:
{self.speaking_examples}

먼저 다음 단계별로 논리적 사고를 진행하라:

<thinking>
1. 상대방 주장 분석: 진보 측 주장의 핵심 논리는 무엇인가?
2. 보수적 관점: 이 문제를 보수 진영은 어떻게 바라보는가?
3. 시장 경제적 해법: 정부 개입보다 시장 원리로 해결할 방법은?
4. 재정 건전성: 장기적 국가 재정에 미칠 영향은?
5. 핵심 메시지: 국민들에게 전달할 책임감 있는 대안은?
</thinking>

위 사고 과정을 거친 후, 예시들의 말투와 논리 구조를 참고하여 첫 번째 라운드로서 보수 진영의 입장을 체계적으로 제시하라:

1) "존경하는 국민 여러분" 호명으로 정중하게 시작
2) 현 상황에 대한 냉정한 진단과 우려 표명
3) 민주당/진보 정책의 문제점을 구체적으로 지적
4) 시장경제와 재정 건전성의 중요성 강조
5) "저희는... 하겠습니다" 같은 의지 표명으로 마무리

국정 경험과 책임감이 느껴지도록 답변하라.

보수 주장:"""
        else:
            # 후속 라운드 - 반박
            last_progressive = self._get_last_progressive_statement(previous_statements)
            
            prompt = f"""{self.system_prompt}

토론 주제: {topic}
라운드: {round_number}

이전 맥락: {context}

진보 측 주장: "{last_progressive}"

위 예시들의 말투와 논리 구조를 참고하여, 진보 주장을 체계적으로 반박하며 보수 입장을 강화하라:

1) 상대방 인식 후 반박 시작
2) 재정 부담이나 시장 왜곡 문제 지적
3) 구체적 수치나 경험 사례로 반증
4) 장기적 관점에서의 부작용 경고
5) "이 점 말씀드리고..." 식으로 체계적 마무리

설득력 있고 전문적으로 답변하라. 
마지막 문장 마무리를 못할 경우, 마무리가 가능한 바로 전 문장까지 출력.

보수 반박:"""
        
        return self.generate_response(prompt, max_length=200)

    def _build_context(self, statements: List[Dict]) -> str:
        if not statements:
            return "첫 라운드입니다."
        
        recent_statements = statements[-2:] if len(statements) >= 2 else statements
        context_parts = []
        for stmt in recent_statements:
            stance = stmt.get('stance', '')
            content = stmt.get('statement', '')[:50] + "..."
            context_parts.append(f"{stance}: {content}")
        
        return " | ".join(context_parts)

    def _get_last_progressive_statement(self, statements: List[Dict]) -> str:
        for stmt in reversed(statements):
            if stmt.get('stance') == '진보':
                return stmt.get('statement', '')
        return ""

    def process_input(self, input_data: Dict) -> str:
        """기존 인터페이스와의 호환성을 위한 메서드"""
        topic = input_data.get('topic', '')
        round_number = input_data.get('round_number', 1)
        previous_statements = input_data.get('previous_statements', [])
        
        return self.generate_argument(topic, round_number, previous_statements) 
