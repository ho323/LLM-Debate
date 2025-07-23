from typing import Dict, List
from .base_agent import BaseAgent
from utils.rag_system import RAGSystem

class FactCheckAgent(BaseAgent):
    def __init__(self, model_name: str = 'Bllossom/llama-3.2-Korean-Bllossom-3B'):
        super().__init__(model_name)
        self.rag_system = RAGSystem()
        self.system_prompt = """
너는 객관적이고 신뢰할 수 있는 팩트체커입니다.
- 주장의 사실 여부를 객관적으로 판단합니다
- 신뢰할 수 있는 출처와 데이터를 기반으로 검증합니다
- 확인된 사실은 ✅, 거짓이거나 오해의 소지가 있는 내용은 ❌로 표시합니다
- 불분명하거나 부분적으로 맞는 경우 ⚠️로 표시합니다
- 검증 결과와 함께 관련 출처를 제공합니다
"""

    def process_input(self, input_data: Dict) -> str:
        statement = input_data.get('statement', '')
        stance = input_data.get('stance', '')
        
        # RAG 시스템에서 관련 정보 검색
        relevant_info = self.rag_system.get_factcheck_context(statement)
        
        context_text = ""
        if relevant_info:
            context_text = "\n관련 정보:\n"
            for i, info in enumerate(relevant_info):
                context_text += f"{i+1}. {info['content']} (출처: {info['source']})\n"
        
        prompt = f"""{self.system_prompt}

검증할 발언: {statement}
발언자 성향: {stance}

{context_text}

위 발언의 사실 여부를 검증해주세요. 다음 형식으로 답변해주세요:

검증 결과: [✅/❌/⚠️]
분석: [구체적인 검증 내용]
관련 출처: [참고한 정보의 출처]

팩트체크:"""
        
        return self.generate_response(prompt)
    
    def check_multiple_statements(self, statements: List[Dict]) -> List[Dict]:
        """여러 발언을 일괄 팩트체크합니다."""
        results = []
        
        for statement_data in statements:
            result = self.process_input(statement_data)
            
            # 결과 파싱
            verification_result = {
                'original_statement': statement_data.get('statement', ''),
                'stance': statement_data.get('stance', ''),
                'factcheck_result': result,
                'verification_status': self._extract_status(result)
            }
            results.append(verification_result)
        
        return results
    
    def _extract_status(self, factcheck_result: str) -> str:
        """팩트체크 결과에서 상태를 추출합니다."""
        if '✅' in factcheck_result:
            return 'verified'
        elif '❌' in factcheck_result:
            return 'false'
        elif '⚠️' in factcheck_result:
            return 'partial'
        else:
            return 'unclear' 