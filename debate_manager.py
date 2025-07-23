from typing import Dict, List, Optional
from agents import (
    ProgressiveAgent, 
    ConservativeAgent, 
    ModeratorAgent, 
    SummaryAgent
)

class DebateManager:
    def __init__(self, model_name: str = 'Bllossom/llama-3.2-Korean-Bllossom-3B'):
        print("토론 시스템 초기화 중...")
        
        # 에이전트들 초기화 (진보 vs 보수만)
        self.progressive_agent = ProgressiveAgent(model_name)
        self.conservative_agent = ConservativeAgent(model_name)
        self.moderator_agent = ModeratorAgent(model_name)
        self.summary_agent = SummaryAgent(model_name)
        
        # 토론 상태 관리
        self.current_topic = ""
        self.statements = []
        self.factcheck_results = []
        self.round_count = 0
        self.max_rounds = 3
        
        print("토론 시스템 초기화 완료!")
    
    def start_debate(self, topic: str) -> Dict:
        """토론을 시작합니다."""
        self.current_topic = topic
        self.statements = []
        self.factcheck_results = []
        self.round_count = 0
        
        print(f"\n=== 토론 시작: {topic} ===")
        
        # 사회자 소개 (간결하게)
        moderator_intro = self.moderator_agent.process_input({
            'action': 'introduce',
            'topic': topic
        })
        
        print(f"\n🎯 사회자: {moderator_intro}")
        
        return {
            'topic': topic,
            'moderator_intro': moderator_intro,
            'status': 'started'
        }
    
    def conduct_round(self) -> Dict:
        """한 라운드의 토론을 진행합니다 (직전 발언 반박)."""
        if self.round_count >= self.max_rounds:
            return {'status': 'completed', 'message': '토론이 종료되었습니다.'}
        
        self.round_count += 1
        print(f"\n--- 라운드 {self.round_count} ---")
        
        round_results = {
            'round': self.round_count,
            'statements': [],
            'factcheck_results': []
        }
        
        # 1. 진보 에이전트 발언 (직전 보수 발언만 전달)
        last_conservative_statement = self._get_last_statement('보수')
        
        progressive_input = {
            'topic': self.current_topic,
            'opponent_statement': last_conservative_statement  # 직전 보수 발언만
        }
        progressive_statement = self.progressive_agent.process_input(progressive_input)
        
        progressive_data = {
            'stance': '진보',
            'content': progressive_statement,
            'round': self.round_count
        }
        self.statements.append(progressive_data)
        round_results['statements'].append(progressive_data)
        
        print(f"\n🔵 진보: {progressive_statement}")
        
        # 진보 발언 팩트체크
        progressive_factcheck = self.moderator_agent.process_input({
            'action': 'factcheck',
            'statement_to_check': progressive_statement
        })
        
        factcheck_result_prog = {
            'stance': '진보',
            'statement': progressive_statement,
            'result': progressive_factcheck
        }
        self.factcheck_results.append(factcheck_result_prog)
        round_results['factcheck_results'].append(factcheck_result_prog)
        
        print(f"📊 팩트체크: {progressive_factcheck}")
        
        # 2. 보수 에이전트 반박 (직전 진보 발언만 전달)
        conservative_input = {
            'topic': self.current_topic,
            'opponent_statement': progressive_statement  # 방금 진보 발언
        }
        conservative_statement = self.conservative_agent.process_input(conservative_input)
        
        conservative_data = {
            'stance': '보수',
            'content': conservative_statement,
            'round': self.round_count
        }
        self.statements.append(conservative_data)
        round_results['statements'].append(conservative_data)
        
        print(f"\n🔴 보수: {conservative_statement}")
        
        # 보수 발언 팩트체크
        conservative_factcheck = self.moderator_agent.process_input({
            'action': 'factcheck',
            'statement_to_check': conservative_statement
        })
        
        factcheck_result_cons = {
            'stance': '보수',
            'statement': conservative_statement,
            'result': conservative_factcheck
        }
        self.factcheck_results.append(factcheck_result_cons)
        round_results['factcheck_results'].append(factcheck_result_cons)
        
        print(f"📊 팩트체크: {conservative_factcheck}")
        
        return round_results
    
    def _get_last_statement(self, stance: str) -> Optional[str]:
        """지정된 성향의 마지막 발언을 가져옵니다."""
        for statement in reversed(self.statements):
            if statement.get('stance') == stance:
                return statement.get('content', '')
        return None
    
    def conclude_debate(self) -> Dict:
        """토론을 마무리하고 요약을 생성합니다."""
        print(f"\n=== 토론 마무리 ===")
        
        # 사회자 마무리 (간결하게)
        moderator_conclusion = self.moderator_agent.process_input({
            'action': 'conclude',
            'statements': self.statements
        })
        
        print(f"\n🎯 사회자: {moderator_conclusion}")
        
        # 팩트체크 요약
        print(f"\n📊 팩트체크 결과:")
        prog_o_count = len([f for f in self.factcheck_results if f['stance'] == '진보' and f['result'] == 'O'])
        prog_x_count = len([f for f in self.factcheck_results if f['stance'] == '진보' and f['result'] == 'X'])
        cons_o_count = len([f for f in self.factcheck_results if f['stance'] == '보수' and f['result'] == 'O'])
        cons_x_count = len([f for f in self.factcheck_results if f['stance'] == '보수' and f['result'] == 'X'])
        
        print(f"진보: O({prog_o_count}) X({prog_x_count})")
        print(f"보수: O({cons_o_count}) X({cons_x_count})")
        
        # 전체 토론 요약
        summary_input = {
            'action': 'summarize_debate',
            'topic': self.current_topic,
            'statements': self.statements
        }
        debate_summary = self.summary_agent.process_input(summary_input)
        
        print(f"\n📋 토론 요약:\n{debate_summary}")
        
        return {
            'topic': self.current_topic,
            'total_rounds': self.round_count,
            'moderator_conclusion': moderator_conclusion,
            'debate_summary': debate_summary,
            'all_statements': self.statements,
            'factcheck_summary': {
                'progressive': {'O': prog_o_count, 'X': prog_x_count},
                'conservative': {'O': cons_o_count, 'X': cons_x_count}
            },
            'status': 'completed'
        }
    
    def run_full_debate(self, topic: str) -> Dict:
        """전체 토론을 자동으로 실행합니다."""
        # 토론 시작
        start_result = self.start_debate(topic)
        
        # 라운드 진행 (진보 vs 보수 교대)
        round_results = []
        while self.round_count < self.max_rounds:
            round_result = self.conduct_round()
            if round_result.get('status') == 'completed':
                break
            round_results.append(round_result)
        
        # 토론 마무리
        conclusion = self.conclude_debate()
        
        return {
            'start_result': start_result,
            'round_results': round_results,
            'conclusion': conclusion
        }
    
    def get_debate_status(self) -> Dict:
        """현재 토론 상태를 반환합니다."""
        progressive_count = len([s for s in self.statements if s.get('stance') == '진보'])
        conservative_count = len([s for s in self.statements if s.get('stance') == '보수'])
        
        return {
            'current_topic': self.current_topic,
            'round_count': self.round_count,
            'max_rounds': self.max_rounds,
            'total_statements': len(self.statements),
            'progressive_statements': progressive_count,
            'conservative_statements': conservative_count,
            'total_factchecks': len(self.factcheck_results)
        } 