from typing import Dict, List, Optional
from agents import (
    ProgressiveAgent, 
    ConservativeAgent, 
    ModeratorAgent, 
    FactCheckAgent, 
    SummaryAgent
)

class DebateManager:
    def __init__(self, model_name: str = 'Bllossom/llama-3.2-Korean-Bllossom-3B'):
        print("토론 시스템 초기화 중...")
        
        # 에이전트들 초기화 (모델 공유로 메모리 효율성 고려)
        self.progressive_agent = ProgressiveAgent(model_name)
        self.conservative_agent = ConservativeAgent(model_name)
        self.moderator_agent = ModeratorAgent(model_name)
        self.factcheck_agent = FactCheckAgent(model_name)
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
        
        # 사회자 소개
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
        """한 라운드의 토론을 진행합니다."""
        if self.round_count >= self.max_rounds:
            return {'status': 'completed', 'message': '토론이 종료되었습니다.'}
        
        self.round_count += 1
        print(f"\n--- 라운드 {self.round_count} ---")
        
        round_results = {
            'round': self.round_count,
            'statements': [],
            'factcheck_results': [],
            'moderator_comment': ''
        }
        
        # 1. 진보 에이전트 발언
        progressive_input = {
            'topic': self.current_topic,
            'opponent_statement': self._get_last_conservative_statement()
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
        
        # 2. 보수 에이전트 반박
        conservative_input = {
            'topic': self.current_topic,
            'opponent_statement': progressive_statement
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
        
        # 3. 팩트체크 (백그라운드 처리)
        statements_to_check = [
            {'statement': progressive_statement, 'stance': '진보'},
            {'statement': conservative_statement, 'stance': '보수'}
        ]
        
        factcheck_results = self.factcheck_agent.check_multiple_statements(statements_to_check)
        self.factcheck_results.extend(factcheck_results)
        round_results['factcheck_results'] = factcheck_results
        
        print(f"\n📊 팩트체크:")
        for result in factcheck_results:
            status_icon = {'verified': '✅', 'false': '❌', 'partial': '⚠️', 'unclear': '❓'}.get(
                result['verification_status'], '❓'
            )
            print(f"   {status_icon} {result['stance']}: {result['factcheck_result'][:100]}...")
        
        # 4. 사회자 코멘트
        moderator_input = {
            'action': 'moderate',
            'topic': self.current_topic,
            'statements': self.statements[-2:]  # 최근 2개 발언
        }
        moderator_comment = self.moderator_agent.process_input(moderator_input)
        round_results['moderator_comment'] = moderator_comment
        
        print(f"\n🎯 사회자: {moderator_comment}")
        
        return round_results
    
    def conclude_debate(self) -> Dict:
        """토론을 마무리하고 요약을 생성합니다."""
        print(f"\n=== 토론 마무리 ===")
        
        # 사회자 마무리
        moderator_conclusion = self.moderator_agent.process_input({
            'action': 'conclude',
            'statements': self.statements
        })
        
        print(f"\n🎯 사회자: {moderator_conclusion}")
        
        # 전체 토론 요약
        summary_input = {
            'action': 'summarize_debate',
            'topic': self.current_topic,
            'statements': self.statements,
            'factcheck_results': self.factcheck_results
        }
        debate_summary = self.summary_agent.process_input(summary_input)
        
        print(f"\n📋 토론 요약:\n{debate_summary}")
        
        return {
            'topic': self.current_topic,
            'total_rounds': self.round_count,
            'moderator_conclusion': moderator_conclusion,
            'debate_summary': debate_summary,
            'all_statements': self.statements,
            'all_factcheck_results': self.factcheck_results,
            'status': 'completed'
        }
    
    def run_full_debate(self, topic: str) -> Dict:
        """전체 토론을 자동으로 실행합니다."""
        # 토론 시작
        start_result = self.start_debate(topic)
        
        # 라운드 진행
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
    
    def _get_last_conservative_statement(self) -> Optional[str]:
        """마지막 보수 측 발언을 가져옵니다."""
        for statement in reversed(self.statements):
            if statement.get('stance') == '보수':
                return statement.get('content', '')
        return None
    
    def add_custom_knowledge(self, content: str, source: str, topic: str):
        """RAG 시스템에 새로운 지식을 추가합니다."""
        self.factcheck_agent.rag_system.add_document(content, source, topic)
        print(f"새로운 지식 추가됨: {topic}")
    
    def get_debate_status(self) -> Dict:
        """현재 토론 상태를 반환합니다."""
        return {
            'current_topic': self.current_topic,
            'round_count': self.round_count,
            'max_rounds': self.max_rounds,
            'total_statements': len(self.statements),
            'factcheck_count': len(self.factcheck_results)
        } 