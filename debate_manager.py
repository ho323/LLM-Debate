from typing import Dict, List, Optional
from agents import (
    ProgressiveAgent, 
    ConservativeAgent, 
    ModeratorAgent, 
    SummaryAgent
)

class DebateManager:
    def __init__(self, model_path: str = '/home/ho/Documents/금융ai/models/EXAONE-4.0-32B-Q4_K_M.gguf'):
        print("토론 시스템 초기화 중...")
        
        # 에이전트들 초기화 (진보 vs 보수만)
        self.progressive_agent = ProgressiveAgent(model_path)
        self.conservative_agent = ConservativeAgent(model_path)
        self.moderator_agent = ModeratorAgent(model_path)
        self.summary_agent = SummaryAgent(model_path)
        
        # 토론 상태 관리
        self.current_topic = ""
        self.statements = []
        self.round_count = 0
        self.max_rounds = 3
        
        print("토론 시스템 초기화 완료!")
    
    def start_debate(self, topic: str) -> Dict:
        """토론을 시작합니다."""
        self.current_topic = topic
        self.statements = []
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
    
    def proceed_round(self) -> Dict:
        """한 라운드를 진행합니다."""
        if self.round_count >= self.max_rounds:
            return {'status': 'finished', 'message': '최대 라운드에 도달했습니다.'}
            
        self.round_count += 1
        round_results = {
            'round': self.round_count,
            'progressive_statement': '',
            'conservative_statement': '',
            'status': 'completed'
        }
        
        print(f"\n--- 라운드 {self.round_count} ---")
        
        # 진보 측 발언
        progressive_statement = self.progressive_agent.generate_argument(
            topic=self.current_topic,
            round_number=self.round_count,
            previous_statements=self.statements
        )
        
        self.statements.append({
            'round': self.round_count,
            'stance': '진보',
            'statement': progressive_statement
        })
        round_results['progressive_statement'] = progressive_statement
        
        print(f"\n🔵 진보: {progressive_statement}")
        
        # 보수 측 발언
        conservative_statement = self.conservative_agent.generate_argument(
            topic=self.current_topic,
            round_number=self.round_count,
            previous_statements=self.statements
        )
        
        self.statements.append({
            'round': self.round_count,
            'stance': '보수',
            'statement': conservative_statement
        })
        round_results['conservative_statement'] = conservative_statement
        
        print(f"\n🔴 보수: {conservative_statement}")
        
        return round_results
    
    def summarize_debate(self) -> Dict:
        """토론을 요약합니다."""
        print(f"\n=== 토론 요약 ===")
        
        # 사회자 마무리
        moderator_conclusion = self.moderator_agent.process_input({
            'action': 'conclude',
            'statements': self.statements
        })
        
        print(f"\n🎯 사회자: {moderator_conclusion}")
        
        # 발언 요약
        print(f"\n📝 발언 요약:")
        prog_count = len([s for s in self.statements if s['stance'] == '진보'])
        cons_count = len([s for s in self.statements if s['stance'] == '보수'])
        print(f"  진보측: {prog_count}건")
        print(f"  보수측: {cons_count}건")
        print(f"  총 라운드: {self.round_count}")
        
        # 상세 요약 생성
        summary = self.summary_agent.summarize_debate(
            topic=self.current_topic,
            statements=self.statements
        )
        
        print(f"\n📊 상세 요약:")
        print(summary)
        
        return {
            'topic': self.current_topic,
            'total_rounds': self.round_count,
            'total_statements': len(self.statements),
            'progressive_statements': prog_count,
            'conservative_statements': cons_count,
            'summary': summary,
            'moderator_conclusion': moderator_conclusion,
            'all_statements': self.statements
        }
    
    def get_debate_status(self) -> Dict:
        """현재 토론 상태를 반환합니다."""
        return {
            'topic': self.current_topic,
            'current_round': self.round_count,
            'max_rounds': self.max_rounds,
            'total_statements': len(self.statements),
            'can_proceed': self.round_count < self.max_rounds
        } 