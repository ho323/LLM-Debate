#!/usr/bin/env python3
"""
정치 토론 시뮬레이션 시스템
진보 vs 보수 에이전트 간 토론, O/X 팩트체크, 요약 기능 제공
"""

import sys
import os
import argparse
from typing import Dict, List
from datetime import datetime
import json
from debate_manager import DebateManager

def ensure_results_dir():
    """결과 저장 디렉토리를 생성합니다."""
    results_dir = os.path.join(os.getcwd(), 'debate_results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    return results_dir

def save_debate_results(results: Dict, topic: str):
    """토론 결과를 debate_results 폴더에 저장합니다."""
    # 결과 저장 디렉토리 확인/생성
    results_dir = ensure_results_dir()
    
    # 파일명 생성 (주제_날짜시간.json)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    topic_slug = topic.replace(" ", "_")[:30]  # 주제를 파일명에 적합하게 변환
    filename = f"{topic_slug}_{timestamp}.json"
    filepath = os.path.join(results_dir, filename)
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"결과가 저장되었습니다: {filepath}")
    except Exception as e:
        print(f"파일 저장 중 오류 발생: {e}")

def main():
    parser = argparse.ArgumentParser(description='AI 정치 토론 시스템 (진보 vs 보수 + 팩트체크)')
    parser.add_argument('--topic', '-t', type=str, 
                       default='최저임금 인상 정책에 대한 찬반 토론',
                       help='토론 주제')
    parser.add_argument('--rounds', '-r', type=int, default=3,
                       help='토론 라운드 수 (기본값: 3)')
    parser.add_argument('--model', '-m', type=str,
                       default='Bllossom/llama-3.2-Korean-Bllossom-3B',
                       help='사용할 Hugging Face 모델')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='대화형 모드로 실행')
    
    args = parser.parse_args()
    
    try:
        # 결과 저장 디렉토리 초기화
        ensure_results_dir()
        
        # 토론 관리자 초기화
        debate_manager = DebateManager(model_name=args.model)
        debate_manager.max_rounds = args.rounds
        
        if args.interactive:
            run_interactive_mode(debate_manager)
        else:
            run_auto_mode(debate_manager, args.topic)
            
    except KeyboardInterrupt:
        print("\n\n토론이 사용자에 의해 중단되었습니다.")
        sys.exit(0)
    except Exception as e:
        print(f"\n오류가 발생했습니다: {e}")
        sys.exit(1)

def run_auto_mode(debate_manager: DebateManager, topic: str):
    """자동 모드: 전체 토론을 한 번에 실행"""
    print(f"🤖 진보 vs 보수 토론을 시작합니다... (실시간 팩트체크 포함)")
    print(f"주제: {topic}")
    print(f"라운드 수: {debate_manager.max_rounds}")
    print("=" * 80)
    
    # 전체 토론 실행
    results = debate_manager.run_full_debate(topic)
    
    print("\n" + "=" * 80)
    print("🎉 토론이 완료되었습니다!")
    
    # 결과 저장 (선택사항)
    save_results = input("\n결과를 파일로 저장하시겠습니까? (y/N): ").lower().strip()
    if save_results == 'y':
        save_debate_results(results, topic)

def run_interactive_mode(debate_manager: DebateManager):
    """대화형 모드: 사용자가 각 단계를 제어"""
    print("🎯 대화형 모드로 시작합니다.")
    print("명령어: start, round, conclude, status, quit")
    
    while True:
        try:
            command = input("\n명령어를 입력하세요: ").strip().lower()
            
            if command == 'quit' or command == 'q':
                print("토론 시스템을 종료합니다.")
                break
                
            elif command == 'start':
                topic = input("토론 주제를 입력하세요: ").strip()
                if topic:
                    debate_manager.start_debate(topic)
                else:
                    print("올바른 주제를 입력해주세요.")
                    
            elif command == 'round':
                if not debate_manager.current_topic:
                    print("먼저 토론을 시작해주세요. (start 명령어 사용)")
                    continue
                    
                result = debate_manager.conduct_round()
                if result.get('status') == 'completed':
                    print("모든 라운드가 완료되었습니다. conclude 명령어로 마무리하세요.")
                    
            elif command == 'conclude':
                if not debate_manager.current_topic:
                    print("진행 중인 토론이 없습니다.")
                    continue
                    
                conclusion = debate_manager.conclude_debate()
                
                save_results = input("\n결과를 파일로 저장하시겠습니까? (y/N): ").lower().strip()
                if save_results == 'y':
                    save_debate_results({
                        'conclusion': conclusion,
                        'topic': debate_manager.current_topic
                    }, debate_manager.current_topic)
                    
            elif command == 'status':
                status = debate_manager.get_debate_status()
                print(f"\n현재 토론 상태:")
                print(f"  주제: {status['current_topic'] or '없음'}")
                print(f"  라운드: {status['round_count']}/{status['max_rounds']}")
                print(f"  진보 발언: {status['progressive_statements']}회")
                print(f"  보수 발언: {status['conservative_statements']}회")
                print(f"  팩트체크: {status['total_factchecks']}건")
                
            elif command == 'help':
                show_help()
                
            else:
                print("알 수 없는 명령어입니다. 'help'를 입력하여 도움말을 확인하세요.")
                
        except KeyboardInterrupt:
            print("\n\n토론이 중단되었습니다.")
            break
        except Exception as e:
            print(f"오류 발생: {e}")

def show_help():
    """도움말 표시"""
    help_text = """
사용 가능한 명령어:
  start       - 새로운 토론 시작
  round       - 다음 라운드 진행 (진보 → 보수 + 각각 팩트체크)
  conclude    - 토론 마무리 및 요약
  status      - 현재 토론 상태 확인
  help        - 이 도움말 표시
  quit/q      - 프로그램 종료

토론 진행 순서:
  1. start 명령어로 토론 시작
  2. round 명령어로 라운드 진행 (여러 번 반복 가능)
  3. conclude 명령어로 토론 마무리
  
각 발언 후 자동으로 O/X 팩트체크가 진행됩니다.
토론 결과는 'debate_results' 폴더에 저장됩니다.
"""
    print(help_text)

if __name__ == "__main__":
    print("🎭 진보 vs 보수 AI 토론 시스템에 오신 것을 환영합니다!")
    print("📊 실시간 O/X 팩트체크 기능 포함")
    print("=" * 60)
    main() 