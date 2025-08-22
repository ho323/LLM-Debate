#!/usr/bin/env python3
"""
정치 토론 시뮬레이션 시스템
진보 vs 보수 에이전트 간 토론 및 요약 기능 제공
"""

import sys
import os
import argparse
from typing import Dict, List
from datetime import datetime
import json
from debate_manager import DebateManager
from utils.rag_system import RAGSystem

def ensure_results_dir():
    """결과 저장 디렉토리를 생성합니다."""
    results_dir = os.path.join(os.getcwd(), 'debate_results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    return results_dir

def save_debate_results(results: Dict, topic: str):
    """토론 결과를 debate_results 폴더에 JSON과 MD로 저장합니다."""
    # 결과 저장 디렉토리 확인/생성
    results_dir = ensure_results_dir()
    
    # 파일명 생성 (주제_날짜시간)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    topic_slug = topic.replace(" ", "_")[:30]  # 주제를 파일명에 적합하게 변환
    base_filename = f"{topic_slug}_{timestamp}"
    
    # JSON 파일 저장
    json_filename = f"{base_filename}.json"
    json_filepath = os.path.join(results_dir, json_filename)
    
    try:
        with open(json_filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"JSON 결과가 저장되었습니다: {json_filepath}")
    except Exception as e:
        print(f"JSON 파일 저장 중 오류 발생: {e}")
    
    # MD 파일 저장
    md_filename = f"{base_filename}.md"
    md_filepath = os.path.join(results_dir, md_filename)
    
    try:
        with open(md_filepath, 'w', encoding='utf-8') as f:
            f.write(f"=== AI 정치 토론 결과 ===\n")
            f.write(f"주제: {topic}\n")
            f.write(f"날짜: {datetime.now().strftime('%Y년 %m월 %d일 %H:%M:%S')}\n")
            f.write(f"총 라운드: {results.get('metadata', {}).get('total_rounds', 'N/A')}\n")
            f.write("=" * 50 + "\n\n")
            
            # 시작 결과
            if 'start_result' in results:
                f.write("🎯 토론 시작\n")
                f.write(f"사회자: {results['start_result'].get('moderator_intro', '')}\n\n")
            
            # 라운드별 결과
            if 'round_results' in results:
                f.write("🔄 라운드별 토론\n")
                f.write("-" * 30 + "\n")
                for i, round_result in enumerate(results['round_results'], 1):
                    f.write(f"\n--- 라운드 {i} ---\n")
                    if 'progressive_statement' in round_result:
                        f.write(f"🔵 진보: {round_result['progressive_statement']}\n\n")
                    if 'conservative_statement' in round_result:
                        f.write(f"🔴 보수: {round_result['conservative_statement']}\n\n")
            
            # 요약 결과
            if 'summary_result' in results:
                f.write("\n📊 토론 요약\n")
                f.write("-" * 30 + "\n")
                if 'moderator_conclusion' in results['summary_result']:
                    f.write(f"🎯 사회자 마무리: {results['summary_result']['moderator_conclusion']}\n\n")
                if 'summary' in results['summary_result']:
                    f.write(f"📋 상세 요약:\n{results['summary_result']['summary']}\n\n")
            
            # 통계 정보
            f.write("\n📈 토론 통계\n")
            f.write("-" * 30 + "\n")
            if 'metadata' in results:
                metadata = results['metadata']
                f.write(f"총 발언 수: {metadata.get('total_statements', 'N/A')}건\n")
                f.write(f"진보 측 발언: {metadata.get('progressive_statements', 'N/A')}건\n")
                f.write(f"보수 측 발언: {metadata.get('conservative_statements', 'N/A')}건\n")
            
            f.write(f"\n=== 토론 종료 ===\n")
            
        print(f"MD 결과가 저장되었습니다: {md_filepath}")
    except Exception as e:
        print(f"MD 파일 저장 중 오류 발생: {e}")

def main():
    parser = argparse.ArgumentParser(description='AI 정치 토론 시스템 (진보 vs 보수)')
    parser.add_argument('--topic', '-t', type=str, 
                       default='민생경제 회복을 위한 정부 역할과 정책 방향',
                       help='토론 주제')
    parser.add_argument('--rounds', '-r', type=int, default=3,
                       help='토론 라운드 수 (기본값: 3)')
    parser.add_argument('--model', '-m', type=str,
                       default='C:/Users/User/Documents/EXAONE-4.0-32B-Q4_K_M.gguf',
                       help='사용할 GGUF 모델 경로')
    parser.add_argument('--llama-cli', type=str,
                       default='C:/Users/User/LLM-Debate/llama.cpp/build/bin/Release/llama-cli.exe',
                       help='llama-cli 실행 파일 경로')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='대화형 모드로 실행')
    parser.add_argument('--auto', '-a', action='store_true',
                       help='자동 모드로 전체 토론 실행')
    
    args = parser.parse_args()
    
    # 토론 매니저 초기화
    try:
        debate_manager = DebateManager(model_path=args.model)
        debate_manager.max_rounds = args.rounds
        
        print(f"🤖 진보 vs 보수 토론을 시작합니다...")
        print(f"📝 주제: {args.topic}")
        print(f"🔄 라운드: {args.rounds}")
        print(f"🧠 모델: {args.model}")
        print(f"🔧 llama-cli: {args.llama_cli}")
        
        if args.auto:
            # 자동 모드
            run_auto_debate(debate_manager, args.topic)
        else:
            # 대화형 모드 (기본값)
            run_interactive_debate(debate_manager, args.topic)
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        print("💡 가능한 해결 방법:")
        print("  1. GGUF 모델 파일 경로 확인")
        print("  2. llama-cli 실행 파일 경로 확인")
        print("  3. llama.cpp 빌드 확인")
        print("  4. 시스템 리소스 확인")
        sys.exit(1)

def run_auto_debate(debate_manager: DebateManager, topic: str):
    """자동으로 전체 토론을 실행합니다."""
    try:
        # 토론 시작
        start_result = debate_manager.start_debate(topic)
        
        # 모든 라운드 진행
        round_results = []
        while debate_manager.round_count < debate_manager.max_rounds:
            round_result = debate_manager.proceed_round()
            round_results.append(round_result)
            
            if round_result.get('status') == 'finished':
                break
        
        # 토론 요약
        summary_result = debate_manager.summarize_debate()
        
        # 결과 저장
        full_results = {
            'start_result': start_result,
            'round_results': round_results,
            'summary_result': summary_result,
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_rounds': debate_manager.round_count,
                'topic': topic
            }
        }
        
        save_debate_results(full_results, topic)
        
    except KeyboardInterrupt:
        print("\n\n토론이 중단되었습니다.")
    except Exception as e:
        print(f"❌ 토론 중 오류 발생: {e}")

def run_interactive_debate(debate_manager: DebateManager, topic: str):
    """대화형 모드로 토론을 진행합니다."""
    try:
        # 토론 시작
        debate_manager.start_debate(topic)
        
        print("\n" + "="*60)
        print("🎮 대화형 토론 모드")
        print("="*60)
        
        while True:
            # 현재 상태 표시
            status = debate_manager.get_debate_status()
            print(f"\n📊 현재 상태:")
            print(f"  라운드: {status['current_round']}/{status['max_rounds']}")
            print(f"  발언 수: {status['total_statements']}건")
            
            print(f"\n💡 명령어:")
            print(f"  round       - 다음 라운드 진행 (진보 → 보수)")
            print(f"  status      - 현재 토론 상태 확인")
            print(f"  summary     - 토론 요약 및 종료")
            print(f"  save        - 현재까지 결과 저장")
            print(f"  quit        - 토론 종료")
            
            print(f"\n💬 토론 진행 방식:")
            print(f"각 라운드에서 진보 → 보수 순으로 발언합니다.")
            
            command = input(f"\n명령어를 입력하세요: ").strip().lower()
            
            if command == 'round':
                if status['can_proceed']:
                    debate_manager.proceed_round()
                else:
                    print("⚠️ 최대 라운드에 도달했습니다. 'summary'로 요약하거나 'quit'로 종료하세요.")
                    
            elif command == 'status':
                print_detailed_status(status)
                
            elif command == 'summary':
                summary_result = debate_manager.summarize_debate()
                
                # 결과 저장
                full_results = {
                    'summary_result': summary_result,
                    'metadata': {
                        'timestamp': datetime.now().isoformat(),
                        'total_rounds': debate_manager.round_count,
                        'topic': topic
                    }
                }
                save_debate_results(full_results, topic)
                break
                
            elif command == 'save':
                current_results = {
                    'current_statements': debate_manager.statements,
                    'metadata': {
                        'timestamp': datetime.now().isoformat(),
                        'total_rounds': debate_manager.round_count,
                        'topic': topic,
                        'status': 'in_progress'
                    }
                }
                save_debate_results(current_results, topic)
                
            elif command == 'quit':
                print("토론을 종료합니다.")
                break
                
            else:
                print("❌ 알 수 없는 명령어입니다.")
                
    except KeyboardInterrupt:
        print("\n\n토론이 중단되었습니다.")
    except Exception as e:
        print(f"❌ 토론 중 오류 발생: {e}")

def print_detailed_status(status: Dict):
    """상세한 토론 상태를 출력합니다."""
    print(f"\n📈 상세 토론 상태:")
    print(f"  주제: {status['topic']}")
    print(f"  진행률: {status['current_round']}/{status['max_rounds']} 라운드")
    print(f"  총 발언: {status['total_statements']}건")
    
    if status['can_proceed']:
        print(f"  상태: 진행 중 ⚡")
    else:
        print(f"  상태: 완료 ✅")

if __name__ == "__main__":
    main() 