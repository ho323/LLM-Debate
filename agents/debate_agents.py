from typing import Dict, List, Set, Optional, Tuple
from .base_agent import BaseAgent
from utils.rag_system import RAGSystem
import hashlib
import re

SELF_CONSTRAINTS = """
스타일 규율:
- 한 단락 5~7문장으로 끝낸다. 나열형 접속사(예: 첫째, 둘째, 셋째, 또한, 더불어, 마지막으로)를 쓰지 않는다.
- '입니다/합니다'의 정중체를 유지하되, 마지막 문장은 분명한 요구/검증 요청/목표 제시로 날을 세운다.
- 숫자는 1~2개만 사용하고, 같은 숫자·사례를 라운드마다 반복하지 않는다.

클러시(Clash) 구조(문장 흐름 고정):
1) 상대 주장 핵심 1문장을 요약한다(스틸맨, 왜곡 금지).
2) 그 주장에 깔린 가정 1가지를 짚고 거기에 반례 또는 결손을 꽂는다(핵심만).
3) 새로운 근거 1~2개(데이터/사례)를 제시한다(이전 라운드와 중복 금지).
4) 정책 트레이드오프를 명시하며 우리 해법이 더 나은 이유를 한 줄로 대비한다.
5) 검증 가능한 요구 또는 행동 촉구로 강하게 마무리한다.

금지 목록:
- '첫째/둘째/셋째/마지막으로/한편' 등 열거체, 불릿, 목록, 괄호 시작, 이모지.
- '종합하면/요컨대' 같은 발표체 결론 남발.
"""

class ArgumentTracker:
    """발언 추적 및 중복 방지 클래스"""
    
    def __init__(self):
        self.used_evidence = set()  # 사용된 근거 해시 저장
        self.used_arguments = set()  # 사용된 논거 해시 저장
        self.keyword_usage = {}  # 키워드별 사용 횟수
        self.evidence_sources = set()  # 사용된 출처들
        
    def add_evidence(self, evidence_text: str, source: str) -> None:
        """사용된 근거와 출처를 추가"""
        try:
            if evidence_text and len(evidence_text.strip()) > 0:
                evidence_hash = hashlib.md5(evidence_text.encode('utf-8')).hexdigest()
                self.used_evidence.add(evidence_hash)
            if source and len(source.strip()) > 0:
                self.evidence_sources.add(source)
        except Exception as e:
            print(f"근거 추가 중 오류: {e}")
        
    def add_argument(self, argument: str) -> None:
        """사용된 논거를 추가"""
        try:
            if not argument or len(argument.strip()) == 0:
                return
                
            # 핵심 키워드 추출하여 빈도 체크
            keywords = self._extract_keywords_safe(argument)
            for keyword in keywords:
                if keyword and len(keyword.strip()) > 0:
                    self.keyword_usage[keyword] = self.keyword_usage.get(keyword, 0) + 1
                
            argument_hash = hashlib.md5(argument.encode('utf-8')).hexdigest()
            self.used_arguments.add(argument_hash)
        except Exception as e:
            print(f"논거 추가 중 오류: {e}")
    
    def is_evidence_used(self, evidence_text: str) -> bool:
        """근거가 이미 사용되었는지 확인"""
        try:
            if not evidence_text or len(evidence_text.strip()) == 0:
                return False
            evidence_hash = hashlib.md5(evidence_text.encode('utf-8')).hexdigest()
            return evidence_hash in self.used_evidence
        except Exception:
            return False
    
    def is_source_overused(self, source: str, max_usage: int = 2) -> bool:
        """특정 출처가 과도하게 사용되었는지 확인"""
        try:
            if not source or len(source.strip()) == 0:
                return False
            return list(self.evidence_sources).count(source) >= max_usage
        except Exception:
            return False
    
    def get_keyword_frequency(self, keyword: str) -> int:
        """특정 키워드의 사용 빈도 반환"""
        return self.keyword_usage.get(keyword, 0)
    
    def _extract_keywords_safe(self, text: str) -> List[str]:
        """안전한 키워드 추출"""
        if not text or len(text.strip()) == 0:
            return []
        
        try:
            # 경제/정치 관련 주요 키워드 패턴
            patterns = [
                r'GDP|성장률|실업률|물가상승률|인플레이션|경기침체',
                r'최저임금|소득불평등|중산층|서민|노동자',
                r'재정지출|국가부채|세율|세수|예산',
                r'규제완화|민영화|공기업|대기업|중소기업',
                r'복지|연금|건강보험|교육비|의료비',
                r'청년|일자리|취업|창업|고용',
                r'부동산|집값|전세|월세|주택'
            ]
            
            keywords = []
            for pattern in patterns:
                try:
                    matches = re.findall(pattern, text)
                    if matches:
                        keywords.extend([m for m in matches if m and len(m.strip()) > 0])
                except Exception:
                    continue
            
            # 중복 제거 및 정리
            unique_keywords = list(set(keywords))
            return unique_keywords[:10]  # 최대 10개만 반환
            
        except Exception as e:
            print(f"키워드 추출 중 오류: {e}")
            return []

class ProgressiveAgent(BaseAgent):
    def __init__(self, model_path: str = 'C:/Users/User/Documents/EXAONE-4.0-32B-Q4_K_M.gguf', rag_system: Optional[RAGSystem] = None):
        super().__init__(model_path)
        self.stance = "진보"
        self.rag_system = rag_system
        self.my_previous_statements = []
        self.opponent_previous_statements = []
        
        # 중복 방지를 위한 추적기
        self.argument_tracker = ArgumentTracker()
        self.shared_tracker = None  # 공유 추적기 (상대방과 공유)
        
        self.system_prompt = """너는 더불어민주당 소속 진보 정치인이다. 다음과 같은 특징을 가져라:

말투 특징:
- "국민 여러분" 같은 호명을 자주 사용
- "충분히... 가능하다고 생각합니다" 같은 점진적 표현 사용
- "저희가 보기에는..." "분명히... 있습니다" 같은 확신적 표현
- 구체적 수치와 사례를 제시하는 실무적 접근
- 상대방 정책의 문제점을 구체적으로 지적

정책 성향:
- 과감한 재정정책과 적극적 정부 역할 강조
- 소득 불평등과 민생경제 문제에 집중
- 중소기업과 자영업자, 플랫폼 노동자 보호
- 대기업 특혜 정책 비판
- 복지 확대와 공공서비스 강화 주장

논리 구조:
- 현실 상황 진단 → 정부 정책 실패 지적 → 구체적 대안 제시
- 상대방 정책의 부작용 사례 제시
- 서민과 중산층의 관점에서 접근

형식 제한(매우 중요):
- 출력은 항상 '한 단락'의 평서문으로 작성한다.
- 문장은 반드시 끝맺는다. 문장이 끊길 것 같을 시 전 문장에서 마무리 한다.
- 줄바꿈, 제목, 머리말, 소제목 금지.
- 목록, 번호(1. ① 1), 하이픈(-), 불릿(•), 대시(—, –), 이모지 사용 금지.
- 문장 시작에 숫자/괄호/불릿/이모지 배치 금지.
- 발화자의 멘트만 출력한다.

다음과 같은 논리적 사고 과정을 거쳐라:
<thinking>
1. 상황 분석: 현재 주어진 주제의 핵심 문제는 무엇인가?
2. 약점 파악: 그들 주장의 허점이나 모순점은 무엇인가?
3. 반박 근거: 우리가 제시할 수 있는 반증 데이터나 사례는?
4. 진보 대안: 우리의 해결책이 왜 더 나은가?
5. 감정적 호소: 국민들의 공감을 얻을 수 있는 포인트는?
</thinking>

""" + SELF_CONSTRAINTS

    def set_shared_tracker(self, shared_tracker: ArgumentTracker):
        """상대방과 공유하는 추적기 설정"""
        self.shared_tracker = shared_tracker

    def _get_filtered_evidence(self, topic: str, max_docs: int = 5) -> List[Dict]:
        """중복되지 않는 새로운 근거 검색"""
        if not self.rag_system:
            return []
        
        try:
            # 더 많은 문서를 검색하여 필터링 여지 확보
            all_docs = self.rag_system.search(query=topic, stance_filter="진보", top_k=max_docs*2)
            filtered_docs = []
            
            for doc in all_docs:
                if not doc or not isinstance(doc, dict):
                    continue
                    
                evidence_text = doc.get('text', '')
                source = doc.get('source', '')
                
                # 중복 체크
                if self.argument_tracker.is_evidence_used(evidence_text):
                    continue
                if self.shared_tracker and self.shared_tracker.is_evidence_used(evidence_text):
                    continue
                if self.argument_tracker.is_source_overused(source):
                    continue
                    
                filtered_docs.append(doc)
                if len(filtered_docs) >= max_docs:
                    break
                    
            return filtered_docs
        except Exception as e:
            print(f"근거 검색 중 오류: {e}")
            return []

    def _analyze_opponent_weakness_safe(self, opponent_statements: List[str]) -> Dict:
        """상대방 발언의 약점과 모순점 분석 (안전한 버전)"""
        # 기본 구조 초기화
        analysis = {
            'contradictions': [],
            'weak_points': [],
            'overused_arguments': [],
            'missing_evidence': []
        }
        
        # 빈 리스트 체크
        if not opponent_statements or len(opponent_statements) == 0:
            return analysis
        
        try:
            # 모순점 찾기 (더 정확한 키워드 매칭)
            contradiction_pairs = [
                (['규제완화', '완화'], ['시장개입', '개입', '정부역할']),
                (['재정건전성', '건전성'], ['지원확대', '확대', '지출증가']),
                (['민간주도', '민간'], ['정부역할', '정부주도', '국가개입']),
            ]
            
            for stmt in opponent_statements:
                if not stmt or len(stmt.strip()) == 0:
                    continue
                    
                stmt_lower = stmt.lower()
                
                for pair in contradiction_pairs:
                    left_keywords, right_keywords = pair
                    
                    # 각 그룹에서 키워드 발견 여부 확인
                    left_found = any(keyword in stmt_lower for keyword in left_keywords)
                    right_found = any(keyword in stmt_lower for keyword in right_keywords)
                    
                    if left_found and right_found:
                        # 실제 발견된 키워드 찾기
                        found_left = next((k for k in left_keywords if k in stmt_lower), left_keywords[0])
                        found_right = next((k for k in right_keywords if k in stmt_lower), right_keywords[0])
                        analysis['contradictions'].append(f"{found_left}와 {found_right} 모순")
            
            # 반복되는 논거 찾기 (안전한 키워드 추출)
            keyword_counts = {}
            for stmt in opponent_statements:
                if not stmt or len(stmt.strip()) == 0:
                    continue
                    
                try:
                    keywords = self.argument_tracker._extract_keywords_safe(stmt)
                    for keyword in keywords:
                        if keyword and len(keyword.strip()) > 0:
                            keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
                except Exception:
                    continue
            
            # 3번 이상 사용된 키워드를 반복 논거로 분류
            overused = [k for k, v in keyword_counts.items() if v >= 3 and k]
            analysis['overused_arguments'] = overused[:5]  # 최대 5개만
            
        except Exception as e:
            print(f"상대방 약점 분석 중 오류: {e}")
        
        return analysis

    def generate_argument(self, topic: str, round_number: int, previous_statements: List[Dict]) -> str:
        """안전한 논증 생성 메서드"""
        try:
            # 발언 기록 업데이트
            self.update_statement_history(previous_statements)
            
            # 상대방 약점 분석 (안전한 버전)
            try:
                opponent_analysis = self._analyze_opponent_weakness_safe(self.opponent_previous_statements)
            except Exception as e:
                print(f"상대방 분석 실패: {e}")
                opponent_analysis = {
                    'contradictions': [],
                    'weak_points': [],
                    'overused_arguments': [],
                    'missing_evidence': []
                }
            
            # 필터링된 근거 검색
            try:
                filtered_docs = self._get_filtered_evidence(topic)
            except Exception as e:
                print(f"근거 검색 실패: {e}")
                filtered_docs = []
            
            # 근거 텍스트 생성
            evidence_text = ""
            selected_docs = []
            if filtered_docs:
                for doc in filtered_docs[:3]:
                    try:
                        if doc and isinstance(doc, dict) and 'text' in doc and 'source' in doc:
                            evidence_text += f"- {doc['text']} (출처: {doc['source']})\n"
                            selected_docs.append(doc)
                    except Exception:
                        continue
            
            evidence_section = f"\n\n📚 새로운 참고 기사:\n{evidence_text}" if evidence_text else ""
            
            # 과거 발언 요약
            try:
                my_statements_summary = self._summarize_previous_arguments(self.my_previous_statements)
                opponent_statements_summary = self._summarize_previous_arguments(self.opponent_previous_statements)
            except Exception as e:
                print(f"발언 요약 실패: {e}")
                my_statements_summary = "기본 논점"
                opponent_statements_summary = "기본 논점"
            
            # 상대방 약점 분석 결과
            weakness_section = ""
            try:
                if (opponent_analysis.get('contradictions') or 
                    opponent_analysis.get('overused_arguments')):
                    weakness_section = f"\n\n🎯 상대방 약점 분석:\n"
                    if opponent_analysis.get('contradictions'):
                        contradictions = opponent_analysis['contradictions'][:3]
                        weakness_section += f"모순점: {', '.join(contradictions)}\n"
                    if opponent_analysis.get('overused_arguments'):
                        overused = opponent_analysis['overused_arguments'][:3]
                        weakness_section += f"반복 논거: {', '.join(overused)}\n"
            except Exception as e:
                print(f"약점 분석 섹션 생성 실패: {e}")
                weakness_section = ""

            # 프롬프트 생성
            if round_number == 1:
                prompt = f"""너는 더불어민주당 소속 진보 정치인이다.

토론 주제: {topic}{evidence_section}

먼저 다음 단계별로 논리적 사고를 진행하라:
<thinking>
1. 상황 분석: 현재 경제/사회 상황의 핵심 문제는 무엇인가?
2. 근거 선택: 제공된 새로운 근거 중 가장 강력한 것은?
3. 핵심 메시지: 국민들에게 전달할 차별화된 대안은?
4. 감정적 호소: 공감을 얻을 수 있는 구체적 사례는?
</thinking>

그 다음 정중한 호칭을 포함하되 과장 없이, 존댓말로 구체적 수치·사례로 현재 상황의 심각성을 제시하고, 정부나 보수 정책의 실패를 새로운 근거로 비판하며, 진보적 대안의 필요성을 분명히 밝힌 뒤 2~3문장으로 힘 있게 마무리하라.

형식 제한: <thinking> 부분은 출력하지 말고, 줄바꿈 없이 단락 하나로만 작성하고, 목록·숫자·괄호 시작·하이픈·불릿·이모지·제목을 사용하지 마라. 발화자의 멘트만 출력하라."""
            else:
                last_conservative = self._get_last_conservative_statement_safe(previous_statements)
                prompt = f"""너는 더불어민주당 소속 진보 정치인이다.

토론 주제: {topic}
상대(보수)의 최근 주장: "{last_conservative}"{evidence_section}

📝 내 과거 주요 논점: {my_statements_summary}
🔴 상대 과거 주요 논점: {opponent_statements_summary}{weakness_section}

먼저 다음 단계별로 논리적 사고를 진행하라:
<thinking>
1. 상대 분석: 상대가 최근에 주장한 핵심과 허점은?
2. 차별화: 내 과거 발언과 다른 새로운 각도는?
3. 약점 공략: 상대의 모순점이나 반복 논거를 어떻게 공격할까?
4. 신규 근거: 제공된 새로운 근거를 어떻게 활용할까?
5. 반전 논리: 상대 논리를 뒤집을 수 있는 관점은?
</thinking>

중요한 제약사항:
- 과거 논점과 겹치지 않는 새로운 각도로 접근하라
- 상대의 약점과 모순점을 정확히 지적하라
- 새로운 근거를 활용하여 차별화된 반박을 하라
- 감정적이지만 논리적인 공격을 하라

그 다음 보수 측의 최근 주장을 정확히 파악하고 그 허점을 날카롭게 지적한 뒤, 존댓말로 새로운 구체적 데이터와 사례로 반증하고, 서민·중산층 관점에서 차별화된 대안을 제시하며 강력하게 마무리하라.

형식 제한: <thinking> 부분과 보수 측 주장은 출력하지 말고, 목록·숫자·괄호 시작·하이픈·불릿·이모지·제목을 사용하지 마라. 발화자의 멘트만 출력하라."""
            
            response = self.generate_response(prompt)
            
            # 사용된 근거와 논거 기록
            if response:
                try:
                    self.my_previous_statements.append(response)
                    self.argument_tracker.add_argument(response)
                    
                    # 사용된 근거들 기록
                    for doc in selected_docs:
                        if doc and isinstance(doc, dict) and 'text' in doc and 'source' in doc:
                            self.argument_tracker.add_evidence(doc['text'], doc['source'])
                            if self.shared_tracker:
                                self.shared_tracker.add_evidence(doc['text'], doc['source'])
                except Exception as e:
                    print(f"발언 기록 실패: {e}")
            
            return response if response else "죄송합니다. 일시적으로 응답을 생성할 수 없습니다."
            
        except Exception as e:
            print(f"논증 생성 중 전체 오류: {e}")
            return "죄송합니다. 시스템 오류로 인해 응답을 생성할 수 없습니다."

    def _get_last_conservative_statement_safe(self, statements: List[Dict]) -> str:
        """안전한 상대방 마지막 발언 추출"""
        try:
            if not statements:
                return ""
            
            for stmt in reversed(statements):
                if stmt and isinstance(stmt, dict) and stmt.get('stance') == '보수':
                    statement = stmt.get('statement', '')
                    return statement if statement else ""
            return ""
        except Exception as e:
            print(f"상대방 발언 추출 실패: {e}")
            return ""

    def _summarize_previous_arguments(self, statements: List[str]) -> str:
        """이전 발언들의 핵심 논점 요약"""
        if not statements:
            return "없음"
        
        try:
            # 최근 2개 발언의 핵심 키워드만 추출
            recent_statements = statements[-2:] if len(statements) > 2 else statements
            all_keywords = []
            
            for stmt in recent_statements:
                if stmt and len(stmt.strip()) > 0:
                    keywords = self.argument_tracker._extract_keywords_safe(stmt)
                    all_keywords.extend(keywords)
            
            # 중복 제거하고 빈도순 정렬
            keyword_freq = {}
            for keyword in all_keywords:
                if keyword and len(keyword.strip()) > 0:
                    keyword_freq[keyword] = keyword_freq.get(keyword, 0) + 1
            
            top_keywords = sorted(keyword_freq.keys(), key=lambda x: keyword_freq[x], reverse=True)[:5]
            return ", ".join(top_keywords) if top_keywords else "기본 논점"
        except Exception as e:
            print(f"발언 요약 중 오류: {e}")
            return "기본 논점"

    def update_statement_history(self, previous_statements: List[Dict]):
        """발언 기록을 업데이트합니다."""
        try:
            self.my_previous_statements = []
            self.opponent_previous_statements = []
            
            for stmt in previous_statements:
                if stmt and isinstance(stmt, dict):
                    if stmt.get('stance') == '진보':
                        statement = stmt.get('statement', '')
                        if statement and len(statement.strip()) > 0:
                            self.my_previous_statements.append(statement)
                    elif stmt.get('stance') == '보수':
                        statement = stmt.get('statement', '')
                        if statement and len(statement.strip()) > 0:
                            self.opponent_previous_statements.append(statement)
        except Exception as e:
            print(f"발언 기록 업데이트 실패: {e}")

    def get_my_previous_statements(self) -> List[str]:
        """내가 과거에 한 발언들을 반환합니다."""
        return self.my_previous_statements.copy()

    def get_opponent_previous_statements(self) -> List[str]:
        """상대가 과거에 한 발언들을 반환합니다."""
        return self.opponent_previous_statements.copy()

    def _build_context(self, statements: List[Dict]) -> str:
        if not statements:
            return "첫 라운드입니다."
        
        try:
            recent_statements = statements[-2:] if len(statements) >= 2 else statements
            context_parts = []
            for stmt in recent_statements:
                if stmt and isinstance(stmt, dict):
                    stance = stmt.get('stance', '')
                    content = stmt.get('statement', '')
                    if content:
                        content_preview = content[:50] + "..." if len(content) > 50 else content
                        context_parts.append(f"{stance}: {content_preview}")
            
            return " | ".join(context_parts)
        except Exception:
            return "컨텍스트 생성 실패"

    def process_input(self, input_data: Dict) -> str:
        """기존 인터페이스와의 호환성을 위한 메서드"""
        try:
            topic = input_data.get('topic', '')
            round_number = input_data.get('round_number', 1)
            previous_statements = input_data.get('previous_statements', [])
            
            return self.generate_argument(topic, round_number, previous_statements)
        except Exception as e:
            print(f"입력 처리 실패: {e}")
            return "입력 처리 중 오류가 발생했습니다."

class ConservativeAgent(BaseAgent):
    def __init__(self, model_path: str = 'C:/Users/User/Documents/EXAONE-4.0-32B-Q4_K_M.gguf', rag_system: Optional[RAGSystem] = None):
        super().__init__(model_path)
        self.stance = "보수"
        self.rag_system = rag_system
        self.my_previous_statements = []
        self.opponent_previous_statements = []
        
        # 중복 방지를 위한 추적기
        self.argument_tracker = ArgumentTracker()
        self.shared_tracker = None  # 공유 추적기
        
        self.system_prompt = """너는 국민의힘 소속 보수 정치인이다. 다음과 같은 특징을 가져라:

말투 특징:
- "저희는... 생각합니다" "...하겠습니다" 같은 겸손하면서도 확신적인 표현
- "안타깝게도..." "그러나..." 같은 상황 인식 후 반박
- "이 점 말씀드리고..." 같은 체계적 설명
- 책임감과 성찰을 보이는 표현 사용
- 구체적 수치와 데이터를 활용한 실증적 접근

정책 성향:
- 시장경제와 민간 주도 성장 강조
- 재정 건전성과 국가부채 우려
- 규제 완화와 기업 투자 환경 개선
- 개인 책임과 자유 선택의 가치
- 혁신과 도전 정신 중시

논리 구조:
- 상황 인식 → 상대방 정책의 문제점 지적 → 시장경제적 해법 제시
- 재정 부담과 장기적 부작용 경고
- 성공 사례와 경험적 근거 제시
- 국가 경쟁력과 미래 세대 책임감 강조

형식 제한(매우 중요):
- 출력은 항상 '한 단락'의 평서문으로 작성한다.
- 문장은 반드시 끝맺는다. 문장이 끊길 것 같을 시 전 문장에서 마무리 한다.
- 줄바꿈, 제목, 머리말, 소제목 금지.
- 목록, 번호(1. ① 1), 하이픈(-), 불릿(•), 대시(—, –), 이모지 사용 금지.
- 문장 시작에 숫자/괄호/불릿/이모지 배치 금지.
- 발화자의 멘트만 출력한다.

다음과 같은 논리적 사고 과정을 거쳐라:
<thinking>
1. 상대방 주장 분석: 진보 측 주장의 핵심 논리는 무엇인가?
2. 근거 제시: 우리가 제시할 수 있는 구체적 데이터나 사례는?
3. 보수적 관점: 시장경제와 재정건전성 관점에서 어떻게 바라보는가?
4. 장기적 부작용: 진보 정책이 경제와 재정에 미칠 장기 영향은?
5. 대안 제시: 시장 원리 기반의 실현 가능한 해법은?
</thinking>

""" + SELF_CONSTRAINTS

    def set_shared_tracker(self, shared_tracker: ArgumentTracker):
        """상대방과 공유하는 추적기 설정"""
        self.shared_tracker = shared_tracker

    def _get_filtered_evidence(self, topic: str, max_docs: int = 5) -> List[Dict]:
        """중복되지 않는 새로운 근거 검색"""
        if not self.rag_system:
            return []
        
        try:
            all_docs = self.rag_system.search(query=topic, stance_filter="보수", top_k=max_docs*2)
            filtered_docs = []
            
            for doc in all_docs:
                if not doc or not isinstance(doc, dict):
                    continue
                    
                evidence_text = doc.get('text', '')
                source = doc.get('source', '')
                
                if self.argument_tracker.is_evidence_used(evidence_text):
                    continue
                if self.shared_tracker and self.shared_tracker.is_evidence_used(evidence_text):
                    continue
                if self.argument_tracker.is_source_overused(source):
                    continue
                    
                filtered_docs.append(doc)
                if len(filtered_docs) >= max_docs:
                    break
                    
            return filtered_docs
        except Exception as e:
            print(f"근거 검색 중 오류: {e}")
            return []

    def _analyze_opponent_weakness_safe(self, opponent_statements: List[str]) -> Dict:
        """상대방 발언의 약점과 모순점 분석 (보수 관점, 안전한 버전)"""
        # 기본 구조 초기화
        analysis = {
            'contradictions': [],
            'weak_points': [],
            'overused_arguments': [],
            'missing_evidence': []
        }
        
        # 빈 리스트 체크
        if not opponent_statements or len(opponent_statements) == 0:
            return analysis
        
        try:
            # 진보 측 모순점 찾기
            contradiction_pairs = [
                (['재정지출', '지출확대'], ['건전성', '재정건전성']),
                (['규제강화', '강화'], ['경제성장', '성장률']),
                (['복지확대', '복지증가'], ['세수부족', '재정부족']),
            ]
            
            for stmt in opponent_statements:
                if not stmt or len(stmt.strip()) == 0:
                    continue
                    
                stmt_lower = stmt.lower()
                
                for pair in contradiction_pairs:
                    left_keywords, right_keywords = pair
                    
                    # 각 그룹에서 키워드 발견 여부 확인
                    left_found = any(keyword in stmt_lower for keyword in left_keywords)
                    right_found = any(keyword in stmt_lower for keyword in right_keywords)
                    
                    if left_found and right_found:
                        # 실제 발견된 키워드 찾기
                        found_left = next((k for k in left_keywords if k in stmt_lower), left_keywords[0])
                        found_right = next((k for k in right_keywords if k in stmt_lower), right_keywords[0])
                        analysis['contradictions'].append(f"{found_left}와 {found_right} 모순")
            
            # 반복 논거 찾기
            keyword_counts = {}
            for stmt in opponent_statements:
                if not stmt or len(stmt.strip()) == 0:
                    continue
                    
                try:
                    keywords = self.argument_tracker._extract_keywords_safe(stmt)
                    for keyword in keywords:
                        if keyword and len(keyword.strip()) > 0:
                            keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
                except Exception:
                    continue
            
            # 3번 이상 사용된 키워드를 반복 논거로 분류
            overused = [k for k, v in keyword_counts.items() if v >= 3 and k]
            analysis['overused_arguments'] = overused[:5]  # 최대 5개만
            
        except Exception as e:
            print(f"상대방 약점 분석 중 오류: {e}")
        
        return analysis

    def generate_argument(self, topic: str, round_number: int, previous_statements: List[Dict]) -> str:
        """안전한 논증 생성 메서드"""
        try:
            self.update_statement_history(previous_statements)
            
            # 상대방 약점 분석
            try:
                opponent_analysis = self._analyze_opponent_weakness_safe(self.opponent_previous_statements)
            except Exception as e:
                print(f"상대방 분석 실패: {e}")
                opponent_analysis = {
                    'contradictions': [],
                    'weak_points': [],
                    'overused_arguments': [],
                    'missing_evidence': []
                }
            
            # 필터링된 근거 검색
            try:
                filtered_docs = self._get_filtered_evidence(topic)
            except Exception as e:
                print(f"근거 검색 실패: {e}")
                filtered_docs = []
            
            # 근거 텍스트 생성
            evidence_text = ""
            selected_docs = []
            if filtered_docs:
                for doc in filtered_docs[:3]:
                    try:
                        if doc and isinstance(doc, dict) and 'text' in doc and 'source' in doc:
                            evidence_text += f"- {doc['text']} (출처: {doc['source']})\n"
                            selected_docs.append(doc)
                    except Exception:
                        continue
            
            evidence_section = f"\n\n📚 새로운 참고 기사:\n{evidence_text}" if evidence_text else ""
            
            # 과거 발언 요약
            try:
                my_statements_summary = self._summarize_previous_arguments(self.my_previous_statements)
                opponent_statements_summary = self._summarize_previous_arguments(self.opponent_previous_statements)
            except Exception as e:
                print(f"발언 요약 실패: {e}")
                my_statements_summary = "기본 논점"
                opponent_statements_summary = "기본 논점"
            
            # 상대방 약점 분석 결과
            weakness_section = ""
            try:
                if (opponent_analysis.get('contradictions') or 
                    opponent_analysis.get('overused_arguments')):
                    weakness_section = f"\n\n🎯 상대방 약점 분석:\n"
                    if opponent_analysis.get('contradictions'):
                        contradictions = opponent_analysis['contradictions'][:3]
                        weakness_section += f"모순점: {', '.join(contradictions)}\n"
                    if opponent_analysis.get('overused_arguments'):
                        overused = opponent_analysis['overused_arguments'][:3]
                        weakness_section += f"반복 논거: {', '.join(overused)}\n"
            except Exception as e:
                print(f"약점 분석 섹션 생성 실패: {e}")
                weakness_section = ""

            if round_number == 1:
                prompt = f"""너는 국민의힘 소속 보수 정치인이다.

토론 주제: {topic}{evidence_section}

먼저 다음 단계별로 논리적 사고를 진행하라:
<thinking>
1. 상황 분석: 현재 경제/사회 상황을 시장경제 관점에서 보면?
2. 근거 선택: 제공된 새로운 근거 중 가장 설득력 있는 것은?
3. 보수적 해법: 시장 원리와 재정건전성을 지키면서 해결할 방법은?
4. 장기 비전: 국가 경쟁력과 미래 세대를 위한 책임은?
</thinking>

그 다음 현 상황을 구체적 수치와 데이터로 냉정히 진단하고 존댓말로 우려를 밝힌 다음, 진보 정책의 문제점을 새로운 근거와 함께 지적하고, 시장경제·재정건전성의 중요성을 실증적 데이터로 강조하며 책임 있는 어조로 마무리하라.

형식 제한: <thinking> 부분은 출력하지 말고, 줄바꿈 없이 단락 하나로만 작성하고, 목록·숫자·괄호 시작·하이픈·불릿·이모지·제목을 절대 사용하지 마라. 발화자의 멘트만 출력하라."""
            else:
                last_progressive = self._get_last_progressive_statement_safe(previous_statements)
                prompt = f"""너는 국민의힘 소속 보수 정치인이다.

토론 주제: {topic}
상대(진보)의 최근 주장: "{last_progressive}"{evidence_section}

📝 내 과거 주요 논점: {my_statements_summary}
🔵 상대 과거 주요 논점: {opponent_statements_summary}{weakness_section}

먼저 다음 단계별로 논리적 사고를 진행하라:
<thinking>
1. 상대방 주장 분석: 진보 측이 최근에 주장한 핵심과 문제점은?
2. 차별화 전략: 내 과거 발언과 다른 새로운 보수적 관점은?
3. 약점 공략: 상대의 모순점과 반복 논거를 어떻게 반박할까?
4. 신규 근거 활용: 새로운 근거로 어떤 논리를 구성할까?
5. 시장경제 원리: 자유시장과 개인책임 관점에서의 해법은?
6. 장기적 관점: 재정건전성과 국가경쟁력 측면의 우려는?
</thinking>

중요한 제약사항:
- 과거 논점과 차별화된 새로운 보수적 각도로 접근하라
- 상대의 약점과 모순을 구체적으로 지적하라
- 새로운 근거를 바탕으로 설득력 있는 반박을 하라
- 감정에 치우치지 않고 데이터 기반으로 논증하라

그 다음 상대의 최근 주장을 존댓말로 논리적으로 반박하고, 구체적 수치와 새로운 경험적 데이터로 재정 부담·장기 부작용을 입증하며, 실증적 근거를 들어 차별화된 보수적 해법을 제시하고 존댓말이지만 강력하게 마무리하라.

형식 제한: <thinking> 부분과 진보 측 주장은 출력하지 말고, 목록·숫자·괄호 시작·하이픈·불릿·이모지·제목을 절대 사용하지 마라. 발화자의 멘트만 출력하라."""
            
            response = self.generate_response(prompt)
            
            # 사용된 근거와 논거 기록
            if response:
                try:
                    self.my_previous_statements.append(response)
                    self.argument_tracker.add_argument(response)
                    
                    # 사용된 근거들 기록
                    for doc in selected_docs:
                        if doc and isinstance(doc, dict) and 'text' in doc and 'source' in doc:
                            self.argument_tracker.add_evidence(doc['text'], doc['source'])
                            if self.shared_tracker:
                                self.shared_tracker.add_evidence(doc['text'], doc['source'])
                except Exception as e:
                    print(f"발언 기록 실패: {e}")
            
            return response if response else "죄송합니다. 일시적으로 응답을 생성할 수 없습니다."
            
        except Exception as e:
            print(f"논증 생성 중 전체 오류: {e}")
            return "죄송합니다. 시스템 오류로 인해 응답을 생성할 수 없습니다."

    def _get_last_progressive_statement_safe(self, statements: List[Dict]) -> str:
        """안전한 상대방 마지막 발언 추출"""
        try:
            if not statements:
                return ""
            
            for stmt in reversed(statements):
                if stmt and isinstance(stmt, dict) and stmt.get('stance') == '진보':
                    statement = stmt.get('statement', '')
                    return statement if statement else ""
            return ""
        except Exception as e:
            print(f"상대방 발언 추출 실패: {e}")
            return ""

    def _summarize_previous_arguments(self, statements: List[str]) -> str:
        """이전 발언들의 핵심 논점 요약"""
        if not statements:
            return "없음"
        
        try:
            recent_statements = statements[-2:] if len(statements) > 2 else statements
            all_keywords = []
            
            for stmt in recent_statements:
                if stmt and len(stmt.strip()) > 0:
                    keywords = self.argument_tracker._extract_keywords_safe(stmt)
                    all_keywords.extend(keywords)
            
            keyword_freq = {}
            for keyword in all_keywords:
                if keyword and len(keyword.strip()) > 0:
                    keyword_freq[keyword] = keyword_freq.get(keyword, 0) + 1
            
            top_keywords = sorted(keyword_freq.keys(), key=lambda x: keyword_freq[x], reverse=True)[:5]
            return ", ".join(top_keywords) if top_keywords else "기본 논점"
        except Exception as e:
            print(f"발언 요약 중 오류: {e}")
            return "기본 논점"

    def update_statement_history(self, previous_statements: List[Dict]):
        """발언 기록을 업데이트합니다."""
        try:
            self.my_previous_statements = []
            self.opponent_previous_statements = []
            
            for stmt in previous_statements:
                if stmt and isinstance(stmt, dict):
                    if stmt.get('stance') == '보수':
                        statement = stmt.get('statement', '')
                        if statement and len(statement.strip()) > 0:
                            self.my_previous_statements.append(statement)
                    elif stmt.get('stance') == '진보':
                        statement = stmt.get('statement', '')
                        if statement and len(statement.strip()) > 0:
                            self.opponent_previous_statements.append(statement)
        except Exception as e:
            print(f"발언 기록 업데이트 실패: {e}")

    def get_my_previous_statements(self) -> List[str]:
        """내가 과거에 한 발언들을 반환합니다."""
        return self.my_previous_statements.copy()

    def get_opponent_previous_statements(self) -> List[str]:
        """상대가 과거에 한 발언들을 반환합니다."""
        return self.opponent_previous_statements.copy()

    def _build_context(self, statements: List[Dict]) -> str:
        if not statements:
            return "첫 라운드입니다."
        
        try:
            recent_statements = statements[-2:] if len(statements) >= 2 else statements
            context_parts = []
            for stmt in recent_statements:
                if stmt and isinstance(stmt, dict):
                    stance = stmt.get('stance', '')
                    content = stmt.get('statement', '')
                    if content:
                        content_preview = content[:50] + "..." if len(content) > 50 else content
                        context_parts.append(f"{stance}: {content_preview}")
            
            return " | ".join(context_parts)
        except Exception:
            return "컨텍스트 생성 실패"

    def process_input(self, input_data: Dict) -> str:
        """기존 인터페이스와의 호환성을 위한 메서드"""
        try:
            topic = input_data.get('topic', '')
            round_number = input_data.get('round_number', 1)
            previous_statements = input_data.get('previous_statements', [])
            
            return self.generate_argument(topic, round_number, previous_statements)
        except Exception as e:
            print(f"입력 처리 실패: {e}")
            return "입력 처리 중 오류가 발생했습니다."

# 토론 관리자 클래스 (두 에이전트 간 공유 추적기 설정)
class DebateManager:
    """토론 진행 및 중복 방지 관리 클래스"""
    
    def __init__(self, progressive_agent: ProgressiveAgent, conservative_agent: ConservativeAgent):
        self.progressive_agent = progressive_agent
        self.conservative_agent = conservative_agent
        
        # 공유 추적기 생성 및 설정
        self.shared_tracker = ArgumentTracker()
        self.progressive_agent.set_shared_tracker(self.shared_tracker)
        self.conservative_agent.set_shared_tracker(self.shared_tracker)
        
        # 토론 기록
        self.debate_history = []
        
        # 토론 상태 관리
        self.round_count = 0
        self.max_rounds = 5
        self.topic = ""
        self.statements = []
    
    def start_debate(self, topic: str):
        """토론 시작"""
        self.topic = topic
        self.round_count = 0
        self.debate_history = []
        self.statements = []
        print(f"📢 토론 주제: {topic}")
        return {'topic': topic, 'status': 'started'}
    
    def proceed_round(self):
        """한 라운드 진행"""
        if self.round_count >= self.max_rounds:
            return {'status': 'finished', 'message': '최대 라운드에 도달했습니다.'}
        
        self.round_count += 1
        return self.conduct_round(self.topic, self.round_count)
    
    def get_debate_status(self) -> Dict:
        """현재 토론 상태 반환"""
        return {
            'topic': self.topic,
            'current_round': self.round_count,
            'max_rounds': self.max_rounds,
            'total_statements': len(self.statements),
            'can_proceed': self.round_count < self.max_rounds
        }
    
    def summarize_debate(self):
        """토론 요약"""
        summary = self.get_debate_summary()
        return {
            'summary': f"{self.round_count}라운드의 토론이 완료되었습니다.",
            'statistics': summary
        }
    
    def conduct_round(self, topic: str, round_number: int) -> Dict:
        """한 라운드 토론 진행 (안전한 버전)"""
        results = {}
        
        try:
            # 진보 측 발언
            print(f"🔵 진보 측 발언 생성 중...")
            progressive_argument = self.progressive_agent.generate_argument(
                topic, round_number, self.debate_history
            )
            
            if progressive_argument:
                prog_statement = {
                    'round': round_number,
                    'stance': '진보',
                    'statement': progressive_argument,
                    'timestamp': self._get_timestamp()
                }
                self.debate_history.append(prog_statement)
                self.statements.append(prog_statement)
                results['progressive'] = prog_statement
                print(f"🔵 진보: {progressive_argument[:100]}...")
        except Exception as e:
            print(f"진보 측 발언 생성 실패: {e}")
            results['progressive'] = {
                'round': round_number,
                'stance': '진보',
                'statement': '죄송합니다. 발언 생성에 실패했습니다.',
                'timestamp': self._get_timestamp()
            }
        
        try:
            # 보수 측 발언
            print(f"🔴 보수 측 발언 생성 중...")
            conservative_argument = self.conservative_agent.generate_argument(
                topic, round_number, self.debate_history
            )
            
            if conservative_argument:
                cons_statement = {
                    'round': round_number,
                    'stance': '보수',
                    'statement': conservative_argument,
                    'timestamp': self._get_timestamp()
                }
                self.debate_history.append(cons_statement)
                self.statements.append(cons_statement)
                results['conservative'] = cons_statement
                print(f"🔴 보수: {conservative_argument[:100]}...")
        except Exception as e:
            print(f"보수 측 발언 생성 실패: {e}")
            results['conservative'] = {
                'round': round_number,
                'stance': '보수',
                'statement': '죄송합니다. 발언 생성에 실패했습니다.',
                'timestamp': self._get_timestamp()
            }
        
        return results
    
    def get_debate_summary(self) -> Dict:
        """토론 요약 정보 반환"""
        try:
            progressive_count = len([s for s in self.debate_history if s.get('stance') == '진보'])
            conservative_count = len([s for s in self.debate_history if s.get('stance') == '보수'])
            
            return {
                'total_rounds': self.round_count,
                'progressive_statements': progressive_count,
                'conservative_statements': conservative_count,
                'used_evidence_count': len(self.shared_tracker.used_evidence),
                'used_sources': list(self.shared_tracker.evidence_sources),
                'keyword_usage': self.shared_tracker.keyword_usage,
                'history': self.debate_history
            }
        except Exception as e:
            print(f"토론 요약 생성 실패: {e}")
            return {
                'total_rounds': self.round_count,
                'error': str(e)
            }
    
    def reset_debate(self):
        """토론 초기화"""
        try:
            self.debate_history = []
            self.statements = []
            self.round_count = 0
            self.topic = ""
            
            self.shared_tracker = ArgumentTracker()
            self.progressive_agent.set_shared_tracker(self.shared_tracker)
            self.conservative_agent.set_shared_tracker(self.shared_tracker)
            
            # 각 에이전트의 개별 추적기도 초기화
            self.progressive_agent.argument_tracker = ArgumentTracker()
            self.conservative_agent.argument_tracker = ArgumentTracker()
            
            # 발언 기록도 초기화
            self.progressive_agent.my_previous_statements = []
            self.progressive_agent.opponent_previous_statements = []
            self.conservative_agent.my_previous_statements = []
            self.conservative_agent.opponent_previous_statements = []
            
            print("✅ 토론 초기화 완료")
        except Exception as e:
            print(f"토론 초기화 실패: {e}")
    
    def _get_timestamp(self):
        """현재 시간 반환"""
        try:
            import datetime
            return datetime.datetime.now().isoformat()
        except Exception:
            return "timestamp_error"

# 사용 예시 및 호환성 함수
def create_debate_system(model_path: str, rag_system: RAGSystem = None) -> DebateManager:
    """토론 시스템 생성 함수"""
    try:
        progressive_agent = ProgressiveAgent(model_path, rag_system)
        conservative_agent = ConservativeAgent(model_path, rag_system)
        
        return DebateManager(progressive_agent, conservative_agent)
    except Exception as e:
        print(f"토론 시스템 생성 실패: {e}")
        raise

# main.py와의 호환성을 위한 기존 인터페이스
class DebateManagerLegacy:
    """기존 main.py와 호환되는 DebateManager"""
    
    def __init__(self, model_path: str):
        try:
            # RAG 시스템 없이 초기화
            self.progressive_agent = ProgressiveAgent(model_path, None)
            self.conservative_agent = ConservativeAgent(model_path, None)
            self.debate_manager = DebateManager(self.progressive_agent, self.conservative_agent)
            
            # 기존 인터페이스를 위한 속성들
            self.max_rounds = 3
            self.round_count = 0
            self.statements = []
            
        except Exception as e:
            print(f"Legacy DebateManager 초기화 실패: {e}")
            raise
    
    def start_debate(self, topic: str):
        """토론 시작 (기존 인터페이스)"""
        return self.debate_manager.start_debate(topic)
    
    def proceed_round(self):
        """라운드 진행 (기존 인터페이스)"""
        result = self.debate_manager.proceed_round()
        
        # 기존 인터페이스를 위한 속성 업데이트
        self.round_count = self.debate_manager.round_count
        self.statements = self.debate_manager.statements
        
        return result
    
    def get_debate_status(self):
        """토론 상태 (기존 인터페이스)"""
        status = self.debate_manager.get_debate_status()
        status['can_proceed'] = self.round_count < self.max_rounds
        return status
    
    def summarize_debate(self):
        """토론 요약 (기존 인터페이스)"""
        return self.debate_manager.summarize_debate()

# main.py에서 사용할 수 있는 호환성 클래스
DebateManager = DebateManagerLegacy  # 기존 main.py와 호환