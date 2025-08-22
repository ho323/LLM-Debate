from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import os
import subprocess
import tempfile

# transformers는 선택적으로 사용
try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ transformers 없음 - 기본 템플릿 사용")

class BaseAgent(ABC):
    def __init__(self, model_path: str = 'C:/Users/User/Documents/EXAONE-4.0-32B-Q4_K_M.gguf'):
        self.model_path = model_path
        self.tokenizer = None
        self.llama_cli_path = "C:/Users/User/LLM-Debate/llama.cpp/build/bin/Release/llama-cli.exe"
        print(f"🔧 BaseAgent 초기화 - 32B 모델 최적화 버전")
        print(f"⏰ 응답 생성 시간: 무제한 (완료될 때까지 대기)")
        self._load_model()
    
    def _load_model(self):
        """토크나이저를 로드하고 모델 경로를 확인합니다."""
        print(f"EXAONE 모델 설정 중: {self.model_path}")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {self.model_path}")
        
        # 토크나이저 로드 (선택적)
        if TRANSFORMERS_AVAILABLE:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained("LGAI-EXAONE/EXAONE-4.0-32B")
                print("✅ EXAONE 토크나이저 로드 성공")
            except Exception as e:
                print(f"⚠️ 토크나이저 로드 실패: {e} - 기본 템플릿 사용")
                self.tokenizer = None
        
        print("EXAONE 모델 설정 완료")
    
    def generate_response(self, prompt: str, max_length: int = 1000, target_length: str = "간결하게") -> str:
        """프롬프트에 대한 응답을 생성합니다."""
        print(f"🔄 32B 모델 응답 생성 시작... (완료될 때까지 대기)")
        
        try:
            # 오류 방지용 템플릿
            if self.tokenizer:
                try:
                    messages = [{"role": "user", "content": prompt}]
                    input_text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                except Exception as e:
                    print(f"⚠️ 토크나이저 템플릿 오류: {e} - 기본 템플릿 사용")
                    input_text = f"User: {prompt}\nAssistant:"
            else:
                # 기본 템플릿 사용
                input_text = f"User: {prompt}\nAssistant:"
            
            # 임시 파일에 입력 저장 (UTF-8 인코딩 명시)
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
                f.write(input_text)
                input_file = f.name
            
            try:
                # 32B 모델용 최적화된 llama-cli 실행 (타임아웃 없음)
                result = subprocess.run(
                    [
                        self.llama_cli_path,
                        "-m", self.model_path,
                        "-f", input_file,
                        "-n", str(max_length),  # 토큰 수 적당히 제한
                        "-c", "2048",                     # 컨텍스트 크기 설정
                        "--temp", "0.7",
                        "--top-p", "0.9",
                        "--repeat-penalty", "1.1",
                        "-no-cnv",
                        "--seed", "42",
                        "-t", "4"                         # CPU 스레드 수 지정
                    ],
                    capture_output=True,
                    text=True,
                    # timeout 제거 - 무제한 대기
                    encoding='utf-8',     # UTF-8 인코딩 명시
                    errors='ignore',      # 인코딩 오류 무시
                    # Windows에서 창 숨기기 및 인코딩 문제 방지
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
                )
                
                if result.returncode != 0:
                    error_msg = "실행 오류"
                    if result.stderr:
                        try:
                            error_msg = result.stderr[:100]  # 오류 메시지 길이 제한
                        except:
                            error_msg = "인코딩 오류로 읽을 수 없음"
                    
                    print(f"llama-cli 실행 오류: {error_msg}")
                    return "응답을 생성할 수 없습니다."
                
                # 응답 추출 (안전하게)
                output = ""
                if result.stdout:
                    try:
                        output = result.stdout.strip()
                        print(f"✅ 응답 생성 완료: {len(output)}자")
                    except Exception as e:
                        print(f"⚠️ 출력 읽기 오류: {e}")
                        return "출력 읽기 실패"
                
                if output:
                    return self._extract_after_think(output)
                else:
                    return "빈 응답이 반환되었습니다."
                
            except Exception as e:
                print(f"subprocess 오류: {e}")
                return "실행 중 오류가 발생했습니다."
            finally:
                # 임시 파일 삭제
                try:
                    if os.path.exists(input_file):
                        os.unlink(input_file)
                except:
                    pass  # 삭제 실패해도 계속
            
        except Exception as e:
            print(f"텍스트 생성 중 오류 발생: {e}")
            return "오류가 발생했습니다."
    
    def _extract_after_think(self, output: str) -> str:
        """'</think>' 이후의 문자열만 반환합니다. 태그가 없으면 원문을 반환합니다."""
        if not output:
            return "응답이 없습니다."
        
        try:
            lower_out = output.lower()
            marker = "</think>"
            idx = lower_out.rfind(marker)
            if idx != -1:
                result = output[idx + len(marker):].strip()
            else:
                result = output.strip()
            
            # [end of text] 토큰 제거
            result = result.replace("[end of text]", "").replace("[END OF TEXT]", "").strip()
            
        
            if "User:" in result:
                result = result.split("User:")[0].strip()
            
            return result if result else "정리된 응답이 없습니다."
            
        except Exception as e:
            print(f"⚠️ 응답 추출 오류: {e}")
            return "응답 추출 실패"
    
    def _clean_response(self, response: str) -> str:
        """응답을 정리합니다 (반복 제거, 불완전 문장 처리)."""
        if not response:
            return "응답을 정리할 수 없습니다."
        
        try:
            # 중복 문장 제거
            sentences = response.split('.')
            unique_sentences = []
            seen = set()
            
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence and sentence not in seen and len(sentence) > 10:
                    seen.add(sentence)
                    unique_sentences.append(sentence)
            
            # 마지막 문장이 불완전하면 제거
            if unique_sentences and len(unique_sentences[-1]) < 20:
                unique_sentences = unique_sentences[:-1]
            
            result = '. '.join(unique_sentences) + ('.' if unique_sentences else '')
            return result if result else "정리된 응답이 없습니다."
            
        except Exception as e:
            print(f"⚠️ 응답 정리 오류: {e}")
            return response  
    
    @abstractmethod
    def process_input(self, input_data: Dict) -> str:
        """각 에이전트별 입력 처리 로직"""
        pass