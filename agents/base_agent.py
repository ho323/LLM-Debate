from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import os
import subprocess
import tempfile
from transformers import AutoTokenizer

class BaseAgent(ABC):
    def __init__(self, model_path: str = '/home/ho/Documents/금융ai/models/EXAONE-4.0-32B-Q4_K_M.gguf'):
        self.model_path = model_path
        self.tokenizer = None
        self.llama_cli_path = "/home/ho/Documents/금융ai/llama.cpp/build/bin/llama-cli"  # llama-cli 경로
        self._load_model()
    
    def _load_model(self):
        """토크나이저를 로드하고 모델 경로를 확인합니다."""
        print(f"EXAONE 모델 설정 중: {self.model_path}")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {self.model_path}")
        
        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained("LGAI-EXAONE/EXAONE-4.0-32B")
        
        print("EXAONE 모델 설정 완료")
    
    def generate_response(self, prompt: str, max_length: int = 512, target_length: str = "간결하게") -> str:
        """프롬프트에 대한 응답을 생성합니다."""
        try:
            # EXAONE 모델용 채팅 템플릿 적용
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            input_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            
            # 임시 파일에 입력 저장
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
                f.write(input_text)
                input_file = f.name
            
            try:
                # llama-cli로 모델 실행
                result = subprocess.run(
                    [
                        self.llama_cli_path,
                        "-m", self.model_path,
                        "-fa",
                        "-ngl", "65",  # GPU 레이어 수
                        "--temp", "0.8",
                        "--top-p", "0.9",
                        "--repeat-penalty", "1.1",
                        "-f", input_file,
                        "-n", str(max_length),  # 최대 토큰 수
                        "-no-cnv"
                    ],
                    capture_output=True,
                    text=True,
                    timeout=300  # 5분 타임아웃
                )
                
                if result.returncode != 0:
                    print(f"llama-cli 실행 오류: {result.stderr}")
                    return "응답을 생성할 수 없습니다."
                
                # 응답 추출
                output = result.stdout.strip()
                return self._extract_after_think(output)
                
            finally:
                # 임시 파일 삭제
                if os.path.exists(input_file):
                    os.unlink(input_file)
            
        except Exception as e:
            print(f"텍스트 생성 중 오류 발생: {e}")
            return "오류가 발생했습니다."
    
    def _extract_after_think(self, output: str) -> str:
        """'</think>' 이후의 문자열만 반환합니다. 태그가 없으면 원문을 반환합니다."""
        if not output:
            return output
        lower_out = output.lower()
        marker = "</think>"
        idx = lower_out.rfind(marker)
        if idx != -1:
            return output[idx + len(marker):].strip()
        return output.strip()
    
    def _clean_response(self, response: str) -> str:
        """응답을 정리합니다 (반복 제거, 불완전 문장 처리)."""
        if not response:
            return response
            
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
        
        return '. '.join(unique_sentences) + ('.' if unique_sentences else '')
    
    @abstractmethod
    def process_input(self, input_data: Dict) -> str:
        """각 에이전트별 입력 처리 로직"""
        pass 