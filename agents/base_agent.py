from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class BaseAgent(ABC):
    def __init__(self, model_name: str = 'Bllossom/llama-3.2-Korean-Bllossom-3B'):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """모델과 토크나이저를 로드합니다."""
        print(f"모델 로딩 중: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            trust_remote_code=True
        )
        
        # 패딩 토큰 설정
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        print("모델 로딩 완료")
    
    def generate_response(self, prompt: str, max_length: int = 512, target_length: str = "간결하게") -> str:
        """프롬프트에 대한 응답을 생성합니다 (반복 방지 강화)."""
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                # top_k=50,
                repetition_penalty=1.1,
                # no_repeat_ngram_size=3,
                # early_stopping=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # 입력 길이만큼 제거하고 새로 생성된 토큰만 디코딩
        input_length = inputs['input_ids'].shape[1]
        response_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
        
        # 불완전한 문장 정리 및 반복 제거
        # response = self._clean_response(response)
        
        return response.strip()
    
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