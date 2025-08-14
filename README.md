# AI 정치 토론 시스템
진보 vs 보수 관점의 AI 에이전트들이 정치 주제에 대해 토론하고 요약을 제공하는 시스템입니다.

## 📋 주요 기능

### 🎭 AI 에이전트 기반 토론
```
┌─────────────────┐    토론    ┌─────────────────┐
│  진보 에이전트   │ ←────────→ │  보수 에이전트   │
│   (Progressive)  │            │ (Conservative)  │
└─────────────────┘            └─────────────────┘
         ↑                              ↑
         └──────────── 🎯 사회자 ────────┘
                    (진행 및 요약)
└─ 요약 에이전트 (📄 최종 정리)
```

## 🚀 사용법

### 1. 기본 설치
```bash
pip install -r requirements.txt
```

### 2. 자동 모드 (전체 토론 실행)
```bash
python main.py --auto --topic "탄소세 도입에 대한 찬반 토론"
```

### 3. 대화형 모드 (단계별 제어)
```bash
python main.py --interactive
```

### 4. 고급 옵션
```bash
# 라운드 수 조정
python main.py --rounds 5 --topic "민생회복 소비쿠폰 도입에 대한 토론"
```

## 📊 시스템 구성

### 🤖 에이전트 구조
- **진보 에이전트**: 사회복지, 평등, 정부 개입 옹호
- **보수 에이전트**: 시장 자유, 개인 책임, 전통 가치 강조  
- **사회자 에이전트**: 중립적 토론 진행 및 마무리
- **요약 에이전트**: 토론 내용 종합 정리

### 💾 결과 저장
토론 결과는 `debate_results/` 폴더에 JSON 형태로 자동 저장됩니다.

```json
{
  "start_result": { "topic": "...", "moderator_intro": "..." },
  "round_results": [
    {
      "round": 1,
      "progressive_statement": "...",
      "conservative_statement": "..."
    }
  ],
  "summary_result": { "summary": "...", "moderator_conclusion": "..." }
}
```

## 🎮 대화형 명령어

### 기본 명령어
- `round` - 다음 라운드 진행 (진보 → 보수)
- `status` - 현재 토론 상태 확인
- `summary` - 토론 요약 및 종료
- `save` - 현재까지 결과 저장
- `quit` - 토론 종료

## 📁 프로젝트 구조

```
LLM-Debate/
├── main.py                 # 메인 실행 파일
├── debate_manager.py       # 토론 관리자
├── requirements.txt        # 의존성 패키지
│
├── agents/                 # AI 에이전트들
│   ├── __init__.py
│   ├── base_agent.py       # 기본 에이전트 클래스
│   ├── debate_agents.py    # 진보/보수 에이전트
│   ├── moderator_agent.py  # 사회자 에이전트
│   └── summary_agent.py    # 요약 에이전트
│
├── utils/                  # 유틸리티
│   ├── __init__.py
│   └── rag_system.py       # RAG 검색 시스템
│
├── data/                   # 참조 데이터
│   └── *.json             # 정치 관련 데이터
│
└── debate_results/         # 토론 결과 저장소
    └── *.json             # 토론 결과 파일들
```

## 🛠️ 기술 스택

- **언어 모델**: Hugging Face Transformers
- **토론 엔진**: 커스텀 멀티 에이전트 시스템
- **검색**: RAG (Retrieval-Augmented Generation)
- **데이터**: JSON 기반 정치 이슈 데이터베이스

## 📈 토론 예시

```
🎯 사회자: 최저임금 인상 정책에 대한 토론을 시작하겠습니다.

🔵 진보: 최저임금 인상은 저소득층의 생활 안정과 소득 불평등 해소에 필수적입니다...

🔴 보수: 최저임금 인상은 중소기업의 부담을 가중시키고 고용 감소를 야기할 수 있습니다...

📊 요약: 양측은 최저임금 정책의 필요성에는 공감하되, 그 수준과 시행 방식에서 차이를 보였습니다...
```

## ⚙️ 설정 옵션

### 기본 설정
- **기본 라운드**: 3라운드
- **기본 모델**: `EXAONE-4.0-32B-Q4_K_M.gguf`
- **언어**: 한국어

### 모델 변경
지원되는 한국어 모델들:
- `EXAONE-4.0-32B-Q4_K_M.gguf`
