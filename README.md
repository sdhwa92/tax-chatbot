# 소득세 챗봇

한국 소득세법에 관한 질문에 답변해주는 RAG 기반 AI 챗봇입니다.

## 기능

- 소득세법 관련 질문에 대한 정확한 답변 제공
- 관련 법조항(XX조)을 명시하여 근거 있는 답변
- Few-shot 예시를 활용한 일관된 답변 형식
- 대화 히스토리를 유지하는 멀티턴 대화
- 사용자 표현을 법률 용어로 자동 변환 (예: "사람" → "거주자")

## 기술 스택

| 분류 | 기술 |
|------|------|
| Frontend | Streamlit |
| LLM | OpenAI GPT-4o |
| Embedding | OpenAI text-embedding-3-large |
| Vector DB | Pinecone |
| Framework | LangChain 0.3.x |

## 아키텍처

```
사용자 질문
    ↓
Dictionary Chain (법률 용어 변환)
    ↓
History-Aware Retriever (대화 맥락 반영 검색)
    ↓
Pinecone Vector DB (소득세법 문서 검색)
    ↓
RAG Chain (문서 기반 답변 생성)
    ↓
GPT-4o 답변 스트리밍
```

## 설치 및 실행

### 1. 패키지 설치

```bash
pip install -r requirements.txt
```

### 2. 환경 변수 설정

`.env` 파일을 생성하고 아래 키를 입력합니다.

```
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

### 3. 실행

```bash
streamlit run chat.py
```

## 프로젝트 구조

```
.
├── chat.py          # Streamlit UI
├── llm.py           # LangChain RAG 체인 구성
├── config.py        # Few-shot 예시 데이터
├── requirements.txt # 패키지 의존성
└── .env             # 환경 변수 (git 미포함)
```

## 배포

[Streamlit Community Cloud](https://streamlit.io/cloud)를 통해 배포합니다. 배포 시 Streamlit Cloud의 Secrets 설정에서 환경 변수를 등록해야 합니다.
