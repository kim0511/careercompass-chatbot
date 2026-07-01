# 진로나침반 (CareerCompass Chatbot)

> 동아대학교 경영정보학과 취준생을 위한 공공기관·공기업 채용공고 기반 Q&A 챗봇

## 소개

공공기관·공기업 채용공고 데이터(72,907건)를 Neo4j 그래프 DB에 저장하고, GPT-4o와 RAG(검색 증강 생성) 기술을 활용해 취업 관련 질문에 답변하는 챗봇 서비스입니다.

**주요 기능**
- 채용공고 기반 Q&A (어학 커트라인, NCS 직무, 지원 자격 등)
- 경영정보학과 커리큘럼 가이드 (취업 경로, 필요 역량 안내)
- 질문 유형에 따라 Cypher 구조 검색 / RAG 벡터 검색 자동 분기

## 기술 스택

| 분류 | 기술 |
|------|------|
| Backend | Python, Flask |
| AI | GPT-4o, RAG (text-embedding-3-small) |
| Database | Neo4j (그래프 DB) |
| Infra | Docker, Docker Compose |

## 시스템 구조

```
사용자 질문
    │
    ├── [경로 A] 채용공고로 답할 수 있는 질문
    │       ├── Cypher: 기관/직무/고용형태/근무지 구조 검색
    │       └── RAG: 공고 본문에서 자격증·요건·직무내용 검색
    │
    └── [경로 B] 채용공고로 답 못하는 질문
            └── RAG: 취업 커뮤니티 Q&A 게시글 검색
```

## 실행 방법

### 1. 환경변수 설정
```bash
cp .env.example .env
# .env 파일에 API 키 및 DB 접속 정보 입력
```

### 2. Docker로 실행
```bash
docker-compose up -d
```

### 3. 로컬 실행
```bash
pip install flask neo4j openai
python server.py
```

접속: `http://localhost:5000`

## 화면 구성

| 페이지 | 경로 | 설명 |
|--------|------|------|
| 홈 | `/` | 서비스 소개 및 기능 선택 |
| 채용 챗봇 | `/chatbot` | 채용공고 기반 Q&A |
| 커리큘럼 가이드 | `/curriculum` | 학과 취업 경로 안내 |

## 질문 예시

- 한전 어학 커트라인이 얼마예요?
- 코레일 NCS 직무가 뭔가요?
- 경영정보학과 나오면 어디 취업해요?
- 데이터 직무는 어떻게 준비하나요?
