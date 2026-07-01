"""
취준생 Q&A 파이프라인 v2 - 채용공고 중심
=================================================
사용자 질문
    │
    ├── [경로 A] 채용공고로 답할 수 있는 질문
    │       ├── Cypher: 기관/직무/고용형태/근무지 구조 검색
    │       └── RAG on Chunk: 공고 본문에서 자격증·요건·직무내용 검색
    │
    └── [경로 B] 채용공고로 답 못하는 질문
            └── RAG on QAPost: 비슷한 경험 게시글 + 채택된 답변

두 경로 결과를 GPT-4o가 합쳐 자연어 답변 생성
"""

import sys, io, json, re, os
from neo4j import GraphDatabase
from openai import OpenAI

# CLI 실행 시에만 stdout UTF-8 강제 적용 (Streamlit import 시엔 건드리지 않음)
if hasattr(sys.stdout, 'buffer'):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

# ── 설정 ──────────────────────────────────────────────────────────
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
NEO4J_URI      = os.environ.get("NEO4J_URI",  "neo4j://127.0.0.1:7687")
NEO4J_USER     = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]
NEO4J_AUTH     = (NEO4J_USER, NEO4J_PASSWORD)
NEO4J_DB       = os.environ.get("NEO4J_DB",   "neo4j")
CHAT_MODEL     = os.environ.get("CHAT_MODEL",  "gpt-4o")
EMBED_MODEL    = os.environ.get("EMBED_MODEL", "text-embedding-3-small")

client = OpenAI(api_key=OPENAI_API_KEY)
driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)


# ── 벡터 인덱스 상태 (앱 시작 시 1회만 확인) ─────────────────────
def _check_vector_indexes() -> dict:
    result = {"chunk": False, "qapost": False}
    try:
        with driver.session(database=NEO4J_DB) as s:
            # chunk_embedding 인덱스
            idx = s.run("SHOW INDEXES WHERE name = 'chunk_embedding'").single()
            if idx and idx["state"] == "ONLINE":
                dim = s.run(
                    "MATCH (c:Chunk) WHERE c.embedding IS NOT NULL "
                    "RETURN size(c.embedding) AS dim LIMIT 1"
                ).single()
                result["chunk"] = bool(dim and dim["dim"] == 1536)

            # qapost_embedding 인덱스
            idx2 = s.run("SHOW INDEXES WHERE name = 'qapost_embedding'").single()
            if idx2 and idx2["state"] == "ONLINE":
                result["qapost"] = True
    except Exception:
        pass
    return result

_INDEX = _check_vector_indexes()


# ── 스키마 ────────────────────────────────────────────────────────
SCHEMA = """
Neo4j 그래프 데이터베이스 스키마:

[채용공고 데이터]
- Organization: 공공기관 (속성: 기관명)
- JobPosting: 채용공고 (속성: 채용제목, 기관명, 고용형태, 근무지,
    등록일(공고 게시일 예:"2023.04.07"), 마감일(지원 마감일 예:"23.04.24"), 상태(예:"마감"/"접수중"),
    detail_url,
    req_english_tests(어학시험 종류 목록), req_toeic_min(토익 최소점수), req_opic_min(오픽 최소등급),
    req_toeic_sp_min(토익스피킹 최소점수), req_teps_min(텝스 최소점수),
    req_education(학력요건: 무관/고졸/초대졸/대졸/석사/박사),
    req_career(경력요건: 신입/경력/신입경력/무관), req_career_years(최소경력연수),
    req_certifications(요구 자격증 목록 예: ["정보처리기사","CPA"]),
    req_major(전공요건: 무관/이공계/상경계/법학/인문/사범/예체능),
    req_preferred(우대사항 목록 예: ["장애인","보훈","지역인재"]),
    req_tech_stack(기술스택 목록 예: ["Python","SQL","GIS"]),
    req_headcount(채용인원 숫자), req_age_limit(연령상한 만나이), req_age_min(연령하한 만나이),
    ncs_se_list(공고 본문에서 추출한 NCS 직무 분류 목록 예: ["01.빅데이터분석","02.데이터엔지니어링"]),
    ncs_so_list(소분류 목록), ncs_jung_list(중분류 목록))
- NCS: 직무 분류 (속성: 대분류, 중분류, 소분류, 세분류)
- Chunk: 채용공고 본문 텍스트 조각 (속성: text, file_name - 자격증/요건/직무내용 포함)

[Q&A 데이터]
- QAPost: 취준생 질문 게시글 (속성: title, body, category, platform, views, is_public_related)
- Comment: 답변 (속성: body, author_job, career_year, is_accepted, likes)

[관계]
- (o:Organization)-[:HAS_POSTING]->(j:JobPosting)
- (j:JobPosting)-[:HAS_NCS]->(n:NCS)
- (j:JobPosting)-[:HAS_CHUNK]->(c:Chunk)
- (q:QAPost)-[:HAS_COMMENT]->(c:Comment)
- (q:QAPost)-[:MENTIONS_ORG]->(o:Organization)
- (q:QAPost)-[:RELATED_NCS]->(n:NCS)

[Cypher 예시]
특정 기관 채용공고: MATCH (o:Organization {기관명:"한국전력공사"})-[:HAS_POSTING]->(j:JobPosting)
NCS 직무별 채용 현황: MATCH (j:JobPosting)-[:HAS_NCS]->(n:NCS) WHERE n.세분류 CONTAINS "데이터"
고용형태별 현황: MATCH (j:JobPosting) WHERE j.고용형태 = "정규직"  -- 주의: CONTAINS '정규직' 금지 (비정규직도 매칭됨)
기술스택 기반 검색: MATCH (j:JobPosting) WHERE "Python" IN j.req_tech_stack RETURN j.채용제목, j.기관명, j.고용형태, j.근무지 LIMIT 10
자격증 기반 검색: MATCH (j:JobPosting) WHERE "정보처리기사" IN j.req_certifications RETURN j.채용제목, j.기관명, j.고용형태 LIMIT 10
복합 조건 검색: MATCH (j:JobPosting) WHERE j.req_career = "신입" AND j.req_major = "이공계" AND "SQL" IN j.req_tech_stack RETURN j.채용제목, j.기관명 LIMIT 10
우대사항 기반: MATCH (j:JobPosting) WHERE "지역인재" IN j.req_preferred RETURN j.채용제목, j.기관명, j.근무지 LIMIT 10
금융기관 검색: MATCH (o:Organization)-[:HAS_POSTING]->(j:JobPosting) WHERE o.기관명 CONTAINS "은행" OR o.기관명 CONTAINS "금융" OR o.기관명 CONTAINS "보험" OR o.기관명 CONTAINS "신용" OR o.기관명 CONTAINS "투자" RETURN j.채용제목, o.기관명 LIMIT 10
"""

# ── 프롬프트 ──────────────────────────────────────────────────────
ROUTE_SYSTEM = """사용자의 질문 유형을 아래 4가지 중 하나로 판단하세요.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[1] dept_career — 동아대 경영정보학과 학생 진로 질문
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- 동아대학교(동아대) 경영정보학과 학생이 진로·커리큘럼·준비 방법을 묻는 질문
  예: "경영정보학과 1학년인데 뭐부터 시작해야 해?"
  예: "경영정보학과 나오면 어디 취업돼?"
  예: "데이터사이언스 모듈이랑 디지털비즈니스 모듈 중 뭐가 좋아?"
  예: "경영정보학과 학생이 따야 할 자격증은?"
  예: "우리 학과 졸업하고 한국해양진흥공사 가려면 어떤 직무로 지원하면 좋을까?"
  예: "우리 과 졸업하고 [특정 기관] 들어가려면 어떤 직무가 맞을까?"
  예: "경영정보학과 출신이 [특정 기관]에서 지원하기 좋은 직무는?"
  ※ "우리 학과/우리 과" + 특정 기관 + 직무 추천 조합은 반드시 dept_career로 분류
→ 학과 커리큘럼·진출 분야 데이터로 답변함

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[2] ncs_prep — 특정 기관 NCS 준비 질문
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- 특정 기관을 언급하며 NCS 준비법·시험과목·직무를 묻는 질문
  예: "한국수력원자력 NCS 어떻게 준비해?"
  예: "한전 NCS 시험 뭐 봐?"
  예: "LH공사 입사하려면 뭐 준비해야 해?"
- 반드시 org_name(기관명 키워드)을 함께 반환할 것
→ 해당 기관 NCS 직무 데이터로 답변함

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[3] posting — 채용 조건·자격 요건 질문
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- 채용공고에서 직접 확인할 수 있는 객관적 조건 질문
  예: "코레일 운전직 자격 조건은?"
  예: "공기업 IT직무 토익 커트라인은?"
  예: "행정직 우대 자격증 알려줘"
  예: "캠코 채용 절차 어떻게 돼?"
  예: "내 스펙으로 지원할 수 있는 공기업은?"
→ 채용공고 DB 검색 → 결과 없으면 Q&A 게시글로 자동 폴백

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[4] experience — 면접·자소서·취업 고민 질문
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- 채용공고가 아닌 실제 경험에서 답을 얻어야 하는 질문
  예: "면접에서 어떻게 답해야 해?"
  예: "자소서 지원동기 어떻게 써?"
  예: "공백기가 있는데 불리할까?"
  예: "최종 탈락 후 멘탈 관리 어떻게 해?"
  예: "공기업 준비 기간 보통 얼마나 걸려?"
  예: "취업 고민인데 방향을 모르겠어"
→ 취업 Q&A 커뮤니티 게시글·댓글로 답변함 (채용공고 검색 생략)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
판단 기준
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 채용공고에 명시된 조건(자격증·어학·전공·경력·우대사항)을 묻는다 → posting
• 실제 경험·감각·조언이 필요한 주관적 질문이다 → experience
• 모호하면 posting 우선 (posting은 결과 없을 때 자동으로 게시글을 찾아줌)

JSON으로만 답하세요:
{"route": "posting" | "experience" | "ncs_prep" | "dept_career", "reason": "한 줄 이유", "org_name": "기관명 키워드 (ncs_prep일 때만, 나머지는 빈 문자열)"}"""

CYPHER_SYSTEM = f"""Neo4j Cypher 전문가입니다. 채용공고 관련 질문에 답할 Cypher를 생성하세요.

{SCHEMA}

규칙:
1. Cypher만 반환. ```cypher ... ``` 블록 사용.
2. LIMIT 20 이하. (Reranker가 상위 8건을 추려내므로 20건 후보 수집 권장)
3. 텍스트 검색은 CONTAINS 사용.
4-0. [필수 — OR/AND 우선순위] WHERE 절에서 OR 조건이 여러 개일 때 반드시 괄호로 묶을 것.
   괄호 없이 쓰면 AND가 OR보다 먼저 적용되어 필터가 깨짐.
   ❌ 잘못된 예: WHERE a CONTAINS 'x' OR a CONTAINS 'y' AND b = 'z'
   ✅ 올바른 예: WHERE (a CONTAINS 'x' OR a CONTAINS 'y') AND b = 'z'
4. RETURN에는 반드시 아래 필드를 포함할 것:
   j.채용제목, j.기관명, j.고용형태, j.근무지,
   j.등록일, j.마감일, j.상태,
   j.req_certifications, j.req_preferred, j.req_major, j.req_toeic_min, j.req_career, j.req_english_tests
   NCS 관련 질문이면 n.대분류, n.중분류, n.소분류, n.세분류 도 포함.
   집계 질문(예: "요구하는게 뭐야", "공통 자격은")에서도 개별 공고의 req_* 필드를 반환해야
   답변 GPT가 집계·분석할 수 있음. AVG/COUNT만 반환하지 말 것.
5. 기관명 검색 시 키워드 하나로 좁히지 말 것. 예: "금융 공기관" → 은행·금융·보험·예금·신용·투자 등 연관 단어를 OR 조건으로 사용.
6. 결과가 적을 것 같으면 기관명 대신 NCS 직무 분류(n.대분류, n.중분류, n.소분류)로 검색하는 방안도 고려.
7. [핵심] 사용자가 본인의 스펙(자격증, 어학점수, 전공 등)을 제시하며 지원 가능한 공고를 물으면:
   - 각 스펙 조건을 OR로 연결해서, 하나라도 해당되는 공고를 모두 검색
   - 자격증: req_certifications(필수) 또는 req_preferred(우대)에 포함되면 매칭
   - 어학: req_toeic_min이 사용자 점수 이하이면 매칭 (예: req_toeic_min <= 700)
           단, req_toeic_min IS NULL인 공고는 어학 조건으로 매칭하지 말 것 (토익 요구가 없는 것이므로)
   - 전공: req_major가 사용자 전공 계열과 일치하면 매칭
   - 경력: 사용자가 '신입'이라고 명시한 경우에만 req_career IN ["신입","신입경력","무관"] 필터 적용.
           조건에 career가 없거나 null이면 경력 필터를 절대 추가하지 말 것.
   - RETURN에 req_certifications, req_preferred, req_major, req_toeic_min, req_career 반드시 포함
   예시 (토익700, 컴활, 경영학과, 신입):
   MATCH (j:JobPosting)
   WHERE j.req_career IN ["신입","신입경력","무관"]
   AND (
     "컴퓨터활용능력" IN j.req_certifications
     OR "컴퓨터활용능력" IN j.req_preferred
     OR j.req_major IN ["상경계","무관"]
     OR (j.req_toeic_min IS NOT NULL AND j.req_toeic_min <= 700)
   )
   WITH j.기관명 AS 기관명, collect(j) AS postings
   WITH 기관명, postings[0..2] AS top2
   UNWIND top2 AS j
   RETURN j.채용제목, 기관명, j.고용형태, j.근무지,
          j.req_certifications, j.req_preferred, j.req_major, j.req_toeic_min, j.req_career
   ORDER BY CASE WHEN j.고용형태 = '정규직' THEN 0 ELSE 1 END
   LIMIT 10
8. 사용자 스펙 기반 공고 검색 시 기관 다양성을 확보할 것.
   같은 기관 공고가 여러 개 매칭되더라도 기관당 최대 2개까지만 포함.
   반드시 다음 패턴 사용:
   WITH j.기관명 AS 기관명, collect(j) AS postings
   WITH 기관명, postings[0..2] AS top2
   UNWIND top2 AS j
9. [정규직 필터/우선] 고용형태 처리 — 매우 중요:
   ⚠️ CONTAINS '정규직' 절대 사용 금지! "비정규직"도 '정규직'을 포함하므로 오염됨.
   - 정규직만 필터: j.고용형태 = '정규직'
   - 정규직 우선 정렬: ORDER BY CASE WHEN j.고용형태 = '정규직' THEN 0 ELSE 1 END
   실제 고용형태 값(DB): '정규직', '비정규직', '무기계약직', '청년인턴(체험형)', '청년인턴(채용형)'
   기관 다양성 패턴(규칙 8)과 함께 쓸 때는 UNWIND 이후 RETURN 바로 앞에 ORDER BY 삽입.

9-1. [기관 특정 + 직무 검색 — 정규직·신입 우선 필수] "○○기관 IT 직무", "○○기관 금융 직무" 등
   특정 기관의 직무를 검색할 때 직무 키워드(IT, 전산 등)를 WHERE 필터로 쓰면
   공공기관 신입 공채("신입직원 채용", "직원 채용" 등 제목)가 통째로 빠짐.
   공공기관 신입 공채는 여러 직무(IT·경영·금융 등)를 하나의 공고로 묶어 모집하기 때문.

   ❌ 잘못된 방식 — 직무 키워드 WHERE 필터:
   WHERE j.기관명 CONTAINS '주택금융' AND j.채용제목 CONTAINS 'IT'
   → 신입 공채 누락됨

   ✅ 올바른 방식 — 기관 필터만 WHERE에, 직무 관련성은 ORDER BY로 우선순위 조정:
   MATCH (j:JobPosting)
   WHERE j.기관명 CONTAINS '주택금융'
   RETURN j.채용제목, j.기관명, j.고용형태, j.근무지, j.등록일, j.마감일, j.상태,
          j.req_certifications, j.req_preferred, j.req_major, j.req_toeic_min, j.req_career, j.req_english_tests
   ORDER BY
     CASE WHEN j.고용형태 = '정규직' THEN 0 ELSE 1 END,
     CASE WHEN j.req_career IN ['신입','신입경력'] OR j.채용제목 CONTAINS '신입' THEN 0 ELSE 1 END,
     CASE WHEN j.채용제목 CONTAINS 'IT' OR j.채용제목 CONTAINS '전산' OR j.채용제목 CONTAINS '정보' THEN 0 ELSE 1 END
   LIMIT 10
10. [스펙 중요도 질문] 사용자가 특정 스펙/경험이 얼마나 요구되는지 묻는 경우, 실제 공고 수를 집계해서 반환할 것.
    예: "인턴 경험을 우대하는 공고가 얼마나 있나?" →
    MATCH (j:JobPosting)
    WHERE ANY(p IN j.req_preferred WHERE p CONTAINS '인턴')
    RETURN count(j) AS 인턴우대공고수
    예: "인턴 경험 우대 공고 샘플" →
    MATCH (j:JobPosting)
    WHERE ANY(p IN j.req_preferred WHERE p CONTAINS '인턴')
    RETURN j.채용제목, j.기관명, j.고용형태, j.req_preferred
    LIMIT 5
11. [집계 질문] "평균 몇 점", "몇 개나 있어" 같은 질문에는 AVG, COUNT를 사용한 집계 Cypher를 생성할 것.
    예: RETURN avg(j.req_toeic_min) AS 평균토익점수, count(j) AS 집계공고수
13. [어학 조건 없는 공고 검색] "토익 없어도 되는 공고", "어학 조건 없는 공고" 같은 질문은
    req_toeic_min IS NULL AND (req_english_tests IS NULL OR req_english_tests = []) 조건으로 검색.
    예시 (토익 없는 정규직 — 기관명 OR 조건은 반드시 괄호로 묶을 것):
    MATCH (o:Organization)-[:HAS_POSTING]->(j:JobPosting)
    WHERE (o.기관명 CONTAINS '공사' OR o.기관명 CONTAINS '공단' OR o.기관명 CONTAINS '공기업')
    AND j.고용형태 = '정규직'
    AND j.req_toeic_min IS NULL
    AND (j.req_english_tests IS NULL OR size(j.req_english_tests) = 0)
    RETURN j.채용제목, o.기관명, j.고용형태, j.근무지, j.req_career, j.req_certifications, j.req_toeic_min
    ORDER BY CASE WHEN j.고용형태 = '정규직' THEN 0 ELSE 1 END
    LIMIT 10

14. [특수 전문직 기본 제외 — 중요] 사용자가 특정 전문 직종(의료, 법조, 회계, 금융전문직 등)을 명시하지 않은
    일반적인 채용공고 검색 시, 아래 전문 자격이 필수(req_certifications)인 공고는 반드시 제외할 것.
    일반 신입/정규직 검색 시 다음 조건을 반드시 추가:
    AND NOT ANY(c IN j.req_certifications WHERE
        c CONTAINS '간호사' OR c CONTAINS '의사' OR c CONTAINS '약사' OR
        c CONTAINS '물리치료사' OR c CONTAINS '방사선사' OR c CONTAINS '임상병리사' OR
        c CONTAINS '치과' OR c CONTAINS '한의사' OR c CONTAINS '수의사' OR
        c CONTAINS '공인회계사' OR c CONTAINS 'CPA' OR c CONTAINS '세무사' OR
        c CONTAINS '변호사' OR c CONTAINS '법무사' OR c CONTAINS '감정평가사' OR
        c CONTAINS '노무사' OR c CONTAINS '관세사')
    [우대사항(req_preferred)에 있는 건 제외 대상이 아님 — 필수(req_certifications)만 제외]

15. [직무 필터링 — 매우 중요] 사용자가 "IT 공고", "금융 직무", "행정 직무" 등 특정 직무를 언급하면
    반드시 채용제목 또는 NCS로 직무를 필터링할 것.
    절대 req_major(전공)만으로 직무를 판단하지 말 것 — req_major="이공계"는 IT가 아니라 의료·간호·생명과학도 포함됨.

    IT 직무 필터 예시 (채용제목 + NCS 동시 활용):
    MATCH (j:JobPosting)
    WHERE (j.채용제목 CONTAINS 'IT' OR j.채용제목 CONTAINS '전산' OR j.채용제목 CONTAINS '정보'
           OR j.채용제목 CONTAINS '소프트웨어' OR j.채용제목 CONTAINS '개발' OR j.채용제목 CONTAINS '시스템')
    또는:
    MATCH (j:JobPosting)-[:HAS_NCS]->(n:NCS)
    WHERE n.대분류 CONTAINS '정보' OR n.중분류 CONTAINS '정보통신' OR n.소분류 CONTAINS 'IT'

    지역 + 직무 + 경력 복합 예시 (서울/부산 신입 IT):
    MATCH (j:JobPosting)
    WHERE (j.근무지 CONTAINS '서울' OR j.근무지 CONTAINS '부산')
    AND j.req_career IN ['신입', '신입경력', '무관']
    AND (j.채용제목 CONTAINS 'IT' OR j.채용제목 CONTAINS '전산' OR j.채용제목 CONTAINS '정보'
         OR j.채용제목 CONTAINS '소프트웨어' OR j.채용제목 CONTAINS '개발')
    RETURN j.채용제목, j.기관명, j.고용형태, j.근무지, j.req_career, j.req_certifications
    ORDER BY CASE WHEN j.고용형태 CONTAINS '정규직' THEN 0 ELSE 1 END
    LIMIT 10

    IT 직무 + 어학 조건 예시:
    MATCH (j:JobPosting)-[:HAS_NCS]->(n:NCS)
    WHERE (n.대분류 CONTAINS '정보' OR n.중분류 CONTAINS '정보통신'
           OR j.채용제목 CONTAINS 'IT' OR j.채용제목 CONTAINS '전산' OR j.채용제목 CONTAINS '정보')
    AND j.req_toeic_min IS NOT NULL
    RETURN j.채용제목, j.기관명, j.req_toeic_min, j.req_opic_min, j.req_english_tests
    ORDER BY j.req_toeic_min DESC
    LIMIT 10
    집계가 필요하면 AVG/COUNT 사용:
    RETURN avg(j.req_toeic_min) AS 평균토익, count(j) AS 공고수

16. [학과 모듈 관련 공고 — 매우 중요] "DS모듈", "데이터사이언스모듈", "DB모듈", "디지털비즈니스모듈" 등
    학과 모듈 이름으로 관련 공고를 묻는 경우, 다음 두 가지를 OR로 결합할 것:
    ① j.ncs_se_list: JobPosting에 추출된 NCS 직무 분류 목록 (이 필드가 있으면 더 정확)
    ② 채용제목 키워드 (j.ncs_se_list가 없는 공고 커버)
    ※ 'AI' 단독 키워드는 조류인플루엔자(Avian Influenza)와 혼동되므로 사용 금지.

    DS모듈(데이터사이언스) 관련 NCS 세분류 키워드:
      ncs_se_list: '빅데이터', '데이터분석', '데이터엔지니어', '인공지능', '기계학습', '통계분석', 'AI', '데이터마이닝'
      채용제목(보조): '데이터', '인공지능', '빅데이터', '머신러닝', '딥러닝', '분석', '통계', 'DATA'

    DS모듈 관련 공고 예시 (ncs_se_list 우선):
    MATCH (j:JobPosting)
    WHERE ((j.ncs_se_list IS NOT NULL AND any(sf IN j.ncs_se_list WHERE sf CONTAINS '빅데이터' OR sf CONTAINS '데이터분석' OR sf CONTAINS '인공지능' OR sf CONTAINS '통계분석'))
        OR (j.채용제목 CONTAINS '데이터' OR j.채용제목 CONTAINS '인공지능'
            OR j.채용제목 CONTAINS '빅데이터' OR j.채용제목 CONTAINS '머신러닝'
            OR j.채용제목 CONTAINS '분석' OR j.채용제목 CONTAINS '통계'))
    AND NOT (j.채용제목 CONTAINS '조류' OR j.채용제목 CONTAINS '습지'
             OR j.채용제목 CONTAINS '철새' OR j.채용제목 CONTAINS '환경감시')
    RETURN j.채용제목, j.기관명, j.고용형태, j.근무지, j.등록일, j.마감일, j.상태,
           j.req_certifications, j.req_preferred, j.req_major, j.req_toeic_min, j.req_career, j.req_english_tests,
           j.ncs_se_list
    ORDER BY
      CASE WHEN j.ncs_se_list IS NOT NULL AND any(sf IN j.ncs_se_list WHERE sf CONTAINS '빅데이터' OR sf CONTAINS '데이터') THEN 0 ELSE 1 END,
      CASE WHEN j.고용형태 = '정규직' THEN 0 ELSE 1 END
    LIMIT 20

    DS모듈 + 지역 필터 예시 (부산):
    MATCH (j:JobPosting)
    WHERE ((j.ncs_se_list IS NOT NULL AND any(sf IN j.ncs_se_list WHERE sf CONTAINS '빅데이터' OR sf CONTAINS '데이터분석' OR sf CONTAINS '인공지능'))
        OR (j.채용제목 CONTAINS '데이터' OR j.채용제목 CONTAINS '인공지능' OR j.채용제목 CONTAINS '빅데이터' OR j.채용제목 CONTAINS '분석'))
    AND NOT (j.채용제목 CONTAINS '조류' OR j.채용제목 CONTAINS '습지' OR j.채용제목 CONTAINS '철새')
    AND j.근무지 CONTAINS '부산'
    RETURN j.채용제목, j.기관명, j.고용형태, j.근무지, j.등록일, j.마감일, j.상태,
           j.req_certifications, j.req_preferred, j.req_major, j.req_toeic_min, j.req_career, j.req_english_tests,
           j.ncs_se_list
    ORDER BY CASE WHEN j.고용형태 = '정규직' THEN 0 ELSE 1 END
    LIMIT 20

    DB모듈(디지털비즈니스) 관련 NCS 세분류 키워드:
      ncs_se_list: 'ERP', 'SCM', '정보시스템', '경영정보', 'IT컨설팅', '핀테크', '금융IT', '디지털'
      채용제목(보조): 'ERP', 'SCM', '디지털', '경영정보', '정보시스템', '기획'

    DB모듈 관련 공고 예시:
    MATCH (j:JobPosting)
    WHERE (j.ncs_se_list IS NOT NULL AND any(sf IN j.ncs_se_list WHERE sf CONTAINS 'ERP' OR sf CONTAINS 'SCM' OR sf CONTAINS '경영정보' OR sf CONTAINS '정보시스템'))
       OR (j.채용제목 CONTAINS 'ERP' OR j.채용제목 CONTAINS 'SCM' OR j.채용제목 CONTAINS '디지털'
           OR j.채용제목 CONTAINS '경영정보' OR j.채용제목 CONTAINS '정보시스템')
    RETURN j.채용제목, j.기관명, j.고용형태, j.근무지, j.등록일, j.마감일, j.상태,
           j.req_certifications, j.req_preferred, j.req_major, j.req_toeic_min, j.req_career, j.req_english_tests,
           j.ncs_se_list
    ORDER BY CASE WHEN j.고용형태 = '정규직' THEN 0 ELSE 1 END
    LIMIT 20

17. [기관별 채용 직무 현황 — 중요] "○○○ 채용 직무는?", "○○○ 어떤 직무 채용해?", "○○○에서 어떤 일 해?"처럼
    특정 기관이 어떤 직무를 뽑는지 묻는 질문은 개별 공고 목록이 아닌 NCS 직무 집계 쿼리를 생성할 것.
    규칙 4의 req_* 필드 목록은 이 유형에서 포함하지 않아도 됨.

    예시 (부산항만공사 채용 직무):
    MATCH (j:JobPosting)-[:HAS_NCS]->(n:NCS)
    WHERE j.기관명 CONTAINS '부산항만공사'
    WITH n.대분류 AS 대분류, n.중분류 AS 중분류, n.소분류 AS 소분류, n.세분류 AS 세분류,
         count(DISTINCT j) AS 공고수
    RETURN 대분류, 중분류, 소분류, 세분류, 공고수
    ORDER BY 공고수 DESC
    LIMIT 15
    ※ null 필터 절대 추가 금지 — 세분류·소분류가 없는 기관도 대분류만으로 의미 있는 답변 가능
"""

# ── 질의 전처리 프롬프트 ──────────────────────────────────────────
QUERY_REWRITE_SYSTEM = """공공기관 취업 Q&A 시스템의 질의 전처리 전문가입니다.
사용자 질문을 분석하여 검색 성능을 높이기 위한 정보를 JSON으로 반환하세요.

작업:
1. normalized_question: 줄임말·구어체를 공식 명칭으로 바꾼 정규화된 질문
   - 정처기 → 정보처리기사, 컴활 → 컴퓨터활용능력, 토스 → 토익스피킹, 전산세무 → 전산세무회계 등
   - "지원할 수 있는 공고 뭐 있어?" 같은 구어체를 검색 친화적 문장으로 변환
   - 학과 모듈명은 반드시 아래 직무 키워드로 풀어서 변환할 것:
     DS모듈 / 데이터사이언스모듈 → "데이터·AI·빅데이터·분석·통계 관련 채용공고"
     DB모듈 / 디지털비즈니스모듈 → "ERP·SCM·디지털·경영정보·기획 관련 채용공고"
   - 명사형 질문(동사 없음)도 검색 의도에 맞게 문장으로 변환
     예: "데이터사이언스모듈 관련 채용공고" → "데이터·AI·빅데이터·분석 관련 채용공고 알려줘"

2. sub_queries: 벡터 검색용 서브 쿼리 목록 (최대 3개)
   - 복합 조건 질문이면 조건별로 분리 (예: 자격증 쿼리, 어학 쿼리, 전공 쿼리를 각각)
   - 단순 질문이면 다양한 표현으로 paraphrase (2~3개)
   - 각 서브 쿼리는 채용공고 문서에 등장할 법한 문장 형태로

3. extracted_conditions: 질문에서 추출한 구조화 조건 (해당 없으면 null)
   - certifications: 자격증 목록 (예: ["정보처리기사","컴퓨터활용능력"])
   - toeic: 토익 최소 점수 (숫자)
   - opic: 오픽 최소 등급 (문자열, 예: "IM1")
   - education: 학력 (무관/고졸/초대졸/대졸/석사/박사)
   - major: 전공 계열 (무관/이공계/상경계/법학/인문/사범)
   - career: 경력 (신입/경력 — 사용자가 명시한 경우에만. 언급 없으면 null)
   - tech_stack: 기술스택 목록 (예: ["Python","SQL"])
   - org_type: 기관 유형 키워드 (예: "금융","에너지","의료")
   - preferred: 우대사항 (예: ["장애인","지역인재","보훈"])

JSON만 반환하세요. 예시:
{
  "normalized_question": "정보처리기사 자격증이 있을 때 지원 가능한 공공기관 채용공고",
  "sub_queries": [
    "정보처리기사 자격증 요구 또는 우대 채용공고",
    "IT 직무 자격증 보유자 지원 가능 공공기관",
    "정보처리기사 가산점 공공기관 채용"
  ],
  "extracted_conditions": {
    "certifications": ["정보처리기사"],
    "career": "신입"
  }
}"""

ANSWER_SYSTEM = """취업 준비생을 위한 AI 상담사입니다.
반드시 아래에 제공된 [검색 결과] 데이터만을 근거로 답변하세요.

[중요] 데이터 기준 안내:
이 시스템의 채용공고 DB는 2021년~2026년 2월에 등록된 과거 공고입니다.
현재 모집 중인 공고가 아닐 수 있으므로, 답변 시 반드시 아래를 지킬 것:
- "지원하세요", "지원 가능합니다", "현재 모집 중입니다" 같은 표현 절대 사용 금지.
- 대신 "DB 기준으로는", "과거 공고 기준으로는", "이런 공고들이 있었어요" 식으로 표현할 것.
- 공고 목록을 보여줄 때는 첫 줄에 반드시 "(2021~2026년 공고 기준이며, 현재 모집 여부는 각 기관에서 확인하세요)" 문구를 한 번만 안내할 것.

원칙:
0. [절대 금지 — 데이터가 있을 때] [채용공고 구조 검색 결과] 또는 [채용공고 본문 검색 결과]에 데이터가
   1건이라도 있으면 다음 표현을 절대 사용하지 말 것:
   ❌ "이 질문은 채용공고·경험 게시글 DB로는 답하기 어려운 영역이에요."
   ❌ "포털 검색을 이용해 주세요."
   ❌ "관련 데이터를 찾지 못했어요."
   데이터가 있으면 반드시 그 데이터로 답변할 것. 위 표현은 [검색 결과 없음]일 때만 허용.

1. [검색 결과]에 데이터가 조금이라도 있으면 반드시 그 내용을 활용해서 답변할 것.
   특히 "요구하는게 뭐야", "필요한 자격은", "어떤 조건이 있어", "어학 요건은", "자격증 조건은" 같은
   요건 집계형 질문은:
   검색된 공고들의 req_certifications·req_preferred·req_major·req_toeic_min·req_career 값을
   집계·분석하여 공통 패턴을 정리해서 답변할 것.

   [직접 답변 우선 — 중요] 특정 조건을 묻는 질문에는 핵심 수치·요건을 첫 줄에 바로 말할 것.
   ❌ 나쁜 예: "일부 공고에서 TOEIC 800점을 요구하기도 했으나, 다른 공고에서는..."
   ✅ 좋은 예: "에스알(SR) 공고의 대부분(80%)에서 TOEIC 800점 이상을 요구했어요."
   → 핵심을 먼저, 부가 설명은 그 다음에.

   [빈도 기준] 요건을 나열할 때 반드시 빈도를 구분할 것:
   - 전체 공고의 절반 이상에서 나오는 요건 → "대부분의 공고에서 요구" + 구체적 건수·점수 명시
   - 일부 공고에서만 나오는 요건(1~2건 등) → "일부 공고에서만 요구" 또는 해당 공고 제목과 함께 명시
   - 1~2건짜리 희귀 요건을 공통 요건처럼 일반화하지 말 것.
   - 소수(20% 이하) 예외 사항을 먼저 언급하며 핵심을 흐리지 말 것.

2. 사용자가 본인의 스펙을 제시하며 지원 가능한 공고를 물으면:
   각 공고마다 아래 형식으로 실제 DB 값을 그대로 표시할 것. "없음", "조건 없음" 같은 말은 쓰지 말 것.
   - 필수 자격증(req_certifications): 값이 있으면 목록 그대로 표시. 사용자 자격증과 겹치는 항목에 ✅ 표시.
     예) 필수 자격증: 정보처리기사, ✅ 컴퓨터활용능력, 사무자동화산업기사
   - 우대사항(req_preferred): 값이 있으면 목록 그대로 표시. 사용자 조건과 겹치면 ✅ 표시.
     예) 우대사항: ✅ 토익700점이상, 장애인, 보훈
   - 전공(req_major): 실제 값 표시. 사용자 전공과 맞으면 ✅.
     예) 전공: ✅ 상경계  또는  전공: 이공계
   - 어학(req_toeic_min): 실제 요구점수 표시. 사용자 점수로 충족되면 ✅.
     예) 토익: ✅ 650점 이상 (보유 700점 충족)  또는  토익: 800점 이상 (미충족)
   - 값이 없는 필드는 아예 표시하지 말 것 (공란으로 두거나 해당 줄 생략)
3. 채용공고 데이터가 있으면 기관명·직무명·조건·수치를 그대로 인용해서 목록 형태로 정리.
   공고 목록을 보여줄 때 반드시 아래 순서로 표시할 것 (값이 없는 필드는 생략):
     - 기관명 - 채용제목
     - 고용형태 / 근무지
     - 등록일 ~ 마감일 (상태가 "마감"이면 "[마감]" 표시)
     - 경력 / 전공
     - NCS (ncs_se_list에 값이 있을 때): 직무 분야 이해를 돕기 위해 핵심 항목 2~3개만 표시
     - 필수 자격증 (req_certifications에 값이 있을 때만)
     - 우대사항 (req_preferred에 값이 있을 때만)
     - 어학: req_toeic_min 값이 있으면 "TOEIC N점 이상", 없으면 표시 생략
   [구조 검색] 결과에 AVG/COUNT 같은 집계 수치가 있으면 그것을 답변의 핵심으로 먼저 제시할 것.
   [본문 검색] 결과가 질문의 직무·분야와 명백히 다른 경우(예: IT 직무 질문에 의료·병원 공고, 행정 질문에 IT 공고)는 해당 결과를 무시하고 언급하지 말 것.
4. 경험 게시글이 있으면 반드시 댓글 내용(특히 채택된 댓글)을 직접 인용하거나 구체적으로 요약할 것.
   절대 "고민이 있었다", "궁금증이 제기되었다" 같은 말로 얼버무리지 말 것.
   좋은 예) "채택된 답변에서는 '공기업은 NCS 필기 비중이 높아서 인턴보다 필기 준비가 더 중요하다'고 했어요."
   나쁜 예) "인턴 경험에 대한 고민이 있었으며 개인 상황에 따라 다르다고 했습니다."
5. 검색된 데이터가 질문과 완전히 맞지 않으면, "정확히 일치하는 데이터는 없지만 관련 데이터를 찾았어요"라고 먼저 밝히고 그 내용 소개.
6. [검색 결과 없음]이 context에 명시된 경우에만:
   - 질문이 공기업/준정부기관 개념, 채용 시즌, 조직문화 등 DB에 없는 일반 지식에 관한 것이면:
     "이 질문은 채용공고·경험 게시글 DB로는 답하기 어려운 영역이에요. 포털 검색을 이용해 주세요." 라고 안내.
   - 그 외엔: "관련 데이터를 찾지 못했어요. 다른 키워드로 다시 질문해 주세요." 라고 답하기.
   ⚠️ 검색 결과에 데이터가 있는데 위 문구를 사용하는 것은 규칙 0 위반.
7. GPT 자신이 알고 있는 일반적인 조언, 팁, 추천사항은 절대 추가하지 않기.
8-1. [검색 결과 해석 규칙]
   - req_toeic_min 값이 null이거나 없는 공고 = 토익 조건이 없는 공고. 반드시 이를 활용해서 답변할 것.
   - req_english_tests 값이 null이거나 빈 배열인 공고 = 어학 조건이 아예 없는 공고.
   - 등록일 = 공고 게시일, 마감일 = 지원 마감일, 상태 = 현재 상태(마감/접수중).
     "지원 기간", "공고 기간", "언제까지" 같은 질문에는 이 세 필드를 사용해서 답변할 것.
     데이터가 있으면 "데이터에 포함되지 않습니다"라고 절대 하지 말 것.
   - 구조 검색 결과에 공고가 1건이라도 있으면 반드시 그 공고 내용을 포함해서 답변할 것.
     절대 "관련 데이터를 찾지 못했어요"라고 하지 말 것.
10. [사회적 배려 항목 필터링 — 모든 답변에 적용] 질문 유형과 무관하게:
   - 장애인·보훈·취업보호대상·저소득·다문화·중장년·지역인재·청년·경력단절·북한이탈주민 등 사회적 배려 대상자 항목은
     우대사항·자격 조건 어디에도 절대 포함하지도, 언급하지도 말 것.
   - 이 항목들은 사회적 배려 채용 트랙이며, 일반 취준생에게는 해당되지 않음.
   - "우대 자격증", "우대사항", "지원 조건", "자격 조건" 등 어떤 질문 형태이든 동일하게 적용.
   - 실제 자격증(정보처리기사, 컴활 등)·어학(TOEIC 등)·직무역량 항목만 우대사항으로 안내할 것.
   - 자격증·어학 항목이 하나도 없으면 "해당 공고의 우대사항에서 자격증·어학 조건을 찾지 못했어요."라고 안내.

11. [전공 req_major null = 전공 무관] req_major 값이 null이거나 데이터에 없는 공고 = 전공 무관.
   - 반드시 "전공 무관"으로 표시할 것.
   - null값을 이공계·상경계·경상 등 특정 전공으로 절대 해석·추론하지 말 것.
   - 공고 제목이나 NCS 직무에서 전공을 추론해서 답하는 것도 금지.
   - req_major가 명시된 경우에만 해당 값을 전공 요건으로 안내할 것.

9. [특수 전문직 기본 제외] 사용자가 특정 전문 직종(의료·법조·회계 등)을 명시하지 않은 일반적인 질문일 때,
   아래 전문 자격이 필수인 공고는 답변에서 제외할 것. 검색 결과에 있어도 무시.
   제외 대상 필수 자격: 간호사, 의사, 약사, 물리치료사, 방사선사, 임상병리사, 치과, 한의사, 수의사,
                        공인회계사, CPA, 세무사, 변호사, 법무사, 감정평가사, 노무사, 관세사
   (우대사항에만 있는 건 제외 대상 아님 — req_certifications 필수 항목만 해당)
   일반 직종 공고가 하나도 없을 때만 "해당 조건의 일반 직종 공고를 찾지 못했어요. 조건을 바꿔서 다시 질문해 주세요."라고 안내.
8. 특정 스펙/경험이 중요한지 묻는 질문의 경우 (예: "인턴 경험 없으면 불리해?"):
   - 먼저 채용공고 데이터 기반 사실을 수치와 함께 제시할 것.
     예) "실제 공고 데이터에서 인턴/일경험을 우대사항으로 명시한 공고는 N건이에요."
     예) "신입 지원 가능 공고 중 관련 경험을 우대하는 공고 예시: [기관명 - 우대사항 목록]"
   - 그 다음 경험 게시글의 채택 댓글 내용을 구체적으로 인용해서 보완.
   - 공고 데이터(객관적 사실)와 경험 게시글(실제 경험담)을 명확히 구분해서 제시.

12. [기관별 채용 직무 안내] 특정 기관의 '채용 직무'를 묻는 질문에 NCS 집계 데이터(대분류·중분류·소분류·세분류, 공고수)가 있으면:
   - 개별 공고 목록(제목, 등록일 등)을 나열하지 말 것.
   - NCS 직무 기준으로 "○○○은 ~~·~~·~~ 직무를 주로 채용해왔어요" 형태로 직무명을 요약할 것.
   - 공고수가 많은 직무를 앞에 배치하고, 건수를 괄호로 함께 표기할 것.
     예) "경영기획 (12건), 총무·인사 (8건), 물류관리 (6건)"
   - 공고 제목(예: "정규직(신입) 채용 공고")을 직무명으로 오인해서 나열하지 말 것.
   - [NCS 레벨 표현 — 중요] 데이터에 소분류·세분류가 없고 대분류만 있는 경우:
     반드시 "NCS 대분류 기준"이라고 표현할 것. "NCS 세분류 목록"이라고 하면 절대 안 됨.
     소분류/세분류 컬럼이 null이거나 비어있으면 해당 레벨은 존재하지 않는 것임.
   - 전체 목록을 나열한 뒤 다시 "주로 많이 뽑는 직무"를 반복 나열하는 것 금지.
     공고수 순서대로 한 번만 정리할 것."""


# ── 공고 데이터 전처리 헬퍼 ─────────────────────────────────────────
_SOCIAL_WELFARE_TERMS = {
    "장애인", "보훈", "취업보호대상", "국가유공자", "저소득", "다문화",
    "중장년", "청년", "경력단절", "북한이탈주민", "지역인재",
    "사회적기업", "보훈대상", "보훈취업보호",
}

def _preprocess_posting_row(row: dict) -> dict:
    """GPT에 전달하기 전 공고 데이터 정제:
    - req_major null/없음 → '전공 무관' (GPT가 전공을 임의로 추론하지 않도록)
    - req_preferred에서 사회적 배려 항목 제거 (청년·장애인·보훈 등을 우대사항으로 오인하지 않도록)
    """
    row = dict(row)
    # req_major null → '전공 무관'
    if not row.get("req_major"):
        row["req_major"] = "전공 무관"
    # req_preferred 사회적 배려 항목 제거
    if isinstance(row.get("req_preferred"), list):
        filtered = [
            p for p in row["req_preferred"]
            if not any(sw in str(p) for sw in _SOCIAL_WELFARE_TERMS)
        ]
        if filtered:
            row["req_preferred"] = filtered
        else:
            row.pop("req_preferred", None)
    return row


# ── Step 0: 질의 전처리 (Query Rewriting + Multi-Query 생성) ──────
def rewrite_query(question: str, history: list[dict] | None = None) -> dict:
    """줄임말 확장·정규화, 서브 쿼리 생성, 조건 추출.
    history가 있으면 직전 대화 맥락을 붙여 지시어(그 공고, 그 중에서 등)를 풀어냄.
    """
    user_msg = question
    if history:
        # 최근 2턴(4개 메시지)만 사용, 어시스턴트 답변은 300자 이내로 축약
        recent = history[-4:]
        ctx_lines = []
        for msg in recent:
            content = msg["content"]
            if msg["role"] == "assistant" and len(content) > 300:
                content = content[:300] + "...(이하 생략)"
            ctx_lines.append(f"[{msg['role']}]: {content}")
        user_msg = "[이전 대화 맥락]\n" + "\n".join(ctx_lines) + f"\n\n[현재 질문]\n{question}"

    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": QUERY_REWRITE_SYSTEM},
                {"role": "user",   "content": user_msg}
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        result = json.loads(resp.choices[0].message.content)
        return {
            "normalized": result.get("normalized_question", question),
            "sub_queries": result.get("sub_queries", [question]),
            "conditions":  result.get("extracted_conditions") or {},
        }
    except Exception:
        return {"normalized": question, "sub_queries": [question], "conditions": {}}


# ── Step 1: 경로 판단 ─────────────────────────────────────────────
def route_question(question: str) -> tuple[str, str, str]:
    """
    질문 유형을 4가지 경로 중 하나로 판단.
      dept_career : 동아대 경영정보학과 진로·커리큘럼 질문
      ncs_prep    : 특정 기관 NCS 준비 질문  (org_name 함께 반환)
      posting     : 채용 조건·자격·요건 질문 → 채용공고 DB → 없으면 QAPost 폴백
      experience  : 면접·자소서·취업 고민    → 바로 QAPost 게시글·댓글로 답변
    """
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": ROUTE_SYSTEM},
            {"role": "user",   "content": question}
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    result = json.loads(resp.choices[0].message.content)
    return (result.get("route", "posting"),
            result.get("reason", ""),
            result.get("org_name", ""))


# ── 공통: 텍스트 임베딩 ──────────────────────────────────────────
def embed_text(text: str) -> list[float]:
    resp = client.embeddings.create(model=EMBED_MODEL, input=text)
    return resp.data[0].embedding


# ── Cypher 사후 보정: OR/AND 우선순위 괄호 자동 추가 ─────────────
def _fix_cypher_parentheses(cypher: str) -> str:
    """WHERE 절에서 괄호 없이 OR 조건이 AND와 섞인 경우 자동으로 괄호를 추가한다.

    패턴 A (OR 조건이 한 줄에):
        WHERE a OR b OR c
        AND d AND e
    → WHERE (a OR b OR c)
      AND d AND e

    패턴 B (OR 조건이 여러 줄에):
        WHERE a
        OR b
        OR c
        AND d
    → WHERE (a
          OR b
          OR c)
      AND d
    """
    lines = cypher.split('\n')
    result = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r'^(\s*WHERE\s+)(.*)', line, re.IGNORECASE)
        if m:
            prefix = m.group(1)
            rest   = m.group(2).rstrip()

            # 패턴 B: WHERE 줄 뒤에 OR 줄들이 이어지고 그 뒤에 AND가 오는 경우
            or_extra = []
            j = i + 1
            while j < len(lines) and re.match(r'^\s*OR\s+', lines[j], re.IGNORECASE):
                or_extra.append(lines[j])
                j += 1
            has_and_after = (j < len(lines) and
                             re.match(r'^\s*AND\s+', lines[j], re.IGNORECASE))

            if or_extra and has_and_after and not rest.lstrip().startswith('('):
                # OR 조건 줄들을 괄호로 묶기
                indent = ' ' * (len(prefix))
                or_block = f"{prefix}({rest}\n"
                or_block += ''.join(f"{l}\n" for l in or_extra)
                or_block = or_block.rstrip('\n') + ')'
                result.append(or_block)
                i = j
                continue

            # 패턴 A: WHERE 한 줄에 OR가 있고 다음 줄이 AND인 경우
            next_idx = i + 1
            if (' OR ' in rest.upper() and
                    not rest.lstrip().startswith('(') and
                    next_idx < len(lines) and
                    re.match(r'^\s*AND\s+', lines[next_idx], re.IGNORECASE)):
                result.append(f"{prefix}({rest})")
                i += 1
                continue

        result.append(line)
        i += 1
    return '\n'.join(result)


# ── Reranker ─────────────────────────────────────────────────────
RERANK_SYSTEM = """채용공고 검색 결과 Reranker입니다.
질문과 후보 검색 결과 목록이 주어지면, 각 후보의 관련성 점수(0.0~1.0)를 평가하세요.

평가 기준:
- 질문에서 요청한 고용형태(정규직/인턴/무기계약직 등)와 일치할수록 높은 점수
- 질문의 직무 분야(IT/금융/행정 등)와 NCS 직무 분류·채용제목이 맞을수록 높은 점수
- 질문에서 언급한 자격증·어학·전공 조건과 공고 요건이 맞을수록 높은 점수
- 질문에서 명시하지 않은 특수 전문직(의료·법조·회계사 등) 필수 자격이 있는 공고는 낮은 점수
- 질문의 지역 조건과 근무지가 맞으면 높은 점수

JSON으로만 반환:
{"scores": [{"index": 0, "score": 0.9}, {"index": 1, "score": 0.3}, ...]}"""


def rerank_results(question: str, candidates: list[dict], top_k: int = 8,
                   mode: str = "graph") -> list[dict]:
    """GPT 기반 Reranker: 후보 목록을 질문 관련성 순으로 재정렬하고 상위 top_k 반환.

    mode="graph"  → 채용공고 구조 데이터 (기관명, 직무, 고용형태 등)
    mode="chunk"  → 채용공고 본문 청크 (텍스트 내용 + 공고명)
    """
    if not candidates or len(candidates) <= 2:
        return candidates[:top_k]

    # 후보 목록을 한 줄 요약으로 변환 (토큰 절약)
    if mode == "graph":
        lines = []
        for i, r in enumerate(candidates):
            기관 = r.get("기관명") or r.get("j.기관명") or ""
            제목 = r.get("채용제목") or r.get("j.채용제목") or ""
            고용 = r.get("고용형태") or r.get("j.고용형태") or ""
            자격 = r.get("req_certifications") or r.get("j.req_certifications") or []
            ncs  = r.get("대분류") or r.get("n.대분류") or ""
            근무 = r.get("근무지") or r.get("j.근무지") or ""
            lines.append(f"{i}. [{기관} — {제목}] 고용:{고용} 근무:{근무} NCS:{ncs} 자격:{자격}")
    else:  # chunk
        lines = []
        for i, r in enumerate(candidates):
            기관 = r.get("기관") or r.get("기관명") or ""
            공고 = r.get("공고명") or ""
            내용 = (r.get("내용") or "")[:100]
            lines.append(f"{i}. [{기관} — {공고}] {내용}")

    prompt = f'질문: "{question}"\n\n후보 {len(candidates)}개:\n' + "\n".join(lines)

    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": RERANK_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        result = json.loads(resp.choices[0].message.content)
        scored = result.get("scores", [])
        # 점수 내림차순 정렬
        scored.sort(key=lambda x: x.get("score", 0), reverse=True)
        valid_idx = [s["index"] for s in scored
                     if isinstance(s.get("index"), int) and 0 <= s["index"] < len(candidates)]
        if valid_idx:
            reranked = [candidates[i] for i in valid_idx[:top_k]]
            # top_k 미달이면 미포함 후보 순서대로 채우기
            included = set(valid_idx[:top_k])
            for j in range(len(candidates)):
                if len(reranked) >= top_k:
                    break
                if j not in included:
                    reranked.append(candidates[j])
            return reranked
    except Exception:
        pass
    return candidates[:top_k]


# ── NCS 준비 경로 ─────────────────────────────────────────────────
NCS_PREP_ANSWER_SYSTEM = """공공기관 취업 준비생을 위한 NCS 직무 안내 AI입니다.
아래 [기관 NCS 데이터]를 바탕으로 해당 기관이 실제로 채용한 NCS 직무를 안내하세요.

답변 구성 (순서대로):
1. 해당 기관이 채용한 전체 직무(NCS 세분류) 목록 — 데이터에 있는 모든 세분류를 빠짐없이 나열
2. 그 중 공고 수가 많은 주요 직무 — 상위 직무를 별도로 강조하며 "주로 이런 직무를 많이 뽑아요" 형태로 안내

원칙:
- [기관 NCS 데이터]에 있는 내용만 사용. 없는 정보는 만들지 말 것.
- 데이터가 없는 기관이면 솔직하게 "DB에 해당 기관 공고가 없습니다"라고 안내.
- GPT 자신이 알고 있는 일반적인 NCS 정보나 시험 준비법은 추가하지 말 것.
- 지원 요건(학력, 자격증, 어학 등)은 별도 질문이 들어오기 전까지 언급하지 말 것."""



DEPT_CAREER_ANSWER_SYSTEM = """동아대학교 경영정보학과 학생 전용 진로 상담 AI입니다.
아래 [학과 데이터]를 기반으로 구체적이고 실용적인 조언을 제공하세요.

답변 원칙:
1. [학과 데이터]에 있는 내용만 사용할 것. 없는 내용은 만들지 말 것.
2. 학생이 물어본 맥락(학년, 관심 모듈, 진로 방향)에 맞게 맞춤 답변할 것.
3. 교과과정 과목명을 구체적으로 언급하면서 "이 과목을 들으면 이런 역량을 기를 수 있어요" 식으로 연결할 것.
4. 자격증은 어떤 진로에 어떤 자격증이 필요한지 구체적으로 연결해서 설명할 것.
5. 실제 취업 기업명(예: 국민건강보험공단, GS리테일, 베스핀글로벌 등)을 언급하면서 현실감 있게 답변할 것.
6. 진로 방향이 2가지 이상이면 각 모듈별로 정리해서 비교해줄 것.
7. 학년별 로드맵이 있으면 해당 학년 기준으로 구체적인 할 일 목록을 제시할 것.
8. 답변 마지막에 안내 문구나 광고성 멘트("추가로 궁금한 점이 있으면..." 등)는 절대 추가하지 말 것.
"""


def _query_dept_context(question: str) -> str:
    """질문 키워드를 분석해 Neo4j에서 필요한 학과 데이터만 조회 후 텍스트로 반환.

    항상 포함: Department 기본 정보 (학과소개·학과장말·교육목표·취업률)
    키워드에 따라 추가:
      - 교과목/수업/커리큘럼/학과교양/전공/필수 → Course 노드
      - 자격증/자격/취득 → Qualification 노드
      - 모듈/DS/DB/데이터사이언스/디지털비즈니스/진출/진로방향 → CareerModule + CareerField
      - 취업처/취업기업/어디취업/어느기업/졸업생 → GradCompany 노드 (섹터별)
      - 비교과/추천활동/대외활동 → ExtraActivity 노드
      - 직무능력/역량/스킬/능력 → CoreSkill 노드
      - (키워드 없으면) 모든 카테고리에서 핵심 정보만
    """
    q = question

    # 키워드 집합
    kw_course    = {"교과목", "수업", "커리큘럼", "과목", "학과교양", "전공필수",
                    "전공선택", "무엇을 배우", "뭐 배우", "뭘 배워", "강의"}
    kw_cert      = {"자격증", "자격", "취득", "ADP", "ADsP", "SQLD", "SQLP",
                    "정보처리기사", "빅데이터분석기사", "SAP", "ERP정보관리사"}
    kw_module    = {"모듈", "데이터사이언스", "디지털비즈니스", "DS모듈", "DB모듈",
                    "진출분야", "진로방향", "어떤 모듈", "뭐가 좋아", "차이"}
    kw_company   = {"취업처", "취업기업", "어디 취업", "어느 기업", "졸업생",
                    "어디로 가", "어디 가", "취업한 곳", "취업했어", "진출현황"}
    kw_activity  = {"비교과", "추천활동", "대외활동", "동아리", "멘토링", "인턴",
                    "프로그램", "활동"}
    kw_skill     = {"직무능력", "역량", "스킬", "능력", "배우는 기술", "기술"}
    kw_employ    = {"취업률", "취업 비율", "얼마나 취업", "취업 통계", "취업 현황"}

    need_course   = any(k in q for k in kw_course)
    need_cert     = any(k in q for k in kw_cert)
    need_module   = any(k in q for k in kw_module)
    need_company  = any(k in q for k in kw_company)
    need_activity = any(k in q for k in kw_activity)
    need_skill    = any(k in q for k in kw_skill)
    need_employ   = any(k in q for k in kw_employ)

    # 아무 키워드도 없으면 전부 조회 (일반 소개 질문)
    fetch_all = not any([need_course, need_cert, need_module,
                         need_company, need_activity, need_skill, need_employ])

    parts = []

    try:
        with driver.session(database=NEO4J_DB) as s:

            # ── Department 기본 정보 (항상) ──────────────────────────
            dept = s.run("""
                MATCH (d:Department {name: '경영정보학과'})
                RETURN d.name AS name, d.college AS college,
                       d.dept_intro AS dept_intro,
                       d.dean_message AS dean_message,
                       d.edu_goal_ds AS edu_goal_ds,
                       d.edu_goal_db AS edu_goal_db,
                       d.intro_why AS intro_why,
                       d.intro_what AS intro_what,
                       d.intro_learn AS intro_learn,
                       d.intro_career AS intro_career,
                       d.employ_rate_2020 AS r2020, d.employ_rate_2021 AS r2021,
                       d.employ_rate_2022 AS r2022, d.employ_rate_2023 AS r2023,
                       d.employ_rate_2024 AS r2024,
                       d.employ_sector_service AS s_service,
                       d.employ_sector_it AS s_it,
                       d.employ_sector_mfg AS s_mfg,
                       d.employ_sector_public AS s_public,
                       d.employ_sector_other AS s_other,
                       d.credit_req_core AS req_core,
                       d.credit_req_module AS req_module
            """).single()

            if dept:
                parts.append(f"[학과 기본 정보]")
                parts.append(f"학과명: {dept['name']} ({dept['college']})")
                parts.append(f"학과소개: {dept['dept_intro']}")
                parts.append(f"학과장 메시지: {dept['dean_message']}")
                parts.append(f"교육목표(데이터사이언스): {dept['edu_goal_ds']}")
                parts.append(f"교육목표(디지털비즈니스): {dept['edu_goal_db']}")
                if need_employ or fetch_all:
                    parts.append(
                        f"취업률: 2020년 {dept['r2020']}% / 2021년 {dept['r2021']}% / "
                        f"2022년 {dept['r2022']}% / 2023년 {dept['r2023']}% / 2024년 {dept['r2024']}%"
                    )
                    parts.append(
                        f"분야별 취업비율: 서비스업(은행·유통등) {dept['s_service']}% / "
                        f"정보통신 {dept['s_it']}% / 제조업 {dept['s_mfg']}% / "
                        f"공기업·공무원 {dept['s_public']}% / 기타(진학등) {dept['s_other']}%"
                    )
                parts.append(
                    f"이수요건: 핵심모듈 {dept['req_core']}과목 필수 이수 + "
                    f"선택 모듈 {dept['req_module']}과목 이상 이수"
                )

            # ── Course ───────────────────────────────────────────────
            if need_course or fetch_all:
                # 카테고리별로 묶어서 출력
                for cat in ["학과교양", "전공필수", "DS모듈", "DB모듈", "전공선택"]:
                    courses = s.run("""
                        MATCH (d:Department {name: '경영정보학과'})-[:HAS_COURSE]->(c:Course)
                        WHERE c.category = $cat
                        RETURN c.code AS code, c.name AS name,
                               c.credits AS credits, c.semester AS semester,
                               c.description AS desc
                        ORDER BY c.semester, c.code
                    """, cat=cat).data()
                    if courses:
                        parts.append(f"\n[교과목 — {cat}]")
                        for c in courses:
                            parts.append(
                                f"  {c['code']} {c['name']} ({c['credits']}학점, "
                                f"{c['semester']}학기)"
                            )
                            if c['desc']:
                                parts.append(f"    설명: {c['desc']}")

            # ── CareerModule + COVERS + LEADS_TO ─────────────────────
            if need_module or fetch_all:
                modules = s.run("""
                    MATCH (d:Department {name: '경영정보학과'})-[:HAS_MODULE]->(cm:CareerModule)
                    RETURN cm.name AS name, cm.description AS desc
                """).data()
                for mod in modules:
                    parts.append(f"\n[진로모듈 — {mod['name']}]")
                    parts.append(f"  설명: {mod['desc']}")
                    # 연도별 이수 과목
                    for yr in [2, 3, 4]:
                        yr_courses = s.run("""
                            MATCH (cm:CareerModule {name: $mname})-[r:COVERS {year: $yr}]->(c:Course)
                            RETURN c.name AS name
                        """, mname=mod['name'], yr=yr).data()
                        if yr_courses:
                            names = ", ".join(c['name'] for c in yr_courses)
                            parts.append(f"  {yr}학년 이수과목: {names}")
                    # 진출분야
                    fields = s.run("""
                        MATCH (cm:CareerModule {name: $mname})-[r:LEADS_TO]->(cf:CareerField)
                        RETURN cf.name AS field, r.description AS desc
                    """, mname=mod['name']).data()
                    if fields:
                        parts.append(f"  주요 진출분야:")
                        for f in fields:
                            parts.append(f"    - {f['field']}: {f['desc']}")

            # ── Qualification ─────────────────────────────────────────
            if need_cert or fetch_all:
                certs = s.run("""
                    MATCH (d:Department {name: '경영정보학과'})-[:HAS_CERT]->(q:Qualification)
                    RETURN q.name AS name, q.category AS cat
                    ORDER BY q.name
                """).data()
                if certs:
                    parts.append("\n[권장 자격증]")
                    parts.append("  " + ", ".join(c['name'] for c in certs))

            # ── CoreSkill ─────────────────────────────────────────────
            if need_skill or fetch_all:
                skills = s.run("""
                    MATCH (d:Department {name: '경영정보학과'})-[:HAS_SKILL]->(sk:CoreSkill)
                    RETURN sk.name AS name
                """).data()
                if skills:
                    parts.append("\n[직무능력 (6가지)]")
                    for i, sk in enumerate(skills, 1):
                        parts.append(f"  {i}. {sk['name']}")

            # ── ExtraActivity ─────────────────────────────────────────
            if need_activity or fetch_all:
                acts = s.run("""
                    MATCH (d:Department {name: '경영정보학과'})-[:HAS_ACTIVITY]->(ea:ExtraActivity)
                    RETURN ea.name AS name, ea.category AS cat, ea.org AS org
                    ORDER BY ea.category, ea.org
                """).data()
                if acts:
                    parts.append("\n[비교과 및 추천활동]")
                    비교과 = [a for a in acts if a['cat'] == '비교과']
                    추천   = [a for a in acts if a['cat'] == '추천활동']
                    if 비교과:
                        # 기관별로 묶기
                        from collections import defaultdict
                        by_org = defaultdict(list)
                        for a in 비교과:
                            by_org[a['org']].append(a['name'])
                        parts.append("  비교과 활동 (기관별):")
                        for org, names in by_org.items():
                            parts.append(f"    [{org}] {', '.join(names)}")
                    if 추천:
                        parts.append("  기타 추천활동:")
                        parts.append("    " + ", ".join(a['name'] for a in 추천))

            # ── GradCompany ───────────────────────────────────────────
            if need_company or fetch_all:
                companies = s.run("""
                    MATCH (d:Department {name: '경영정보학과'})-[:GRAD_EMPLOYED]->(gc:GradCompany)
                    RETURN gc.sector AS sector, collect(gc.name) AS names
                    ORDER BY gc.sector
                """).data()
                if companies:
                    parts.append("\n[졸업생 주요 취업처]")
                    for row in companies:
                        names_str = ", ".join(row['names'])
                        parts.append(f"  [{row['sector']}] {names_str}")

    except Exception as e:
        parts.append(f"[Neo4j 조회 오류: {e}]")

    return "\n".join(parts)


def dept_career_answer(question: str, history: list[dict] | None = None) -> str:
    """동아대 경영정보학과 학생 질문에 Neo4j 학과 데이터로 답변"""
    dept_context = _query_dept_context(question)
    context = f"[학과 데이터]\n{dept_context}\n\n[학생 질문]\n{question}"

    messages = [{"role": "system", "content": DEPT_CAREER_ANSWER_SYSTEM}]
    if history:
        for msg in history[-4:]:
            content = msg["content"]
            if msg["role"] == "assistant" and len(content) > 800:
                content = content[:800] + "...(이하 생략)"
            messages.append({"role": msg["role"], "content": content})
    messages.append({"role": "user", "content": context})

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.3,
    )
    return resp.choices[0].message.content




def ncs_prep_search(org_name: str, detail: bool = True) -> dict:
    """기관명으로 NCS 준비에 필요한 데이터 조회:
    - detail=True : 세분류·능력단위까지 반환 (기본값)
    - detail=False: 대분류만 반환 (간결한 답변용)
    """
    result = {"org_name": org_name, "ncs_jobs": [], "requirements": {}, "total": 0}
    try:
        with driver.session(database=NEO4J_DB) as s:
            if detail:
                ncs_rows = s.run("""
                    MATCH (j:JobPosting)-[:HAS_NCS]->(n:NCS)
                    WHERE j.기관명 CONTAINS $org
                    WITH n.대분류 AS 대분류,
                         n.중분류 AS 중분류,
                         n.소분류 AS 소분류,
                         n.세분류 AS 세분류,
                         n.능력단위 AS 능력단위,
                         count(DISTINCT j) AS 공고수
                    ORDER BY 공고수 DESC
                    RETURN 대분류, 중분류, 소분류, 세분류, 능력단위, 공고수
                    LIMIT 50
                """, org=org_name).data()
            else:
                ncs_rows = s.run("""
                    MATCH (j:JobPosting)-[:HAS_NCS]->(n:NCS)
                    WHERE j.기관명 CONTAINS $org
                    WITH n.대분류 AS 대분류,
                         count(DISTINCT j) AS 공고수
                    ORDER BY 공고수 DESC
                    RETURN 대분류, 공고수
                """, org=org_name).data()
            result["ncs_jobs"] = ncs_rows

            # 2) 기관 요구 조건 집계
            req = s.run("""
                MATCH (j:JobPosting)
                WHERE j.기관명 CONTAINS $org
                WITH count(j) AS 총공고수,
                     [x IN collect(j.req_certifications) WHERE x IS NOT NULL AND size(x) > 0 | x] AS 자격증목록,
                     [x IN collect(DISTINCT j.req_education) WHERE x IS NOT NULL AND x <> '' | x] AS 학력목록,
                     [x IN collect(DISTINCT j.req_career) WHERE x IS NOT NULL AND x <> '' | x] AS 경력목록,
                     [x IN collect(DISTINCT j.req_major) WHERE x IS NOT NULL AND x <> '무관' AND x <> '' | x] AS 전공목록,
                     count(CASE WHEN j.req_toeic_min IS NOT NULL THEN 1 END) AS 토익요구공고수,
                     avg(j.req_toeic_min) AS 평균토익
                RETURN 총공고수, 자격증목록, 학력목록, 경력목록, 전공목록, 토익요구공고수, 평균토익
            """, org=org_name).single()

            if req:
                result["total"] = req["총공고수"]
                # 자격증 빈도 집계
                cert_counter = {}
                for cert_list in (req["자격증목록"] or []):
                    if isinstance(cert_list, list):
                        for c in cert_list:
                            cert_counter[c] = cert_counter.get(c, 0) + 1
                    elif isinstance(cert_list, str):
                        cert_counter[cert_list] = cert_counter.get(cert_list, 0) + 1
                top_certs = sorted(cert_counter.items(), key=lambda x: -x[1])[:10]
                result["requirements"] = {
                    "학력": req["학력목록"],
                    "경력": req["경력목록"],
                    "전공": req["전공목록"],
                    "자격증_빈도": top_certs,
                    "토익요구공고수": req["토익요구공고수"],
                    "평균토익": round(req["평균토익"], 0) if req["평균토익"] else None,
                }
    except Exception as e:
        result["error"] = str(e)
    return result


# ── Step 2A: 채용공고 경로 ────────────────────────────────────────
def cypher_search(question: str, conditions: dict = None) -> tuple[str, list[dict]]:
    """Cypher 생성 + 실행. conditions가 있으면 구조화 조건을 함께 전달해 정확도 향상"""
    user_msg = f"질문: {question}"
    if conditions:
        user_msg += f"\n\n[추출된 조건 — Cypher 작성 시 적극 활용]\n{json.dumps(conditions, ensure_ascii=False)}"
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": CYPHER_SYSTEM},
            {"role": "user",   "content": user_msg}
        ],
        temperature=0,
    )
    raw = resp.choices[0].message.content
    m = re.search(r"```cypher\s*(.*?)```", raw, re.DOTALL)
    cypher = m.group(1).strip() if m else raw.strip()

    # OR/AND 우선순위 괄호 자동 보정
    cypher = _fix_cypher_parentheses(cypher)

    try:
        with driver.session(database=NEO4J_DB) as s:
            data = s.run(cypher).data()
        # Neo4j 타입(날짜 등) → 문자열 변환
        data = [{k: str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v
                 for k, v in row.items()} for row in data]
    except Exception as e:
        data = []
        cypher = f"오류: {e}"
    return cypher, data


def chunk_rag_search(question: str, org_name: str = None, top_k: int = 5) -> list[dict]:
    """채용공고 본문(Chunk)에서 벡터 RAG 검색 - 1536차원"""
    if not _INDEX["chunk"]:
        return []
    try:
        with driver.session(database=NEO4J_DB) as s:
            embedding = embed_text(question)

            if org_name:
                # 특정 기관 공고 Chunk로 범위를 좁혀서 벡터 검색
                # (Neo4j는 필터링된 벡터 검색을 직접 지원하지 않으므로
                #  전체 검색 후 기관명으로 필터링)
                results = s.run("""
                    CALL db.index.vector.queryNodes('chunk_embedding', $k2, $emb)
                    YIELD node AS c, score
                    MATCH (j:JobPosting)-[:HAS_CHUNK]->(c)
                    MATCH (o:Organization)-[:HAS_POSTING]->(j)
                    WHERE o.기관명 CONTAINS $org
                    RETURN o.기관명    AS 기관,
                           j.채용제목  AS 공고명,
                           j.고용형태  AS 고용형태,
                           c.text      AS 내용,
                           score
                    ORDER BY score DESC
                    LIMIT $k
                """, emb=embedding, org=org_name, k=top_k, k2=top_k * 10).data()
            else:
                # 전체 Chunk에서 유사도 검색 → 연결된 공고 정보 함께 반환
                results = s.run("""
                    CALL db.index.vector.queryNodes('chunk_embedding', $k, $emb)
                    YIELD node AS c, score
                    MATCH (j:JobPosting)-[:HAS_CHUNK]->(c)
                    OPTIONAL MATCH (o:Organization)-[:HAS_POSTING]->(j)
                    RETURN o.기관명    AS 기관,
                           j.채용제목  AS 공고명,
                           j.고용형태  AS 고용형태,
                           c.text      AS 내용,
                           score
                    ORDER BY score DESC
                    LIMIT $k
                """, emb=embedding, k=top_k).data()

        return results
    except Exception as e:
        return []


def chunk_rag_search_multi(queries: list[str], top_k: int = 5) -> list[dict]:
    """Multi-Query 벡터 검색 + Reranker.

    각 서브 쿼리로 top_k*3 개씩 검색 → 중복 제거 → Reranker로 top_k 선별.
    벡터 유사도만으로는 놓칠 수 있는 관련도를 GPT Reranker가 재평가한다.
    """
    seen = set()
    all_results = []
    fetch_k = top_k * 3  # 각 쿼리당 3배 더 가져와서 Reranker 후보 확보
    for q in queries:
        results = chunk_rag_search(q, top_k=fetch_k)
        for r in results:
            # 공고명+내용 앞부분으로 중복 판별
            key = (r.get("공고명", "") + (r.get("내용", "") or "")[:40])
            if key not in seen:
                seen.add(key)
                all_results.append(r)
    if not all_results:
        return []

    # 1차: 벡터 score 내림차순 정렬
    all_results.sort(key=lambda x: x.get("score", 0), reverse=True)

    # Reranker 적용 (후보가 top_k 초과일 때만 — 후보가 적으면 그냥 반환)
    if len(all_results) > top_k:
        # 쿼리는 첫 번째 서브 쿼리 사용 (가장 대표적인 질문)
        all_results = rerank_results(queries[0], all_results, top_k=top_k, mode="chunk")
    return all_results[:top_k]


# ── 키워드 폴백 검색 ──────────────────────────────────────────────
def _keyword_fallback(question: str) -> list[dict]:
    """벡터 검색 실패 시 키워드 기반 Chunk 검색 폴백"""
    LANG_TRIGGERS = {"어학", "토익", "토스", "오픽", "텝스", "토플", "opic",
                     "toeic", "toefl", "teps", "영어", "어학점수", "어학성적",
                     "어학시험", "외국어"}
    PAY_TRIGGERS  = {"월급", "연봉", "급여", "보수", "임금", "페이", "pay", "salary"}
    q_lower = question.lower().replace("?", "")
    q_words = set(q_lower.split())
    rows = []
    try:
        with driver.session(database=NEO4J_DB) as s:
            if q_words & PAY_TRIGGERS:
                for pk in ["보수수준", "월급", "급여", "연봉", "보수"]:
                    rows += s.run("""
                        MATCH (j:JobPosting)-[:HAS_CHUNK]->(c:Chunk)
                        WHERE c.text CONTAINS $kw
                        RETURN j.채용제목 AS 공고명, j.기관명 AS 기관,
                               j.고용형태 AS 고용형태, c.text AS 내용
                        LIMIT 5
                    """, kw=pk).data()
                    if len(rows) >= 5:
                        break
            elif q_words & LANG_TRIGGERS:
                for lk in ["토익", "토스", "오픽", "텝스", "토플", "어학",
                           "TOEIC", "OPIc", "TOEFL", "TEPS"]:
                    rows += s.run("""
                        MATCH (j:JobPosting)-[:HAS_CHUNK]->(c:Chunk)
                        WHERE c.text CONTAINS $kw
                        RETURN j.채용제목 AS 공고명, j.기관명 AS 기관,
                               j.고용형태 AS 고용형태, c.text AS 내용
                        LIMIT 5
                    """, kw=lk).data()
                    if len(rows) >= 5:
                        break
            else:
                for kw in [w for w in q_lower.split() if len(w) > 1][:3]:
                    r = s.run("""
                        MATCH (j:JobPosting)-[:HAS_CHUNK]->(c:Chunk)
                        WHERE c.text CONTAINS $kw
                        RETURN j.채용제목 AS 공고명, j.기관명 AS 기관,
                               j.고용형태 AS 고용형태, c.text AS 내용
                        LIMIT 5
                    """, kw=kw).data()
                    if r:
                        rows = r
                        break
    except Exception:
        pass
    return rows[:5]


# ── Step 2B: 경험 게시글 경로 ─────────────────────────────────────
QAPOST_SCORE_THRESHOLD = 0.80  # 이 점수 미만이면 "관련 없음"으로 처리

# 스펙 중요도 관련 키워드 (QAPost 검색 트리거)
SPEC_KEYWORDS = {"중요", "필요", "불리", "유리", "도움", "필수", "없어도",
                 "있어야", "어때", "써야", "해야", "효과", "가산점"}

def qapost_rag_search(question: str, top_k: int = 5,
                      public_only: bool = False,
                      min_score: float | None = None) -> list[dict]:
    """QAPost 벡터 검색 - 1536차원
    public_only=True : is_public_related=true 게시글만 (공기업 특정 기관 질문용)
    min_score        : 명시하면 QAPOST_SCORE_THRESHOLD 대신 사용
    """
    if not _INDEX["qapost"]:
        return []
    threshold = min_score if min_score is not None else QAPOST_SCORE_THRESHOLD
    embedding = embed_text(question)
    try:
        with driver.session(database=NEO4J_DB) as s:
            results = s.run("""
                CALL db.index.vector.queryNodes('qapost_embedding', $k, $emb)
                YIELD node AS p, score
                WHERE score >= $threshold
                AND ($public_only = false OR p.is_public_related = true)
                OPTIONAL MATCH (p)-[:HAS_COMMENT]->(c:Comment)
                RETURN p.title    AS 제목,
                       p.body     AS 본문,
                       p.category AS 카테고리,
                       p.platform AS 플랫폼,
                       score,
                       collect({
                           내용:    c.body,
                           직무:    c.author_job,
                           채택:    c.is_accepted,
                           좋아요:  c.likes
                       })[0..5] AS 상위댓글
                ORDER BY score DESC
            """, k=top_k, emb=embedding,
                threshold=threshold,
                public_only=public_only).data()
        return results
    except Exception:
        return []


# ── 공기업 공통 준비 집계 ─────────────────────────────────────────
# 경영정보학과 관련 NCS 대분류 (잡알리오 전체 공고 대상 — 기관명 필터 불필요)
_MIS_NCS_CATEGORIES = ["경영·회계·사무", "사업관리", "정보통신", "금융·보험"]

# ── 지역별 기관 목록 집계 ──────────────────────────────────────────
_REGION_KEYWORDS = [
    "부산", "서울", "인천", "대구", "광주", "대전", "울산", "세종",
    "경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주",
]
_ORG_LIST_TRIGGERS = ["전부", "모두", "다 말해", "목록", "어디어디", "어떤 기관", "전체", "알려줘", "있어", "있나", "뭐뭐"]

# 경영정보학과와 거리 먼 기관 제외 키워드
_EXCLUDE_ORG_KEYWORDS = [
    "의료", "의학원", "병원", "상담복지", "복지개발",
    "시설관리단", "학교법인", "요양", "재활", "간호",
]

def location_org_search(location: str, top_k: int = 15) -> dict:
    """특정 근무지(지역)의 경영정보학과 관련 기관 목록 집계
    — NCS 대분류 필터 + 의료·복지 기관 제외
    """
    result = {"location": location, "orgs": [], "total_all": 0}
    try:
        with driver.session(database=NEO4J_DB) as s:
            # 전체 기관 수 (필터 없이)
            total = s.run("""
                MATCH (j:JobPosting)
                WHERE j.근무지 CONTAINS $loc
                RETURN count(DISTINCT j.기관명) AS n
            """, loc=location).single()["n"]
            result["total_all"] = total

            # 경영정보 관련 + 의료·복지 제외 필터
            rows = s.run("""
                MATCH (j:JobPosting)-[:HAS_NCS]->(n:NCS)
                WHERE j.근무지 CONTAINS $loc
                  AND n.대분류 IN $ncs_cats
                RETURN j.기관명 AS 기관명, count(DISTINCT j) AS 공고수
                ORDER BY 공고수 DESC
                LIMIT 40
            """, loc=location, ncs_cats=_MIS_NCS_CATEGORIES).data()

            filtered = [
                r for r in rows
                if not any(kw in r["기관명"] for kw in _EXCLUDE_ORG_KEYWORDS)
            ]
            result["orgs"] = filtered[:top_k]
    except Exception:
        pass
    return result

def common_prep_search() -> dict:
    """잡알리오 전체 공고 중 경영정보학과 관련 NCS 직무 집계
    — '공기업/공공기관 공통 준비' 질문 전용
    필터: NCS 대분류 IN [경영·회계·사무, 사업관리, 정보통신, 금융·보험]
    """
    result = {"total": 0, "certifications": [], "toeic": [], "majors": [], "ncs": []}
    try:
        with driver.session(database=NEO4J_DB) as s:
            NCS_CATS = _MIS_NCS_CATEGORIES

            # 전체 공고 수
            total = s.run("""
                MATCH (j:JobPosting)-[:HAS_NCS]->(n:NCS)
                WHERE n.대분류 IN $ncs_cats
                RETURN count(DISTINCT j) AS n
            """, ncs_cats=NCS_CATS).single()["n"]
            result["total"] = total

            # 자격증 빈도 (전문직·사회적 배려 자격증 제외)
            EXCLUDE_CERTS = {
                # 사회적 배려 대상 관련
                "장애인", "보훈", "취업보호대상", "저소득", "다문화",
                "중장년", "지역인재", "청년", "경력단절", "북한이탈주민",
                # 의료·보건 전문직
                "간호사", "사회복지사", "임상심리사", "의사", "약사",
                "물리치료사", "방사선사", "임상병리사", "치과위생사",
                # 법률·회계·노무 전문직
                "변호사", "법무사", "CPA", "공인회계사", "세무사",
                "공인노무사", "행정사", "감정평가사", "관세사",
                # 건설·토목 전문직
                "건축사", "건축기사", "토목기사", "측량기사",
            }
            cert_rows = s.run("""
                MATCH (j:JobPosting)-[:HAS_NCS]->(n:NCS)
                WHERE n.대분류 IN $ncs_cats
                  AND j.req_certifications IS NOT NULL
                WITH DISTINCT j
                UNWIND j.req_certifications AS cert
                WITH cert, count(DISTINCT j) AS 건수
                WHERE 건수 >= 2
                RETURN cert, 건수
                ORDER BY 건수 DESC
                LIMIT 20
            """, ncs_cats=NCS_CATS).data()
            result["certifications"] = [
                r for r in cert_rows if r["cert"] not in EXCLUDE_CERTS
            ][:10]

            # 토익 분포
            toeic_rows = s.run("""
                MATCH (j:JobPosting)-[:HAS_NCS]->(n:NCS)
                WHERE n.대분류 IN $ncs_cats
                  AND j.req_toeic_min IS NOT NULL
                RETURN j.req_toeic_min AS 점수, count(DISTINCT j) AS 건수
                ORDER BY 건수 DESC
                LIMIT 8
            """, ncs_cats=NCS_CATS).data()
            result["toeic"] = toeic_rows

            # 전공 요건
            major_rows = s.run("""
                MATCH (j:JobPosting)-[:HAS_NCS]->(n:NCS)
                WHERE n.대분류 IN $ncs_cats
                  AND j.req_major IS NOT NULL
                RETURN j.req_major AS 전공, count(DISTINCT j) AS 건수
                ORDER BY 건수 DESC
                LIMIT 8
            """, ncs_cats=NCS_CATS).data()
            result["majors"] = major_rows

            # NCS 대분류 분포
            ncs_rows = s.run("""
                MATCH (j:JobPosting)-[:HAS_NCS]->(n:NCS)
                WHERE n.대분류 IN $ncs_cats
                WITH n.대분류 AS 대분류, count(DISTINCT j) AS 건수
                ORDER BY 건수 DESC
                RETURN 대분류, 건수
            """, ncs_cats=NCS_CATS).data()
            result["ncs"] = ncs_rows

    except Exception:
        pass
    return result


# ── Step 3: 최종 답변 ─────────────────────────────────────────────
def generate_answer(question: str, route: str,
                    posting_data: dict, experience_data: list,
                    history: list[dict] | None = None) -> str:
    graph_rows  = posting_data.get("graph", [])
    chunk_rows  = posting_data.get("chunks", [])

    context = f"사용자 질문: {question}\n\n"
    has_data = False

    if route == "common_prep":
        cp = posting_data.get("common_prep", {})
        total = cp.get("total", 0)
        context += (
            f"[공공기관·공기업 경영/IT/금융 직무 채용공고 {total}건 집계 결과]\n"
            "(필터 기준: NCS 직무가 경영·회계·사무 / 사업관리 / 정보통신 / 금융·보험 중 하나인 공고 전체)\n\n"
        )
        if cp.get("certifications"):
            context += "자격증 요건 빈도 (상위 순):\n"
            for r in cp["certifications"]:
                context += f"  - {r['cert']}: {r['건수']}건\n"
        if cp.get("toeic"):
            context += "어학(TOEIC) 요건 빈도:\n"
            for r in cp["toeic"]:
                context += f"  - {r['점수']}점 이상: {r['건수']}건\n"
        if cp.get("majors"):
            context += "전공 요건 빈도:\n"
            for r in cp["majors"]:
                context += f"  - {r['전공']}: {r['건수']}건\n"
        if cp.get("ncs"):
            context += "NCS 직무 대분류 분포:\n"
            for r in cp["ncs"]:
                context += f"  - {r['대분류']}: {r['건수']}건\n"
        context += (
            "\n[답변 지시] 위 집계 수치를 바탕으로 경영정보학과 학생에게 맞는 공기업 취업 준비 방법을 안내하세요.\n"
            f"- '2021~2026년 2월 기준 공공기관·공기업 경영·IT·금융 직무 채용공고 {total}건을 분석하면...' 형태로 시작\n"
            "- 자격증·어학·전공 순서로 '가장 많이 요구된 것부터' 구체적인 건수와 함께 안내\n"
            "- NCS 직무는 '어떤 직무군에 공고가 많은지' 맥락으로 활용\n"
            "- 건수를 구체적으로 언급 (예: 'N건의 공고에서 요구')\n"
            "- 간호사, 사회복지사 등 경영정보학과와 무관한 전문직 자격증은 언급하지 말 것\n"
            "- 경영정보학과 학생이 실제로 준비할 수 있는 자격증(컴퓨터활용능력, 정보처리기사, 한국사 등)을 강조\n"
        )
        has_data = True

    if route == "posting":
        # 집계 보조 데이터가 있으면 먼저 context에 추가 (GPT가 수치 기반으로 답변하도록)
        aggregate = posting_data.get("aggregate")
        cert_freq = posting_data.get("cert_freq", [])
        toeic_freq = posting_data.get("toeic_freq", [])

        if cert_freq or toeic_freq or aggregate:
            context += "[전체 공고 집계 통계 — LIMIT 없이 전수 집계]\n"
            if aggregate:
                context += json.dumps(aggregate, ensure_ascii=False) + "\n"
            if cert_freq:
                context += "자격증 요건 빈도 (많이 요구된 순):\n"
                for r in cert_freq:
                    context += f"  - {r['cert']}: {r['건수']}건\n"
            if toeic_freq:
                context += "TOEIC 점수 요건 분포:\n"
                for r in toeic_freq:
                    context += f"  - {r['점수']}점 이상: {r['건수']}건\n"
            context += (
                "\n[답변 지시]\n"
                "- 자격증·어학 빈도 수치가 있으면 반드시 '가장 많이 요구된 것부터' 순서대로 직접 나열할 것\n"
                "- 공고 목록을 나열하지 말고, 집계 수치를 근거로 '○○ 자격증이 N건 공고에서 요구됩니다'처럼 답할 것\n"
                "- 샘플 공고는 참고용으로만 활용하고 본문에 직접 인용하지 말 것\n\n"
            )
            has_data = True
        if graph_rows:
            # 행 수 제한 + 전처리 (req_major null→전공 무관, req_preferred 사회적 배려 항목 제거)
            graph_limited = [_preprocess_posting_row(r) for r in graph_rows[:8]]

            # NCS null 서브필드 제거 + 레벨 감지 → GPT가 잘못된 레벨명 쓰는 것 방지
            _NCS_SUB_KEYS = ("세분류", "n.세분류", "소분류", "n.소분류", "중분류", "n.중분류")
            _NCS_TOP_KEYS = ("대분류", "n.대분류")
            # null인 NCS 서브필드를 아예 제거 (GPT가 필드명을 보고 오용하지 못하도록)
            graph_limited = [
                {k: v for k, v in row.items()
                 if not (k in _NCS_SUB_KEYS and not v)}
                for row in graph_limited
            ]
            _has_세분류 = any(r.get("세분류") or r.get("n.세분류") for r in graph_limited)
            _has_소분류 = any(r.get("소분류") or r.get("n.소분류") for r in graph_limited)
            _has_중분류 = any(r.get("중분류") or r.get("n.중분류") for r in graph_limited)
            _has_대분류 = any(r.get("대분류") or r.get("n.대분류") for r in graph_limited)
            if _has_대분류 and not _has_세분류 and not _has_소분류 and not _has_중분류:
                context += "[NCS 레벨 안내] 이 데이터는 NCS 대분류만 존재합니다. 답변에서 반드시 '대분류 기준'으로만 표현하고 '세분류', '소분류', '중분류' 표현은 절대 사용하지 말 것.\n\n"

            context += f"[채용공고 구조 검색 결과 (샘플 {len(graph_limited)}건)]\n{json.dumps(graph_limited, ensure_ascii=False)}\n\n"
            has_data = True
        if chunk_rows:
            # chunk 내용이 길 수 있으므로 text 필드는 300자로 제한
            chunk_limited = []
            for r in chunk_rows[:5]:
                row = dict(r)
                if isinstance(row.get("내용"), str):
                    row["내용"] = row["내용"][:300]
                chunk_limited.append(row)
            context += f"[채용공고 본문 검색 결과]\n{json.dumps(chunk_limited, ensure_ascii=False)}\n\n"
            has_data = True

    if experience_data:
        # 경험 게시글은 본문을 더 길게 포함 (GPT가 근거로 쓸 수 있도록)
        exp_context = []
        platforms_used = set()
        for r in experience_data:
            platform = r.get("플랫폼") or ""
            if platform:
                platforms_used.add(platform)
            entry = {
                "제목": r.get("제목", ""),
                "카테고리": r.get("카테고리", ""),
                "플랫폼": platform,
                "유사도점수": round(r.get("score", 0), 3),
                "본문요약": (r.get("본문") or "")[:500],
                "채택댓글및상위댓글": [
                    {
                        "내용": (c.get("내용") or "")[:400],
                        "직무": c.get("직무", ""),
                        "채택여부": c.get("채택", False),
                        "좋아요": c.get("좋아요", 0),
                    }
                    for c in (r.get("상위댓글") or [])
                    if c.get("내용")
                ]
            }
            exp_context.append(entry)
        platform_str = "·".join(sorted(platforms_used)) if platforms_used else "링커리어·잡코리아"
        context += f"[유사 경험 게시글 — 출처: {platform_str} (유사도 {QAPOST_SCORE_THRESHOLD} 이상만 포함)]\n"
        context += json.dumps(exp_context, ensure_ascii=False)
        context += "\n\n"
        has_data = True

    if not has_data:
        context += "[검색 결과 없음 — 관련 데이터를 찾지 못했습니다]\n"



    # 경험 게시글 경로일 때 창작 금지 + 프레이밍 지시를 context에도 명시
    if route == "experience":
        context += (
            "\n\n[답변 방식 지시]\n"
            "답변 첫 문장은 반드시 다음 형태로 시작할 것:\n"
            "  '(질문에 언급된 기관/분야)을 준비하는 취준생들이 링커리어·잡코리아에 올린 고민 글을 보면,'\n"
            "단, 기관명이 질문에 없으면 '비슷한 고민을 하는 취준생들이 링커리어·잡코리아에 올린 고민 글을 보면,'으로 시작.\n\n"
            "이후 내용:\n"
            "- 위 게시글·댓글을 요약·인용해서 구체적으로 소개할 것\n"
            "- 게시글이 질문과 완전히 일치하지 않아도 유사한 내용이면 반드시 소개\n"
            "- 위 데이터에 없는 내용(일반 취업 조언, 자격증 추천 등)은 절대 추가하지 말 것\n"
            "- 게시글이 한 건도 없을 때만 '관련 게시글을 찾지 못했어요'라고 답할 것"
        )

    # 이전 대화 맥락 포함 (최대 3턴 = 6개 메시지)
    # 어시스턴트 답변은 800자로 축약해서 컨텍스트 폭발 방지
    messages = [{"role": "system", "content": ANSWER_SYSTEM}]
    if history:
        for msg in history[-6:]:
            content = msg["content"]
            if msg["role"] == "assistant" and len(content) > 800:
                content = content[:800] + "...(이하 생략)"
            messages.append({"role": msg["role"], "content": content})
    messages.append({"role": "user", "content": context})

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.3,  # 창작 억제를 위해 낮춤
    )
    return resp.choices[0].message.content


# ── 파이프라인 핵심 함수 (app.py / ask() 공동 사용) ─────────────────
def process_question(question: str,
                     history: list[dict] | None = None,
                     ncs_detail: bool = True) -> tuple[str, dict]:
    """전체 파이프라인 실행 후 (answer, meta) 반환.

    Args:
        question: 현재 사용자 질문
        history:  이전 대화 목록. 각 항목은 {"role": "user"|"assistant", "content": str}

    meta = {
        "route":      "posting" | "experience",
        "reason":     str,
        "normalized": str,
        "graph":      list[dict],
        "chunks":     list[dict],
        "experience": list[dict],
    }
    """
    # 0. 질의 전처리 (이전 대화 맥락으로 지시어 해소)
    rewritten   = rewrite_query(question, history=history)
    normalized  = rewritten["normalized"]
    sub_queries = rewritten["sub_queries"]
    conditions  = rewritten["conditions"]

    # 1. 경로 판단
    route, reason, org_name = route_question(normalized)

    posting_data    = {"cypher": "", "graph": [], "chunks": [], "aggregate": None}
    experience_data: list[dict] = []

    # 2-0a. 동아대 경영정보학과 학생 진로 경로
    if route == "dept_career":
        answer = dept_career_answer(question, history=history)
        meta = {
            "route":      "dept_career",
            "reason":     reason,
            "normalized": normalized,
            "graph":      [],
            "chunks":     [],
            "experience": [],
            "source":     "dept_career",
        }
        return answer, meta

    # 2-0b. 기관 NCS 준비 경로
    if route == "ncs_prep":
        ncs_data = ncs_prep_search(org_name or normalized, detail=ncs_detail)
        # NCS 레벨 자동 감지 → 대분류만 있으면 GPT에 명시적으로 알림
        ncs_jobs = ncs_data.get("ncs_jobs", [])
        _ncs_has_se  = any(r.get("세분류") or r.get("소분류") or r.get("중분류") for r in ncs_jobs)
        _ncs_has_dae = any(r.get("대분류") for r in ncs_jobs)
        _ncs_level_note = ""
        if _ncs_has_dae and not _ncs_has_se:
            _ncs_level_note = "[NCS 레벨 안내] 이 기관의 NCS 데이터는 대분류만 존재합니다. 답변에서 반드시 'NCS 대분류 기준'으로만 표현하고 'NCS 세분류', 'NCS 소분류' 표현은 절대 사용하지 말 것.\n\n"
        context = f"{_ncs_level_note}사용자 질문: {question}\n\n[기관 NCS 데이터]\n{json.dumps(ncs_data, ensure_ascii=False)}"
        messages = [
            {"role": "system", "content": NCS_PREP_ANSWER_SYSTEM},
            {"role": "user",   "content": context}
        ]
        resp = client.chat.completions.create(
            model=CHAT_MODEL, messages=messages, temperature=0.3
        )
        answer = resp.choices[0].message.content
        meta = {
            "route":      "ncs_prep",
            "reason":     reason,
            "normalized": normalized,
            "org_name":   org_name,
            "graph":      ncs_data.get("ncs_jobs", []),
            "chunks":     [],
            "experience": [],
            "source":     "ncs_prep",
        }
        return answer, meta

    # 공기업 관련 질문 여부 감지 (is_public_related 필터 적용 기준)
    # ※ "공기업", "공공기관" 같은 일반 단어는 제외 — 너무 광범위해서 public_only 필터가 역효과
    _PUBLIC_ORG_KEYWORDS = [
        "공사", "공단", "진흥원", "진흥공단", "보증기금", "관리공사",
        "캠코", "신보", "중진공", "TIPA", "고용노동", "항만공사",
        "한국자산관리", "신용보증기금", "중소기업기술정보", "중소벤처기업진흥",
    ]
    _is_public_question = any(kw in normalized for kw in _PUBLIC_ORG_KEYWORDS)

    def _qapost_with_fallback(q: str, top_k: int = 5, public_only: bool = False) -> list[dict]:
        """3단계 폴백:
        1) public_only + 기본 임계값(0.80)
        2) public_only 해제 + 기본 임계값(0.80)
        3) public_only 해제 + 낮은 임계값(0.75) — 광범위한 질문 대응
        """
        results = qapost_rag_search(q, top_k=top_k, public_only=public_only)
        if not results and public_only:
            results = qapost_rag_search(q, top_k=top_k, public_only=False)
        if not results:
            results = qapost_rag_search(q, top_k=top_k, public_only=False, min_score=0.75)
        return results

    # 2-0c-0. 지역별 기관 목록 경로 — "부산 공공기관 다 말해줘" 유형
    _detected_region = next((r for r in _REGION_KEYWORDS if r in normalized), None)
    _is_org_list_query = (
        _detected_region is not None
        and any(kw in normalized for kw in ["기관", "기업", "공기업", "공공기관"])
        and any(kw in normalized for kw in _ORG_LIST_TRIGGERS)
    )
    if _is_org_list_query and _detected_region:
        org_data = location_org_search(_detected_region)
        orgs = org_data.get("orgs", [])
        loc = org_data.get("location", _detected_region)
        total_all = org_data.get("total_all", 0)
        if orgs:
            lines = "\n".join(f"{i+1}. {r['기관명']}" for i, r in enumerate(orgs))
            answer = (
                f"{loc} 소재 공공기관·공기업은 총 {total_all}곳이 있어요.\n"
                f"그 중 경영정보학과 학생이 관심 가질 만한 기관을 채용 공고 수 기준으로 추려봤어요.\n\n"
                f"{lines}\n\n"
                f"(2021~2026년 공고 기준이며, 현재 모집 여부는 각 기관에서 확인하세요)"
            )
        else:
            answer = f"{loc} 소재 관련 기관 정보를 찾지 못했어요."
        return answer, {
            "route": "org_list",
            "reason": f"{loc} 기관 목록 집계",
            "normalized": normalized,
            "graph": [], "chunks": [], "experience": [],
            "source": "posting",
        }

    # 2-0c-1. 공기업 공통 준비 경로 — 특정 기관 없이 "공기업/공공기관 공통 준비" 묻는 질문
    _COMMON_PREP_SECTOR = ["공기업", "공공기관", "공공"]
    _COMMON_PREP_INTENT = ["공통", "준비", "어떻게 준비", "뭘 준비", "무엇을 준비", "미리"]
    _is_common_prep = (
        not _is_public_question
        and any(kw in normalized for kw in _COMMON_PREP_SECTOR)
        and any(kw in normalized for kw in _COMMON_PREP_INTENT)
    )
    if _is_common_prep:
        cp_data = common_prep_search()
        answer = generate_answer(question, "common_prep", {"common_prep": cp_data}, [], history=history)
        return answer, {
            "route":      "common_prep",
            "reason":     "공기업 공통 준비 집계 경로",
            "normalized": normalized,
            "graph":      [],
            "chunks":     [],
            "experience": [],
            "source":     "posting",
        }

    # 2-0c-2. 경험·고민 경로 — 채용공고 검색 생략, 바로 QAPost로
    if route == "experience":
        experience_data = _qapost_with_fallback(normalized, top_k=5, public_only=_is_public_question)
        answer = generate_answer(question, "experience", posting_data, experience_data, history=history)
        meta = {
            "route":      "experience",
            "reason":     reason,
            "normalized": normalized,
            "graph":      [],
            "chunks":     [],
            "experience": experience_data,
            "reranked":   False,
            "source":     "experience",
        }
        return answer, meta

    # 2. 채용공고 경로 (posting)
    if route == "posting":
        cypher, graph_rows = cypher_search(normalized, conditions=conditions)
        posting_data["cypher"] = cypher

        # ── Python 레벨 사후 필터 ──────────────────────────────────────
        # Cypher의 OR/AND 우선순위 오류 등으로 조건이 깨진 결과를 보정
        def _get_field(row: dict, *keys) -> str:
            """row에서 키(별칭 포함) 값을 문자열로 반환"""
            for k in keys:
                if k in row and row[k] is not None:
                    return str(row[k])
            return ""

        # 1) 정규직 필터: 사용자가 '정규직'을 명시했으면 비정규직·기간제 제거
        # ⚠️ "비정규직" CONTAINS "정규직" → 반드시 "비정규직" 제외 체크 필요
        if "정규직" in normalized and graph_rows:
            def _is_regular(row: dict) -> bool:
                고용 = _get_field(row, "고용형태", "j.고용형태")
                return "정규직" in 고용 and "비정규직" not in 고용
            filtered = [r for r in graph_rows if _is_regular(r)]
            if filtered:
                graph_rows = filtered
            else:
                # 결과가 있어도 전부 비정규직 → Cypher 오류로 추정
                # graph_rows를 비워서 아래 retry 로직이 돌도록 유도
                graph_rows = []

        # 2) 특수 전문직 필터: 일반 질문 시 전문 면허 필수 공고 제거
        SPECIAL_LICENSES = {
            "간호사","의사","약사","물리치료사","방사선사","임상병리사",
            "치과","한의사","수의사","공인회계사","CPA","세무사",
            "변호사","법무사","감정평가사","노무사","관세사"
        }
        SPECIAL_KEYWORDS = {
            "간호","의료","보건","약학","법조","회계사","세무사","변호사"
        }
        is_general_query = not any(kw in normalized for kw in SPECIAL_KEYWORDS)
        if is_general_query and graph_rows:
            def _has_special_license(row: dict) -> bool:
                certs = row.get("req_certifications") or row.get("j.req_certifications") or []
                if isinstance(certs, str):
                    certs = [certs]
                return any(any(lic in cert for lic in SPECIAL_LICENSES) for cert in certs)
            filtered2 = [r for r in graph_rows if not _has_special_license(r)]
            if filtered2:
                graph_rows = filtered2

        # 3) Reranker: Cypher가 반환한 후보를 질문 관련성 순으로 재정렬
        #    - LIMIT 20으로 넓게 가져온 결과 중 실제로 관련 높은 8건만 GPT 답변에 사용
        #    - 고용형태·직무·자격 조건 불일치 결과를 아래 순위로 내려 품질 향상
        if len(graph_rows) > 4:
            graph_rows = rerank_results(normalized, graph_rows, top_k=8, mode="graph")

        posting_data["graph"] = graph_rows

        # 5) 단일 기관 쿼리에서 정규직 공채 누락 시 보완 쿼리 ────────────────
        # 직무 키워드 WHERE 필터 때문에 정규직 신입 공채가 빠진 경우 자동 보완.
        # (예: "○○기관 IT직무" → IT전문인력 비정규직만 반환되고 신입 공채 누락)
        if graph_rows:
            _has_regular = any(
                "정규직" in str(_get_field(r, "고용형태", "j.고용형태"))
                and "비정규직" not in str(_get_field(r, "고용형태", "j.고용형태"))
                for r in graph_rows
            )
            if not _has_regular:
                _orgs = {
                    _get_field(r, "기관명", "j.기관명")
                    for r in graph_rows
                    if _get_field(r, "기관명", "j.기관명")
                }
                if len(_orgs) == 1:   # 단일 기관 쿼리일 때만 보완
                    _org_kw = list(_orgs)[0]
                    try:
                        with driver.session(database=NEO4J_DB) as s:
                            supp = s.run("""
                                MATCH (j:JobPosting)
                                WHERE j.기관명 CONTAINS $org AND j.고용형태 = '정규직'
                                RETURN j.채용제목, j.기관명, j.고용형태, j.근무지,
                                       j.등록일, j.마감일, j.상태,
                                       j.req_certifications, j.req_preferred, j.req_major,
                                       j.req_toeic_min, j.req_career, j.req_english_tests
                                ORDER BY j.등록일 DESC
                                LIMIT 4
                            """, org=_org_kw).data()
                            supp = [
                                {k: str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v
                                 for k, v in row.items()}
                                for row in supp
                            ]
                        if supp:
                            graph_rows = supp + graph_rows   # 정규직 공채를 맨 앞에
                            posting_data["graph"] = graph_rows
                    except Exception:
                        pass

        # 4) 집계형 질문 감지 → 보조 집계 Cypher 추가 실행
        # "요구하는게 뭐야", "공통 자격은" 같은 질문은 LIMIT 10 샘플만으론 부정확
        # → 어학·자격증·경력 분포를 COUNT로 집계해서 context에 추가
        AGGREGATE_TRIGGERS = {
            "요구하는", "요구사항", "요건이", "조건이", "필요한", "필요조건",
            "공통적으로", "어떤 자격", "어떤 조건", "어학", "자격증", "우대",
            "뭐가 필요", "뭐 필요", "어떤게 필요", "어떤 게 필요",
        }
        # normalized 또는 원본 question 어느 쪽에서든 트리거 감지
        is_aggregate_q = (
            any(kw in normalized for kw in AGGREGATE_TRIGGERS)
            or any(kw in question for kw in AGGREGATE_TRIGGERS)
        )
        if is_aggregate_q and graph_rows:
            # NCS 대분류 키워드를 질문 또는 graph_rows에서 추출
            NCS_대분류_LIST = [
                "사업관리", "경영·회계·사무", "경영회계사무", "금융·보험", "금융보험",
                "교육·자연·사회과학", "법률·경찰·소방", "보건·의료", "보건의료",
                "사회복지·종교", "문화·예술·디자인", "운전·운송", "영업판매",
                "경비·청소", "이용·숙박·여행", "음식서비스", "건설", "기계",
                "재료", "화학·바이오", "섬유·의복", "전기·전자", "정보통신",
                "식품가공", "환경·에너지·안전", "농림어업",
            ]
            ncs_kw = None
            # 1) 질문 텍스트에서 찾기
            for 분류 in NCS_대분류_LIST:
                if 분류 in normalized or 분류.replace("·", "") in normalized:
                    ncs_kw = 분류.split("·")[0]  # "경영" 등 첫 단어로 CONTAINS 검색
                    break
            # 2) graph_rows에서 찾기 (fallback)
            if not ncs_kw:
                for row in graph_rows[:3]:
                    for k in ["대분류", "n.대분류"]:
                        if k in row and row[k]:
                            ncs_kw = str(row[k]).split("·")[0]
                            break
                    if ncs_kw:
                        break
            if ncs_kw:
                try:
                    with driver.session(database=NEO4J_DB) as s:
                        agg = s.run("""
                            MATCH (j:JobPosting)-[:HAS_NCS]->(n:NCS)
                            WHERE n.대분류 CONTAINS $kw
                            RETURN
                                count(j) AS 전체공고수,
                                count(CASE WHEN j.req_toeic_min IS NOT NULL THEN 1 END) AS 토익요구,
                                count(CASE WHEN j.req_english_tests IS NOT NULL
                                           AND size(j.req_english_tests) > 0 THEN 1 END) AS 어학요구,
                                count(CASE WHEN j.req_certifications IS NOT NULL
                                           AND size(j.req_certifications) > 0 THEN 1 END) AS 자격증요구,
                                count(CASE WHEN j.req_career = '신입' THEN 1 END) AS 신입가능,
                                count(CASE WHEN j.고용형태 = '정규직' THEN 1 END) AS 정규직수,
                                avg(j.req_toeic_min) AS 평균토익점수
                        """, kw=ncs_kw).single()
                        if agg:
                            posting_data["aggregate"] = dict(agg)

                        # 자격증 빈도 집계 — "필요한 자격증이 뭐야" 유형 질문용
                        cert_agg = s.run("""
                            MATCH (j:JobPosting)-[:HAS_NCS]->(n:NCS)
                            WHERE n.대분류 CONTAINS $kw
                              AND j.req_certifications IS NOT NULL
                            UNWIND j.req_certifications AS cert
                            WITH cert, count(DISTINCT j) AS 건수
                            WHERE 건수 >= 2
                            RETURN cert, 건수
                            ORDER BY 건수 DESC
                            LIMIT 10
                        """, kw=ncs_kw).data()
                        if cert_agg:
                            posting_data["cert_freq"] = cert_agg

                        # 토익 점수대 분포
                        toeic_agg = s.run("""
                            MATCH (j:JobPosting)-[:HAS_NCS]->(n:NCS)
                            WHERE n.대분류 CONTAINS $kw
                              AND j.req_toeic_min IS NOT NULL
                            RETURN j.req_toeic_min AS 점수, count(j) AS 건수
                            ORDER BY 건수 DESC
                            LIMIT 6
                        """, kw=ncs_kw).data()
                        if toeic_agg:
                            posting_data["toeic_freq"] = toeic_agg

                except Exception:
                    pass

        # 0건이면 조건 넓혀서 재시도
        if not graph_rows:
            retry_prompt = (
                f"{normalized}\n\n"
                "주의: 이전 검색 결과가 0건이었습니다(OR/AND 우선순위 오류 가능성 있음). "
                "반드시 OR 조건 전체를 괄호로 묶을 것: WHERE (A OR B OR C) AND D AND E\n"
                "기관명 조건을 더 넓게 OR로 확장하거나, NCS 직무 분류로 검색하는 Cypher를 새로 작성하세요."
            )
            cypher2, graph_rows2 = cypher_search(retry_prompt, conditions=conditions)
            if graph_rows2:
                posting_data["cypher"] = cypher2
                posting_data["graph"]  = graph_rows2
                graph_rows = graph_rows2

        # 기관명 힌트 추출 — field key 직접 참조 (값 내용 기반 추측 X)
        org_hint = None
        for row in graph_rows[:3]:
            for key in ["기관", "기관명", "org", "기관이름"]:
                if key in row and isinstance(row[key], str) and len(row[key]) > 1:
                    org_hint = row[key]
                    break
            if org_hint:
                break

        # Multi-Query 벡터 검색
        chunk_rows = chunk_rag_search_multi(sub_queries, top_k=5)
        if not chunk_rows and org_hint:
            chunk_rows = chunk_rag_search(normalized, org_name=org_hint, top_k=5)

        # 벡터 검색 실패 시 키워드 폴백
        if not chunk_rows:
            chunk_rows = _keyword_fallback(normalized)

        # 폴백 결과 관련성 검증 — 질문 핵심 키워드가 하나도 없으면 버림
        if chunk_rows and not graph_rows:
            core_words = [w for w in normalized.replace("?", "").split()
                          if len(w) >= 2 and w not in {"관련", "공고", "알려줘", "있어", "뭐야", "어떻게", "보여줘", "알고", "싶어", "채용"}]
            if core_words:
                filtered = [
                    r for r in chunk_rows
                    if any(kw in (r.get("text", "") or "") for kw in core_words)
                ]
                # 필터링 후 절반 이상 날아가면 원본 유지, 전부 날아가면 빈 리스트
                chunk_rows = filtered if filtered else []

        posting_data["chunks"] = chunk_rows

    # 3. posting 결과 없을 때 QAPost 자동 폴백
    has_posting_result = bool(posting_data.get("graph")) or bool(posting_data.get("chunks"))
    if not has_posting_result:
        # 채용공고로 답할 수 없는 경우 → Q&A 게시글 댓글로 대체
        experience_data = _qapost_with_fallback(normalized, top_k=5, public_only=_is_public_question)
        effective_route = "experience" if experience_data else "posting"
    else:
        experience_data = []
        effective_route = "posting"

    # 4. 최종 답변 (이전 대화 맥락 포함)
    answer = generate_answer(question, effective_route, posting_data, experience_data, history=history)

    meta = {
        "route":      effective_route,
        "reason":     reason,
        "normalized": normalized,
        "graph":      posting_data["graph"],
        "chunks":     posting_data["chunks"],
        "experience": experience_data,
        "reranked":   True,   # posting 경로는 항상 Reranker 적용
        "source":     effective_route,   # "posting" or "experience"
    }
    return answer, meta


# ── 메인 함수 ─────────────────────────────────────────────────────
def ask(question: str, verbose: bool = True) -> str:
    print(f"\n{'='*60}")
    print(f"질문: {question}")
    print('='*60)

    answer, meta = process_question(question)

    if verbose:
        route = meta["route"]
        print(f"\n[질의 정규화] {meta['normalized']}")
        route_label = {
            "posting":     "채용공고",
            "experience":  "경험 게시글",
            "ncs_prep":    "기관 NCS 준비",
            "dept_career": "동아대 경영정보학과 진로",
        }.get(route, route)
        print(f"[경로] {route_label} — {meta['reason']}")
        reranked_tag = " (Reranker 적용)" if meta.get("reranked") else ""
        print(f"[구조 검색] {len(meta['graph'])}건{reranked_tag}")
        chunk_rows = meta["chunks"]
        mode = "벡터+Reranker" if chunk_rows and "score" in chunk_rows[0] else "키워드"
        print(f"[Chunk 검색({mode})] {len(chunk_rows)}건")
        if meta["experience"]:
            print(f"[경험 게시글] {len(meta['experience'])}건")
            for r in meta["experience"]:
                print(f"  (score={r.get('score', 0):.3f}) {r.get('제목', '')[:50]}")

    print(f"\n[답변]\n{answer}")
    return answer


# ── 대화형 루프 ───────────────────────────────────────────────────
def chat():
    print("=" * 60)
    print("취준생 Q&A 서비스 (종료: q)")
    print("=" * 60)
    while True:
        try:
            sys.stdout.write("\n질문: ")
            sys.stdout.flush()
            question = sys.stdin.readline().strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in ("q", "quit", "종료", "exit"):
            break
        if not question:
            continue
        ask(question, verbose=True)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        ask(" ".join(sys.argv[1:]), verbose=True)
    else:
        chat()
    driver.close()
