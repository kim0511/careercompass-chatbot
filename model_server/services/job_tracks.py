from db.neo4j import get_session

JOB_TRACKS = [
    {
        "id": "data_analyst", "name": "데이터 분석가", "color": "blue",
        "module": "데이터사이언스모듈",
        "description": "데이터에서 비즈니스 인사이트를 도출해 의사결정을 지원하는 직무. SQL·통계·시각화·Python 역량이 핵심.",
        "cert_names": ["데이터분석준전문가(ADsP)", "SQL개발자(SQLD)", "빅데이터분석기사", "사회조사분석사"],
        "activity_names": ["데이터사이언스 기초과정", "SW중심대학", "생성형 AI 마스터 클래스", "경영정보학과 실무자 초청 특강과 취업 준비 전략 특강"],
        "ncs_name": "데이터분석및데이터사이언스",
        "chatbot_q": "데이터 분석가 채용공고 알려줘",
        "tip": "2학년에 Python·SQL 기초를 다지고, 3학년부터 ADsP → SQLD 순서로 자격증을 취득하세요.",
    },
    {
        "id": "ai_engineer", "name": "AI·머신러닝 엔지니어", "color": "blue",
        "module": "데이터사이언스모듈",
        "description": "머신러닝·딥러닝 모델을 개발·운영하는 직무. Python 심화·수학적 모델링·AI 서비스 구현 능력이 중요.",
        "cert_names": ["정보처리기사", "빅데이터분석기사", "데이터분석준전문가(ADsP)"],
        "activity_names": ["생성형 AI 마스터 클래스", "프롬프트 엔지니어링 기초 이해와 생성형 AI활용 사례분석 활동", "데이터사이언스 기초과정", "SW중심대학"],
        "ncs_name": "인공지능 및 소프트웨어개발",
        "chatbot_q": "AI 머신러닝 엔지니어 채용공고 알려줘",
        "tip": "머신러닝Ⅰ·Ⅱ 이후 Kaggle 등 공모전에 참가하면 포트폴리오가 됩니다. 정보처리기사는 3학년 목표로 준비하세요.",
    },
    {
        "id": "erp_consultant", "name": "ERP·IT 컨설턴트", "color": "teal",
        "module": "디지털비즈니스모듈",
        "description": "기업 ERP 시스템을 구축·운영하거나 IT 전략을 자문하는 직무. SAP·ERP 실무 경험과 비즈니스 이해가 핵심.",
        "cert_names": ["ERP정보관리사", "SAP 인증시험(ABAP, FI, CO, MM, PP)", "정보처리기사"],
        "activity_names": ["경영정보학과 실무자 초청 특강과 취업 준비 전략 특강", "Dong-A Frontiers", "Dong-A Leaders", "재직선배 초청 교육"],
        "ncs_name": "경영정보시스템 및 IT컨설팅",
        "chatbot_q": "ERP IT 컨설턴트 채용공고 알려줘",
        "tip": "ERP 과목(3학년)과 ERP실무(4학년)를 이수한 뒤 ERP정보관리사 취득을 추천합니다. SAP 실습 경험을 쌓으면 차별화됩니다.",
    },
    {
        "id": "digital_pm", "name": "디지털 전략·기획 PM", "color": "teal",
        "module": "디지털비즈니스모듈",
        "description": "디지털 전환 전략을 수립하고 IT 서비스 개발 프로젝트를 기획·관리하는 직무.",
        "cert_names": ["정보처리기사", "컴퓨터활용능력시험", "사무자동화 산업기사"],
        "activity_names": ["취·창업 역량개발 프로그램", "경영정보학과 실무자 초청 특강과 취업 준비 전략 특강", "진로동아리 Career Design", "진로또래멘토링 리드다움"],
        "ncs_name": "창업및디지털비즈니스기획",
        "chatbot_q": "디지털 기획 PM 채용공고 알려줘",
        "tip": "서비스기획·UX/UI 과목(2학년)과 프로젝트관리(4학년)를 연계해서 이수하면 PM 역량을 체계적으로 쌓을 수 있습니다.",
    },
    {
        "id": "public_admin", "name": "공공기관·행정정보", "color": "teal",
        "module": "디지털비즈니스모듈",
        "description": "공공기관의 정보시스템 운영·디지털 행정을 담당하는 직무. NCS 기반 공채 준비가 핵심.",
        "cert_names": ["컴퓨터활용능력시험", "사무자동화 산업기사", "워드프로세서", "ITQ"],
        "activity_names": ["Dong-A Frontiers", "Dong-A Leaders", "진로심리검사 및 해석 상담", "지역연계 진로멘토링"],
        "ncs_name": "공공기관및행정정보",
        "chatbot_q": "공공기관 행정정보 채용공고 알려줘",
        "tip": "컴퓨터활용능력 1급은 필수로 취득하고, NCS 직업기초능력 학습을 병행하세요. Dong-A Frontiers 프로그램이 공채 준비에 도움됩니다.",
    },
    {
        "id": "fintech", "name": "핀테크·금융 IT", "color": "teal",
        "module": "디지털비즈니스모듈",
        "description": "디지털 금융 서비스 기획, 금융 데이터 분석, 핀테크 전략을 담당하는 직무.",
        "cert_names": ["SQL개발자(SQLD)", "데이터분석준전문가(ADsP)", "ERP정보관리사"],
        "activity_names": ["생성형 AI를 활용한 판테크기반 기술의 실무 적용 사례", "디지털부산아카데미", "경영정보학과 실무자 초청 특강과 취업 준비 전략 특강"],
        "ncs_name": "금융IT및핀테크",
        "chatbot_q": "핀테크 금융IT 채용공고 알려줘",
        "tip": "DB모듈의 ERP·SCM 과목을 이수하면서 SQLD를 취득하면 금융권 IT 직무 지원 시 유리합니다.",
    },
]


def get_job_tracks_data() -> list:
    result = []
    try:
        with get_session() as s:
            for jt in JOB_TRACKS:
                by_year = {}
                for yr in [2, 3, 4]:
                    courses = s.run("""
                        MATCH (cm:CareerModule {name: $mod})-[:COVERS {year: $yr}]->(c:Course)
                        RETURN c.name AS name ORDER BY c.code
                    """, mod=jt["module"], yr=yr).data()
                    if courses:
                        by_year[yr] = [c["name"] for c in courses]

                certs = []
                for cert_name in jt["cert_names"]:
                    row = s.run("""
                        MATCH (d:Department {name: '경영정보학과'})-[:HAS_CERT]->(q:Qualification {name: $name})
                        RETURN q.name AS name
                    """, name=cert_name).single()
                    if row:
                        certs.append(row["name"])

                activities = []
                for act_name in jt["activity_names"]:
                    row = s.run("""
                        MATCH (d:Department {name: '경영정보학과'})-[:HAS_ACTIVITY]->(ea:ExtraActivity {name: $name})
                        RETURN ea.name AS name, ea.org AS org
                    """, name=act_name).single()
                    if row:
                        activities.append({"name": row["name"], "org": row["org"] or ""})

                ncs_row = s.run("""
                    MATCH (cm:CareerModule {name: $mod})-[:HAS_NCS]->(nf:NCSField {name: $ncs})
                    RETURN nf.name AS name, nf.ncs_path AS path, nf.details AS details
                """, mod=jt["module"], ncs=jt["ncs_name"]).single()

                result.append({
                    **jt,
                    "by_year":    by_year,
                    "certs":      certs,
                    "activities": activities,
                    "ncs":        dict(ncs_row) if ncs_row else None,
                })
    except Exception:
        pass
    return result
