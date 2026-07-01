import json
import os
import threading
import webbrowser
from flask import Flask, render_template, request, jsonify, Response
from neo4j import GraphDatabase
from qa_pipeline import process_question

# ── Neo4j 연결 ────────────────────────────────────────────────────
_NEO4J_URI  = os.environ.get("NEO4J_URI",  "bolt://127.0.0.1:7687")
_NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
_NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]
_NEO4J_AUTH = (_NEO4J_USER, _NEO4J_PASSWORD)
_NEO4J_DB   = os.environ.get("NEO4J_DB",  "neo4j")
_neo4j_driver = GraphDatabase.driver(_NEO4J_URI, auth=_NEO4J_AUTH)


def get_curriculum_data() -> dict:
    """Neo4j에서 커리큘럼 페이지용 학과 데이터 조회"""
    data = {
        "dept": {},
        "modules": [],
        "courses_by_category": {},
        "qualifications": [],
        "skills": [],
        "activities": [],
        "companies_by_sector": {},
        "employ_stats": {},
    }
    try:
        with _neo4j_driver.session(database=_NEO4J_DB) as s:

            # 1) Department 기본정보
            dept = s.run("""
                MATCH (d:Department {name: '경영정보학과'})
                RETURN d.dept_intro AS intro, d.dean_message AS dean,
                       d.edu_goal_ds AS goal_ds, d.edu_goal_db AS goal_db,
                       d.employ_rate_2020 AS r20, d.employ_rate_2021 AS r21,
                       d.employ_rate_2022 AS r22, d.employ_rate_2023 AS r23,
                       d.employ_rate_2024 AS r24,
                       d.employ_sector_service AS s_svc,
                       d.employ_sector_it AS s_it,
                       d.employ_sector_mfg AS s_mfg,
                       d.employ_sector_public AS s_pub,
                       d.employ_sector_other AS s_etc,
                       d.credit_req_core AS req_core,
                       d.credit_req_module AS req_mod
            """).single()
            if dept:
                data["dept"] = dict(dept)
                data["employ_stats"] = {
                    "rates": [
                        {"year": 2020, "rate": dept["r20"]},
                        {"year": 2021, "rate": dept["r21"]},
                        {"year": 2022, "rate": dept["r22"]},
                        {"year": 2023, "rate": dept["r23"]},
                        {"year": 2024, "rate": dept["r24"]},
                    ],
                    "sectors": [
                        {"name": "서비스업(은행·유통 등)", "pct": dept["s_svc"]},
                        {"name": "정보통신",              "pct": dept["s_it"]},
                        {"name": "제조업",                "pct": dept["s_mfg"]},
                        {"name": "공기업·공무원",          "pct": dept["s_pub"]},
                        {"name": "기타(진학 등)",          "pct": dept["s_etc"]},
                    ],
                }

            # 2) CareerModule (연도별 과목 + 진출분야 포함)
            modules_raw = s.run("""
                MATCH (d:Department {name: '경영정보학과'})-[:HAS_MODULE]->(cm:CareerModule)
                RETURN cm.name AS name, cm.description AS desc
                ORDER BY cm.name
            """).data()

            for mod in modules_raw:
                mod_name = mod["name"]
                # 연도별 과목
                by_year = {}
                for yr in [2, 3, 4]:
                    yr_courses = s.run("""
                        MATCH (cm:CareerModule {name: $n})-[:COVERS {year: $yr}]->(c:Course)
                        RETURN c.name AS name, c.code AS code
                        ORDER BY c.semester, c.code
                    """, n=mod_name, yr=yr).data()
                    if yr_courses:
                        by_year[yr] = yr_courses
                # 진출분야
                fields = s.run("""
                    MATCH (cm:CareerModule {name: $n})-[r:LEADS_TO]->(cf:CareerField)
                    RETURN cf.name AS field, r.description AS desc
                """, n=mod_name).data()
                # NCS 직무 분류
                ncs_list = s.run("""
                    MATCH (cm:CareerModule {name: $n})-[:HAS_NCS]->(nf:NCSField)
                    RETURN nf.name AS name, nf.ncs_path AS ncs_path, nf.details AS details
                    ORDER BY nf.name
                """, n=mod_name).data()
                data["modules"].append({
                    "name":    mod_name,
                    "desc":    mod["desc"],
                    "by_year": by_year,
                    "fields":  fields,
                    "ncs":     ncs_list,
                })

            # 3) 과목 (카테고리별)
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
                    data["courses_by_category"][cat] = courses

            # 4) 자격증
            data["qualifications"] = s.run("""
                MATCH (d:Department {name: '경영정보학과'})-[:HAS_CERT]->(q:Qualification)
                RETURN q.name AS name, q.category AS cat
                ORDER BY q.name
            """).data()

            # 5) 직무능력
            data["skills"] = [r["name"] for r in s.run("""
                MATCH (d:Department {name: '경영정보학과'})-[:HAS_SKILL]->(sk:CoreSkill)
                RETURN sk.name AS name
            """).data()]

            # 6) 비교과·추천활동
            data["activities"] = s.run("""
                MATCH (d:Department {name: '경영정보학과'})-[:HAS_ACTIVITY]->(ea:ExtraActivity)
                RETURN ea.name AS name, ea.category AS cat, ea.org AS org
                ORDER BY ea.category, ea.org
            """).data()

            # 7) 졸업생 취업처 (섹터별)
            companies = s.run("""
                MATCH (d:Department {name: '경영정보학과'})-[:GRAD_EMPLOYED]->(gc:GradCompany)
                RETURN gc.sector AS sector, collect(gc.name) AS names
                ORDER BY gc.sector
            """).data()
            for row in companies:
                data["companies_by_sector"][row["sector"]] = row["names"]

    except Exception as e:
        data["error"] = str(e)
    return data

app = Flask(__name__)

@app.after_request
def add_cors(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    return response

@app.route('/api/<path:subpath>', methods=['OPTIONS'])
def options_handler(subpath):
    return '', 200

# ── 직무별 준비 경로 큐레이션 (JobTrack — 별도 Neo4j 노드 없이 기존 노드 활용) ──
JOB_TRACKS = [
    {
        "id": "data_analyst",
        "name": "데이터 분석가",
        "color": "blue",
        "module": "데이터사이언스모듈",
        "description": "데이터에서 비즈니스 인사이트를 도출해 의사결정을 지원하는 직무. SQL·통계·시각화·Python 역량이 핵심.",
        "cert_names": ["데이터분석준전문가(ADsP)", "SQL개발자(SQLD)", "빅데이터분석기사", "사회조사분석사"],
        "activity_names": ["데이터사이언스 기초과정", "SW중심대학", "생성형 AI 마스터 클래스", "경영정보학과 실무자 초청 특강과 취업 준비 전략 특강"],
        "ncs_name": "데이터분석및데이터사이언스",
        "chatbot_q": "데이터 분석가 채용공고 알려줘",
        "tip": "2학년에 Python·SQL 기초를 다지고, 3학년부터 ADsP → SQLD 순서로 자격증을 취득하세요.",
    },
    {
        "id": "ai_engineer",
        "name": "AI·머신러닝 엔지니어",
        "color": "blue",
        "module": "데이터사이언스모듈",
        "description": "머신러닝·딥러닝 모델을 개발·운영하는 직무. Python 심화·수학적 모델링·AI 서비스 구현 능력이 중요.",
        "cert_names": ["정보처리기사", "빅데이터분석기사", "데이터분석준전문가(ADsP)"],
        "activity_names": ["생성형 AI 마스터 클래스", "프롬프트 엔지니어링 기초 이해와 생성형 AI활용 사례분석 활동", "데이터사이언스 기초과정", "SW중심대학"],
        "ncs_name": "인공지능 및 소프트웨어개발",
        "chatbot_q": "AI 머신러닝 엔지니어 채용공고 알려줘",
        "tip": "머신러닝Ⅰ·Ⅱ 이후 Kaggle 등 공모전에 참가하면 포트폴리오가 됩니다. 정보처리기사는 3학년 목표로 준비하세요.",
    },
    {
        "id": "erp_consultant",
        "name": "ERP·IT 컨설턴트",
        "color": "teal",
        "module": "디지털비즈니스모듈",
        "description": "기업 ERP 시스템을 구축·운영하거나 IT 전략을 자문하는 직무. SAP·ERP 실무 경험과 비즈니스 이해가 핵심.",
        "cert_names": ["ERP정보관리사", "SAP 인증시험(ABAP, FI, CO, MM, PP)", "정보처리기사"],
        "activity_names": ["경영정보학과 실무자 초청 특강과 취업 준비 전략 특강", "Dong-A Frontiers", "Dong-A Leaders", "재직선배 초청 교육"],
        "ncs_name": "경영정보시스템 및 IT컨설팅",
        "chatbot_q": "ERP IT 컨설턴트 채용공고 알려줘",
        "tip": "ERP 과목(3학년)과 ERP실무(4학년)를 이수한 뒤 ERP정보관리사 취득을 추천합니다. SAP 실습 경험을 쌓으면 차별화됩니다.",
    },
    {
        "id": "digital_pm",
        "name": "디지털 전략·기획 PM",
        "color": "teal",
        "module": "디지털비즈니스모듈",
        "description": "디지털 전환 전략을 수립하고 IT 서비스 개발 프로젝트를 기획·관리하는 직무.",
        "cert_names": ["정보처리기사", "컴퓨터활용능력시험", "사무자동화 산업기사"],
        "activity_names": ["취·창업 역량개발 프로그램", "경영정보학과 실무자 초청 특강과 취업 준비 전략 특강", "진로동아리 Career Design", "진로또래멘토링 리드다움"],
        "ncs_name": "창업및디지털비즈니스기획",
        "chatbot_q": "디지털 기획 PM 채용공고 알려줘",
        "tip": "서비스기획·UX/UI 과목(2학년)과 프로젝트관리(4학년)를 연계해서 이수하면 PM 역량을 체계적으로 쌓을 수 있습니다.",
    },
    {
        "id": "public_admin",
        "name": "공공기관·행정정보",
        "color": "teal",
        "module": "디지털비즈니스모듈",
        "description": "공공기관의 정보시스템 운영·디지털 행정을 담당하는 직무. NCS 기반 공채 준비가 핵심.",
        "cert_names": ["컴퓨터활용능력시험", "사무자동화 산업기사", "워드프로세서", "ITQ"],
        "activity_names": ["Dong-A Frontiers", "Dong-A Leaders", "진로심리검사 및 해석 상담", "지역연계 진로멘토링"],
        "ncs_name": "공공기관및행정정보",
        "chatbot_q": "공공기관 행정정보 채용공고 알려줘",
        "tip": "컴퓨터활용능력 1급은 필수로 취득하고, NCS 직업기초능력 학습을 병행하세요. Dong-A Frontiers 프로그램이 공채 준비에 도움됩니다.",
    },
    {
        "id": "fintech",
        "name": "핀테크·금융 IT",
        "color": "teal",
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
    """기존 Neo4j 노드(Course·Qualification·ExtraActivity·NCSField)를 조합해 직무별 준비 경로 반환"""
    result = []
    try:
        with _neo4j_driver.session(database=_NEO4J_DB) as s:
            for jt in JOB_TRACKS:
                # 1) 모듈 과목 (학년별, COVERS 관계)
                by_year = {}
                for yr in [2, 3, 4]:
                    courses = s.run("""
                        MATCH (cm:CareerModule {name: $mod})-[:COVERS {year: $yr}]->(c:Course)
                        RETURN c.name AS name
                        ORDER BY c.code
                    """, mod=jt["module"], yr=yr).data()
                    if courses:
                        by_year[yr] = [c["name"] for c in courses]

                # 2) 자격증 (Qualification 노드에서 이름 매칭)
                certs = []
                for cert_name in jt["cert_names"]:
                    row = s.run("""
                        MATCH (d:Department {name: '경영정보학과'})-[:HAS_CERT]->(q:Qualification {name: $name})
                        RETURN q.name AS name
                    """, name=cert_name).single()
                    if row:
                        certs.append(row["name"])

                # 3) 추천 활동 (ExtraActivity 노드에서 이름 매칭)
                activities = []
                for act_name in jt["activity_names"]:
                    row = s.run("""
                        MATCH (d:Department {name: '경영정보학과'})-[:HAS_ACTIVITY]->(ea:ExtraActivity {name: $name})
                        RETURN ea.name AS name, ea.org AS org
                    """, name=act_name).single()
                    if row:
                        activities.append({"name": row["name"], "org": row["org"] or ""})

                # 4) NCS 정보 (NCSField 노드 매칭)
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
    except Exception as e:
        pass
    return result


ORGS_FILE = os.path.join(os.path.dirname(__file__), "orgs.json")

def load_orgs():
    try:
        with open(ORGS_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return [o for o in data if isinstance(o, str) and o.strip()]
    except Exception:
        return []

# ── 부산 소재 공공기관 (동아대 경영정보학과 졸업생 진출 이력 기반, Neo4j GradCompany 확인)
# 가장 추천: 한국자산관리공사(캠코) — 본사 부산 혁신도시, IT·데이터 직무 명확, 경영정보학과 최적합
FEATURED_ORGS = [
    "한국자산관리공사",   # ★ 추천 — 본사 부산 혁신도시, IT/데이터 직무
    "신용보증기금",       # 본사 부산 이전, 금융IT/경영정보 직무
    "중소기업기술정보진흥원",  # 부산 소재, IT·경영 지원 직무
    "중소벤처기업진흥공단",    # 부산 이전 공공기관, 경영·IT 직무
    "부산항만공사",            # 본사 부산, 경영·정보통신 직무
]

# 가장 추천 기관 (사이드바 기본 선택 + 추천 배지)
RECOMMENDED_ORG = "한국자산관리공사"

ORG_SUGGESTIONS = {
    "한국자산관리공사": [
        "한국자산관리공사(캠코) 주요 채용 직무는?",
    ],
    "부산항만공사": [
        "부산항만공사 지원 자격 조건은?",
    ],
    "한국해양진흥공사": [
        "한국해양진흥공사 신입 채용도 해?",
    ],
    "주택도시보증공사": [
        "주택도시보증공사(HUG) 어떤 직무를 채용해?",
    ],
    "기술보증기금": [
        "기술보증기금 자격증 조건은?",
    ],
}

_SPA_PATH = os.environ.get("SPA_PATH", os.path.join(os.path.dirname(__file__), "templates", "index.html"))

@app.route("/")
def index():
    """모바일 SPA 서빙 — API_BASE를 환경변수 MODEL_SERVER_URL로 교체해서 반환"""
    model_server_url = os.environ.get("MODEL_SERVER_URL", "")
    try:
        with open(_SPA_PATH, encoding="utf-8") as f:
            html = f.read()
        html = html.replace(
            "const API_BASE = 'http://localhost:5000';",
            f"const API_BASE = '{model_server_url}';"
        )
        return html, 200, {
            "Content-Type": "text/html; charset=utf-8",
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
        }
    except FileNotFoundError:
        return render_template("index.html")

@app.route("/chatbot")
def chatbot():
    orgs = load_orgs()
    return render_template(
        "chatbot.html",
        featured_orgs=FEATURED_ORGS,
        recommended_org=RECOMMENDED_ORG,
        orgs=orgs,
    )

@app.route("/curriculum")
def curriculum():
    data = get_curriculum_data()
    data["job_tracks"] = get_job_tracks_data()
    return render_template("curriculum.html", **data)

@app.route("/api/curriculum")
def api_curriculum():
    data = get_curriculum_data()
    data["job_tracks"] = get_job_tracks_data()
    return jsonify(data)

@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json(force=True)
    question = data.get("question", "").strip()
    history = data.get("history", [])
    ncs_detail = data.get("ncs_detail", True)
    if not question:
        return jsonify({"error": "질문이 없습니다."}), 400
    try:
        answer, meta = process_question(question, history=history or None, ncs_detail=ncs_detail)
        return jsonify({"answer": answer, "meta": meta})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/suggestions")
def api_suggestions():
    org = request.args.get("org", "")
    suggestions = ORG_SUGGESTIONS.get(org, [f"{org} 채용 자격 조건은?", f"{org} 전형 절차는?", f"{org} 우대사항은?", f"{org} NCS 영역은?"])
    return jsonify({"suggestions": suggestions})

if __name__ == "__main__":
    # 디버그 모드에서 reloader가 두 번 실행되는 걸 방지
    if not os.environ.get("WERKZEUG_RUN_MAIN"):
        threading.Timer(1.5, lambda: webbrowser.open("http://127.0.0.1:5000")).start()
    app.run(host="0.0.0.0", debug=True, port=5000)
