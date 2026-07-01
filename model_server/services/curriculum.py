from db.neo4j import get_session


def get_curriculum_data() -> dict:
    data = {
        "dept": {}, "modules": [], "courses_by_category": {},
        "qualifications": [], "skills": [], "activities": [],
        "companies_by_sector": {}, "employ_stats": {},
    }
    try:
        with get_session() as s:
            # 1) 학과 기본정보
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
                        {"year": y, "rate": dept[f"r{str(y)[2:]}"]}
                        for y in [2020, 2021, 2022, 2023, 2024]
                    ],
                    "sectors": [
                        {"name": "서비스업(은행·유통 등)", "pct": dept["s_svc"]},
                        {"name": "정보통신",              "pct": dept["s_it"]},
                        {"name": "제조업",                "pct": dept["s_mfg"]},
                        {"name": "공기업·공무원",          "pct": dept["s_pub"]},
                        {"name": "기타(진학 등)",          "pct": dept["s_etc"]},
                    ],
                }

            # 2) 진로 모듈
            for mod in s.run("""
                MATCH (d:Department {name: '경영정보학과'})-[:HAS_MODULE]->(cm:CareerModule)
                RETURN cm.name AS name, cm.description AS desc ORDER BY cm.name
            """).data():
                name = mod["name"]
                by_year = {}
                for yr in [2, 3, 4]:
                    courses = s.run("""
                        MATCH (cm:CareerModule {name: $n})-[:COVERS {year: $yr}]->(c:Course)
                        RETURN c.name AS name, c.code AS code ORDER BY c.semester, c.code
                    """, n=name, yr=yr).data()
                    if courses:
                        by_year[yr] = courses
                data["modules"].append({
                    "name":    name,
                    "desc":    mod["desc"],
                    "by_year": by_year,
                    "fields":  s.run("""
                        MATCH (cm:CareerModule {name: $n})-[r:LEADS_TO]->(cf:CareerField)
                        RETURN cf.name AS field, r.description AS desc
                    """, n=name).data(),
                    "ncs": s.run("""
                        MATCH (cm:CareerModule {name: $n})-[:HAS_NCS]->(nf:NCSField)
                        RETURN nf.name AS name, nf.ncs_path AS ncs_path, nf.details AS details
                        ORDER BY nf.name
                    """, n=name).data(),
                })

            # 3) 카테고리별 과목
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
                RETURN q.name AS name, q.category AS cat ORDER BY q.name
            """).data()

            # 5) 핵심 역량
            data["skills"] = [r["name"] for r in s.run("""
                MATCH (d:Department {name: '경영정보학과'})-[:HAS_SKILL]->(sk:CoreSkill)
                RETURN sk.name AS name
            """).data()]

            # 6) 비교과 활동
            data["activities"] = s.run("""
                MATCH (d:Department {name: '경영정보학과'})-[:HAS_ACTIVITY]->(ea:ExtraActivity)
                RETURN ea.name AS name, ea.category AS cat, ea.org AS org
                ORDER BY ea.category, ea.org
            """).data()

            # 7) 졸업생 취업처
            for row in s.run("""
                MATCH (d:Department {name: '경영정보학과'})-[:GRAD_EMPLOYED]->(gc:GradCompany)
                RETURN gc.sector AS sector, collect(gc.name) AS names ORDER BY gc.sector
            """).data():
                data["companies_by_sector"][row["sector"]] = row["names"]

    except Exception as e:
        data["error"] = str(e)
    return data
