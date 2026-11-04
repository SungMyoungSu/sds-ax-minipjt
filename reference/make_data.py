# make_data.py - Day4 백데이터 3종을 만드는 스크립트 (교육용 더미 데이터)
"""실행하면 같은 폴더에 employees.json, assets.json, projects.json을 만듭니다.

실행: python make_data.py

5, 6, 7일차가 이 데이터를 그대로 이어 쓰므로 값은 바꾸지 마세요.
"""
import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))

EMPLOYEES = [
    {"id": 1, "name": "김하늘", "team": "클라우드운영팀", "role": "팀장", "email": "haneul.kim@samsungsds.example.com"},
    {"id": 2, "name": "박도윤", "team": "물류플랫폼팀", "role": "백엔드", "email": "doyun.park@samsungsds.example.com"},
    {"id": 3, "name": "이서준", "team": "클라우드운영팀", "role": "인프라 엔지니어", "email": "seojun.lee@samsungsds.example.com"},
    {"id": 4, "name": "최민준", "team": "물류플랫폼팀", "role": "프론트엔드", "email": "minjun.choi@samsungsds.example.com"},
    {"id": 5, "name": "정하은", "team": "클라우드운영팀", "role": "데이터 분석가", "email": "haeun.jung@samsungsds.example.com"},
]

# 노트북 5대, 모니터 2대, GPU 서버 1대로 총 8건입니다. GPU 서버는 6일차 MCP 통합 실습이 조회하므로 빼지 마세요.
# 5-2의 총액 계산(노트북 5대 x 200만원 = 1,000만원)이 이 구성에 맞춰져 있으니 건수를 바꾸지 마세요.
ASSETS = [
    {"asset_id": "A-1001", "type": "노트북", "model": "갤럭시북4 프로", "owner": "김하늘", "status": "사용중"},
    {"asset_id": "A-1002", "type": "노트북", "model": "갤럭시북4 프로", "owner": "박도윤", "status": "사용중"},
    {"asset_id": "A-1003", "type": "노트북", "model": "갤럭시북4", "owner": "이서준", "status": "사용중"},
    {"asset_id": "A-1004", "type": "노트북", "model": "갤럭시북4", "owner": None, "status": "재고"},
    {"asset_id": "A-1005", "type": "노트북", "model": "갤럭시북4 프로", "owner": "최민준", "status": "사용중"},
    {"asset_id": "A-1006", "type": "모니터", "model": "뷰피니티 S8", "owner": "김하늘", "status": "사용중"},
    {"asset_id": "A-1007", "type": "모니터", "model": "뷰피니티 S8", "owner": "정하은", "status": "사용중"},
    {"asset_id": "A-1008", "type": "GPU 서버", "model": "NVIDIA A100 80GB", "owner": "이서준", "status": "사용중"},
]

PROJECTS = [
    {"project_id": "P-2601", "name": "차세대 물류 플랫폼 구축", "team": "물류플랫폼팀", "status": "진행중", "member_count": 12},
    {"project_id": "P-2602", "name": "클라우드 비용 최적화", "team": "클라우드운영팀", "status": "진행중", "member_count": 7},
    {"project_id": "P-2603", "name": "물류 관제 대시보드 고도화", "team": "물류플랫폼팀", "status": "진행중", "member_count": 5},
    {"project_id": "P-2604", "name": "사내 AI 어시스턴트 도입", "team": "클라우드운영팀", "status": "진행중", "member_count": 4},
]


def write_json(filename: str, rows: list) -> str:
    """한글이 깨지지 않게 UTF-8과 ensure_ascii=False로 씁니다."""
    path = os.path.join(BASE, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    return path


if __name__ == "__main__":
    for filename, rows in [("employees.json", EMPLOYEES),
                           ("assets.json", ASSETS),
                           ("projects.json", PROJECTS)]:
        write_json(filename, rows)
        print(f"생성: {filename} ({len(rows)}건)")

    laptops = [a for a in ASSETS if a["type"] == "노트북"]
    total = len(laptops) * 2000000
    print(f"확인: 자산 {len(ASSETS)}건 중 노트북 {len(laptops)}대, 한 대당 200만원이면 총액 {total:,}원")
