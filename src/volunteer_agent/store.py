"""SQLite 저장소. 모든 데이터는 `DATA_DIR/volunteer.db`(사용자 PC)에만 둔다.

이 계층은 값을 저장·조회만 한다. 시간 단위·미래 날짜 같은 입력 검증과 집계는 3단계 집계 엔진에서 한다.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path
from typing import Any

from volunteer_agent.config import Settings
from volunteer_agent.models import (
    Activity,
    Deadline,
    Notification,
    Plan,
    PolicyClause,
    PolicyDocument,
    Profile,
    RuleCandidate,
    UserSetting,
    WatchlistItem,
)

DB_FILENAME = "volunteer.db"
SCHEMA_VERSION = 3  # 2: rule_candidates.status에 superseded 추가, 3: user_settings.alert_consent

SCHEMA = """
CREATE TABLE IF NOT EXISTS user_settings(
  id INTEGER PRIMARY KEY CHECK(id = 1),
  user_type TEXT NOT NULL CHECK(user_type IN ('employee','student')),
  alert_consent INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS profiles(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_type TEXT NOT NULL CHECK(user_type IN ('employee','student')),
  source TEXT NOT NULL DEFAULT 'none'
    CHECK(source IN ('company_policy','school_requirement','self_goal','none')),
  goal_hours REAL,
  period_start TEXT,
  period_end TEXT,
  certificate_required INTEGER NOT NULL DEFAULT 0,
  allowed_type_codes TEXT NOT NULL DEFAULT '[]',
  major TEXT,
  active INTEGER NOT NULL DEFAULT 1,
  created_at TEXT NOT NULL,
  ended_at TEXT
);
CREATE TABLE IF NOT EXISTS policy_documents(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  filename TEXT NOT NULL,
  format TEXT NOT NULL CHECK(format IN ('md','pdf')),
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS policy_clauses(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  document_id INTEGER NOT NULL REFERENCES policy_documents(id) ON DELETE CASCADE,
  clause_id TEXT NOT NULL,
  text TEXT NOT NULL,
  position INTEGER NOT NULL,
  page INTEGER
);
CREATE TABLE IF NOT EXISTS rule_candidates(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  profile_id INTEGER NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
  document_id INTEGER REFERENCES policy_documents(id) ON DELETE SET NULL,
  status TEXT NOT NULL DEFAULT 'candidate' CHECK(status IN ('candidate','confirmed','superseded')),
  goal_hours REAL,
  period_start TEXT,
  period_end TEXT,
  allowed_type_codes TEXT NOT NULL DEFAULT '[]',
  certificate_required INTEGER,
  source_quotes TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL,
  confirmed_at TEXT
);
CREATE TABLE IF NOT EXISTS activities(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  start_date TEXT NOT NULL,
  end_date TEXT,
  organization TEXT,
  content TEXT,
  hours REAL NOT NULL,
  type_code TEXT NOT NULL,
  certificate_path TEXT,
  major_relevance TEXT,
  lesson TEXT,
  school_planned INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS deadlines(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  kind TEXT NOT NULL CHECK(kind IN ('period_end','recruit_end','schedule')),
  due_date TEXT NOT NULL,
  title TEXT NOT NULL DEFAULT '',
  ref_id TEXT
);
CREATE TABLE IF NOT EXISTS watchlist(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  program_id TEXT NOT NULL UNIQUE,
  title TEXT NOT NULL,
  url TEXT,
  recruit_end TEXT,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS plans(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  profile_id INTEGER NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
  content TEXT NOT NULL,
  approved INTEGER NOT NULL DEFAULT 0,
  approved_at TEXT,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS notifications(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  deadline_id INTEGER NOT NULL REFERENCES deadlines(id) ON DELETE CASCADE,
  mark TEXT NOT NULL CHECK(mark IN ('D-30','D-7','D-1')),
  channel TEXT NOT NULL CHECK(channel IN ('smtp','eml')),
  path TEXT,
  sent_at TEXT NOT NULL,
  UNIQUE(deadline_id, mark)
);
"""


class StoreError(Exception):
    """저장소 규칙 위반(예: 사용자 설정 중복 저장, 없는 id)."""


# ---- 변환 도우미 ----

def _d(value: date | None) -> str | None:
    return value.isoformat() if value else None


def _pd(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


def _dt(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _pdt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _b(value: bool | None) -> int | None:
    return None if value is None else int(bool(value))


def _pb(value: int | None) -> bool | None:
    return None if value is None else bool(value)


class Store:
    """SQLite 접근 계층. `Store.open(settings)`로 연다.

    now: 현재 시각을 돌려주는 함수(테스트와 `--today` 주입용).
    """

    def __init__(self, conn: sqlite3.Connection, now: Callable[[], datetime]):
        self.conn = conn
        self.now = now

    @classmethod
    def open(cls, settings: Settings, now: Callable[[], datetime] | None = None) -> Store:
        path = settings.ensure_data_dir() / DB_FILENAME
        return cls.connect(path, now or (lambda: datetime.now(settings.tz)))

    @classmethod
    def connect(cls, path: Path | str, now: Callable[[], datetime]) -> Store:
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(SCHEMA)
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        conn.commit()
        return cls(conn, now)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _stamp(self) -> str:
        return self.now().isoformat()

    def _insert(self, sql: str, params: tuple[Any, ...]) -> int:
        with self.conn:
            cur = self.conn.execute(sql, params)
        return int(cur.lastrowid)

    def _require(self, table: str, row_id: int) -> sqlite3.Row:
        row = self.conn.execute(f"SELECT * FROM {table} WHERE id = ?", (row_id,)).fetchone()
        if row is None:
            raise StoreError(f"{table}에 id {row_id}가 없습니다.")
        return row

    # ---- 사용자 설정 (PC당 1명) ----

    def get_user_setting(self) -> UserSetting | None:
        row = self.conn.execute("SELECT * FROM user_settings WHERE id = 1").fetchone()
        if row is None:
            return None
        return UserSetting(user_type=row["user_type"], created_at=_pdt(row["created_at"]),
                           alert_consent=bool(row["alert_consent"]))

    def init_user_setting(self, user_type: str) -> UserSetting:
        """처음 실행할 때 한 번만 저장한다. 유형은 이후 바꾸지 않는다."""
        if self.get_user_setting() is not None:
            raise StoreError("사용자 설정이 이미 있습니다.")
        self._insert(
            "INSERT INTO user_settings(id, user_type, created_at) VALUES (1, ?, ?)",
            (user_type, self._stamp()),
        )
        return self.get_user_setting()

    def set_alert_consent(self, consent: bool) -> UserSetting:
        """메일 알림 수신 동의(SMTP를 처음 설정할 때 받는다, G11)."""
        if self.get_user_setting() is None:
            raise StoreError("사용자 설정이 없습니다.")
        with self.conn:
            self.conn.execute("UPDATE user_settings SET alert_consent = ? WHERE id = 1", (int(consent),))
        return self.get_user_setting()

    # ---- 기준 프로필 ----

    @staticmethod
    def _profile(row: sqlite3.Row) -> Profile:
        return Profile(
            id=row["id"],
            user_type=row["user_type"],
            source=row["source"],
            goal_hours=row["goal_hours"],
            period_start=_pd(row["period_start"]),
            period_end=_pd(row["period_end"]),
            certificate_required=bool(row["certificate_required"]),
            allowed_type_codes=json.loads(row["allowed_type_codes"]),
            major=row["major"],
            active=bool(row["active"]),
            created_at=_pdt(row["created_at"]),
            ended_at=_pdt(row["ended_at"]),
        )

    def add_profile(self, p: Profile) -> Profile:
        p.id = self._insert(
            "INSERT INTO profiles(user_type, source, goal_hours, period_start, period_end,"
            " certificate_required, allowed_type_codes, major, active, created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            (p.user_type, p.source, p.goal_hours, _d(p.period_start), _d(p.period_end),
             int(p.certificate_required), json.dumps(p.allowed_type_codes), p.major,
             int(p.active), self._stamp()),
        )
        return self.get_profile(p.id)

    def get_profile(self, profile_id: int) -> Profile:
        return self._profile(self._require("profiles", profile_id))

    def list_profiles(self, active_only: bool = False) -> list[Profile]:
        sql = "SELECT * FROM profiles" + (" WHERE active = 1" if active_only else "") + " ORDER BY id"
        return [self._profile(r) for r in self.conn.execute(sql)]

    def update_profile(self, p: Profile) -> Profile:
        self._require("profiles", p.id)
        with self.conn:
            self.conn.execute(
                "UPDATE profiles SET user_type=?, source=?, goal_hours=?, period_start=?, period_end=?,"
                " certificate_required=?, allowed_type_codes=?, major=? WHERE id=?",
                (p.user_type, p.source, p.goal_hours, _d(p.period_start), _d(p.period_end),
                 int(p.certificate_required), json.dumps(p.allowed_type_codes), p.major, p.id),
            )
        return self.get_profile(p.id)

    def end_profile(self, profile_id: int) -> Profile:
        """프로필을 지우지 않고 끝난 것으로 표시한다(기록과 이력은 남김)."""
        self._require("profiles", profile_id)
        with self.conn:
            self.conn.execute(
                "UPDATE profiles SET active = 0, ended_at = ? WHERE id = ?", (self._stamp(), profile_id)
            )
        return self.get_profile(profile_id)

    # ---- 사내규정 원문 ----

    def add_policy_document(self, doc: PolicyDocument) -> PolicyDocument:
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO policy_documents(filename, format, created_at) VALUES (?,?,?)",
                (doc.filename, doc.format, self._stamp()),
            )
            doc_id = int(cur.lastrowid)
            for i, c in enumerate(doc.clauses):
                self.conn.execute(
                    "INSERT INTO policy_clauses(document_id, clause_id, text, position, page)"
                    " VALUES (?,?,?,?,?)",
                    (doc_id, c.clause_id, c.text, i, c.page),
                )
        return self.get_policy_document(doc_id)

    def get_policy_document(self, document_id: int) -> PolicyDocument:
        row = self._require("policy_documents", document_id)
        clauses = [
            PolicyClause(id=r["id"], document_id=r["document_id"], clause_id=r["clause_id"],
                         text=r["text"], position=r["position"], page=r["page"])
            for r in self.conn.execute(
                "SELECT * FROM policy_clauses WHERE document_id = ? ORDER BY position", (document_id,)
            )
        ]
        return PolicyDocument(id=row["id"], filename=row["filename"], format=row["format"],
                              clauses=clauses, created_at=_pdt(row["created_at"]))

    def find_policy_document(self, filename: str) -> PolicyDocument | None:
        row = self.conn.execute(
            "SELECT id FROM policy_documents WHERE filename = ? ORDER BY id DESC", (filename,)
        ).fetchone()
        return None if row is None else self.get_policy_document(row["id"])

    def delete_policy_document(self, document_id: int) -> None:
        """문서와 조항을 지운다. 그 문서에서 나온 규칙 후보의 document_id는 NULL이 된다."""
        self._require("policy_documents", document_id)
        with self.conn:
            self.conn.execute("DELETE FROM policy_documents WHERE id = ?", (document_id,))

    def list_policy_documents(self) -> list[PolicyDocument]:
        ids = [r["id"] for r in self.conn.execute("SELECT id FROM policy_documents ORDER BY id")]
        return [self.get_policy_document(i) for i in ids]

    # ---- 규칙 후보와 확정 규칙 ----

    @staticmethod
    def _rule(row: sqlite3.Row) -> RuleCandidate:
        return RuleCandidate(
            id=row["id"],
            profile_id=row["profile_id"],
            document_id=row["document_id"],
            status=row["status"],
            goal_hours=row["goal_hours"],
            period_start=_pd(row["period_start"]),
            period_end=_pd(row["period_end"]),
            allowed_type_codes=json.loads(row["allowed_type_codes"]),
            certificate_required=_pb(row["certificate_required"]),
            source_quotes=json.loads(row["source_quotes"]),
            created_at=_pdt(row["created_at"]),
            confirmed_at=_pdt(row["confirmed_at"]),
        )

    def add_rule_candidate(self, r: RuleCandidate) -> RuleCandidate:
        r.id = self._insert(
            "INSERT INTO rule_candidates(profile_id, document_id, status, goal_hours, period_start,"
            " period_end, allowed_type_codes, certificate_required, source_quotes, created_at)"
            " VALUES (?,?,'candidate',?,?,?,?,?,?,?)",
            (r.profile_id, r.document_id, r.goal_hours, _d(r.period_start), _d(r.period_end),
             json.dumps(r.allowed_type_codes), _b(r.certificate_required),
             json.dumps(r.source_quotes, ensure_ascii=False), self._stamp()),
        )
        return self.get_rule(r.id)

    def get_rule(self, rule_id: int) -> RuleCandidate:
        return self._rule(self._require("rule_candidates", rule_id))

    def list_rules(self, profile_id: int | None = None, status: str | None = None) -> list[RuleCandidate]:
        sql, params = "SELECT * FROM rule_candidates WHERE 1=1", []
        if profile_id is not None:
            sql += " AND profile_id = ?"
            params.append(profile_id)
        if status is not None:
            sql += " AND status = ?"
            params.append(status)
        return [self._rule(r) for r in self.conn.execute(sql + " ORDER BY id", params)]

    def mark_rule_superseded(self, rule_id: int) -> RuleCandidate:
        """새 규칙이 확정되면 이전 확정 규칙은 지우지 않고 superseded로 남긴다(이력)."""
        self._require("rule_candidates", rule_id)
        with self.conn:
            self.conn.execute("UPDATE rule_candidates SET status = 'superseded' WHERE id = ?", (rule_id,))
        return self.get_rule(rule_id)

    def mark_rule_confirmed(self, rule_id: int) -> RuleCandidate:
        """상태만 바꾼다. 사용자 명시 승인 확인과 프로필 반영은 6단계 confirm_rules에서 한다."""
        self._require("rule_candidates", rule_id)
        with self.conn:
            self.conn.execute(
                "UPDATE rule_candidates SET status = 'confirmed', confirmed_at = ? WHERE id = ?",
                (self._stamp(), rule_id),
            )
        return self.get_rule(rule_id)

    # ---- 활동 ----

    @staticmethod
    def _activity(row: sqlite3.Row) -> Activity:
        return Activity(
            id=row["id"],
            start_date=_pd(row["start_date"]),
            end_date=_pd(row["end_date"]),
            organization=row["organization"],
            content=row["content"],
            hours=row["hours"],
            type_code=row["type_code"],
            certificate_path=row["certificate_path"],
            major_relevance=row["major_relevance"],
            lesson=row["lesson"],
            school_planned=bool(row["school_planned"]),
            created_at=_pdt(row["created_at"]),
            updated_at=_pdt(row["updated_at"]),
        )

    def add_activity(self, a: Activity) -> Activity:
        stamp = self._stamp()
        a.id = self._insert(
            "INSERT INTO activities(start_date, end_date, organization, content, hours, type_code,"
            " certificate_path, major_relevance, lesson, school_planned, created_at, updated_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (_d(a.start_date), _d(a.end_date), a.organization, a.content, a.hours, a.type_code,
             a.certificate_path, a.major_relevance, a.lesson, int(a.school_planned), stamp, stamp),
        )
        return self.get_activity(a.id)

    def get_activity(self, activity_id: int) -> Activity:
        return self._activity(self._require("activities", activity_id))

    def list_activities(self) -> list[Activity]:
        return [self._activity(r) for r in
                self.conn.execute("SELECT * FROM activities ORDER BY start_date, id")]

    def update_activity(self, a: Activity) -> Activity:
        self._require("activities", a.id)
        with self.conn:
            self.conn.execute(
                "UPDATE activities SET start_date=?, end_date=?, organization=?, content=?, hours=?,"
                " type_code=?, certificate_path=?, major_relevance=?, lesson=?, school_planned=?,"
                " updated_at=? WHERE id=?",
                (_d(a.start_date), _d(a.end_date), a.organization, a.content, a.hours, a.type_code,
                 a.certificate_path, a.major_relevance, a.lesson, int(a.school_planned),
                 self._stamp(), a.id),
            )
        return self.get_activity(a.id)

    def delete_activity(self, activity_id: int) -> None:
        self._require("activities", activity_id)
        with self.conn:
            self.conn.execute("DELETE FROM activities WHERE id = ?", (activity_id,))

    # ---- 마감 ----

    @staticmethod
    def _deadline(row: sqlite3.Row) -> Deadline:
        return Deadline(id=row["id"], kind=row["kind"], due_date=_pd(row["due_date"]),
                        title=row["title"], ref_id=row["ref_id"])

    def add_deadline(self, d: Deadline) -> Deadline:
        d.id = self._insert(
            "INSERT INTO deadlines(kind, due_date, title, ref_id) VALUES (?,?,?,?)",
            (d.kind, _d(d.due_date), d.title, d.ref_id),
        )
        return self.get_deadline(d.id)

    def get_deadline(self, deadline_id: int) -> Deadline:
        return self._deadline(self._require("deadlines", deadline_id))

    def list_deadlines(self) -> list[Deadline]:
        return [self._deadline(r) for r in
                self.conn.execute("SELECT * FROM deadlines ORDER BY due_date, id")]

    def delete_deadline(self, deadline_id: int) -> None:
        self._require("deadlines", deadline_id)
        with self.conn:
            self.conn.execute("DELETE FROM deadlines WHERE id = ?", (deadline_id,))

    # ---- 관심 목록 ----

    @staticmethod
    def _watch(row: sqlite3.Row) -> WatchlistItem:
        return WatchlistItem(id=row["id"], program_id=row["program_id"], title=row["title"],
                             url=row["url"], recruit_end=_pd(row["recruit_end"]),
                             created_at=_pdt(row["created_at"]))

    def add_watchlist_item(self, w: WatchlistItem) -> WatchlistItem:
        """같은 program_id가 이미 있으면 새로 넣지 않고 기존 항목을 돌려준다."""
        row = self.conn.execute("SELECT * FROM watchlist WHERE program_id = ?", (w.program_id,)).fetchone()
        if row is not None:
            return self._watch(row)
        w.id = self._insert(
            "INSERT INTO watchlist(program_id, title, url, recruit_end, created_at) VALUES (?,?,?,?,?)",
            (w.program_id, w.title, w.url, _d(w.recruit_end), self._stamp()),
        )
        return self._watch(self._require("watchlist", w.id))

    def list_watchlist(self) -> list[WatchlistItem]:
        return [self._watch(r) for r in self.conn.execute("SELECT * FROM watchlist ORDER BY id")]

    def remove_watchlist_item(self, program_id: str) -> bool:
        with self.conn:
            cur = self.conn.execute("DELETE FROM watchlist WHERE program_id = ?", (program_id,))
        return cur.rowcount > 0

    # ---- 계획 ----

    @staticmethod
    def _plan(row: sqlite3.Row) -> Plan:
        return Plan(id=row["id"], profile_id=row["profile_id"], content=json.loads(row["content"]),
                    approved=bool(row["approved"]), approved_at=_pdt(row["approved_at"]),
                    created_at=_pdt(row["created_at"]))

    def add_plan(self, p: Plan) -> Plan:
        p.id = self._insert(
            "INSERT INTO plans(profile_id, content, approved, created_at) VALUES (?,?,0,?)",
            (p.profile_id, json.dumps(p.content, ensure_ascii=False), self._stamp()),
        )
        return self.get_plan(p.id)

    def get_plan(self, plan_id: int) -> Plan:
        return self._plan(self._require("plans", plan_id))

    def list_plans(self, profile_id: int | None = None) -> list[Plan]:
        if profile_id is None:
            rows = self.conn.execute("SELECT * FROM plans ORDER BY id")
        else:
            rows = self.conn.execute("SELECT * FROM plans WHERE profile_id = ? ORDER BY id", (profile_id,))
        return [self._plan(r) for r in rows]

    def mark_plan_approved(self, plan_id: int) -> Plan:
        """상태만 바꾼다. 사용자 명시 승인 확인은 13단계 approve_plan에서 한다."""
        self._require("plans", plan_id)
        with self.conn:
            self.conn.execute("UPDATE plans SET approved = 1, approved_at = ? WHERE id = ?",
                              (self._stamp(), plan_id))
        return self.get_plan(plan_id)

    # ---- 알림 발송 기록 ----

    def was_notified(self, deadline_id: int, mark: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM notifications WHERE deadline_id = ? AND mark = ?", (deadline_id, mark)
        ).fetchone()
        return row is not None

    def record_notification(self, n: Notification) -> Notification:
        """발송(또는 .eml 저장)에 성공한 뒤에만 부른다. 같은 마감·같은 D-day는 한 번만 기록된다."""
        if self.was_notified(n.deadline_id, n.mark):
            raise StoreError(f"마감 {n.deadline_id}의 {n.mark} 알림은 이미 기록되어 있습니다.")
        n.id = self._insert(
            "INSERT INTO notifications(deadline_id, mark, channel, path, sent_at) VALUES (?,?,?,?,?)",
            (n.deadline_id, n.mark, n.channel, n.path, self._stamp()),
        )
        row = self._require("notifications", n.id)
        return Notification(id=row["id"], deadline_id=row["deadline_id"], mark=row["mark"],
                            channel=row["channel"], path=row["path"], sent_at=_pdt(row["sent_at"]))

    def list_notifications(self, deadline_id: int | None = None) -> list[Notification]:
        if deadline_id is None:
            rows = self.conn.execute("SELECT * FROM notifications ORDER BY id")
        else:
            rows = self.conn.execute(
                "SELECT * FROM notifications WHERE deadline_id = ? ORDER BY id", (deadline_id,))
        return [Notification(id=r["id"], deadline_id=r["deadline_id"], mark=r["mark"],
                             channel=r["channel"], path=r["path"], sent_at=_pdt(r["sent_at"]))
                for r in rows]
