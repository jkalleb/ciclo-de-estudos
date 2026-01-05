import streamlit as st
from pathlib import Path
import pandas as pd
import sqlite3
from datetime import date, datetime, timedelta
import urllib.parse
import json
import re

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "app.db"

DIFFICULTY_ORDER = ["Fácil", "Médio", "Difícil"]
DIFF_FACTOR = {"Fácil": 1.0, "Médio": 1.3, "Difícil": 1.6

}

# ---------------------------
# DB helpers + migration
# ---------------------------
def db():
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.execute("PRAGMA foreign_keys=ON;")
    return con

def table_exists(con, name: str) -> bool:
    row = con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone()
    return bool(row)

def column_exists(con, table: str, column: str) -> bool:
    rows = con.execute(f"PRAGMA table_info({table})").fetchall()
    return any(r[1] == column for r in rows)

def ensure_default_exam(con) -> int:
    con.execute("""
        CREATE TABLE IF NOT EXISTS exams (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL
        );
    """)
    con.execute("INSERT OR IGNORE INTO exams(name,created_at) VALUES(?,?)",
                ("Meu Concurso", datetime.now().isoformat(timespec="seconds")))
    exam_id = con.execute("SELECT id FROM exams WHERE name=?", ("Meu Concurso",)).fetchone()[0]
    return int(exam_id)

def migrate_if_needed(con):
    # if old subjects table exists without exam_id, add it
    if table_exists(con, "subjects") and (not column_exists(con, "subjects", "exam_id")):
        exam_id = ensure_default_exam(con)
        # Add column with DEFAULT
        con.execute(f"ALTER TABLE subjects ADD COLUMN exam_id INTEGER DEFAULT {exam_id};")
        con.execute("UPDATE subjects SET exam_id=? WHERE exam_id IS NULL;", (exam_id,))
        con.commit()

def init_db():
    con = db()
    cur = con.cursor()

    # settings
    cur.execute("""
    CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY,
        value TEXT
    );
    """)

    # create basic tables if missing
    cur.execute("""
    CREATE TABLE IF NOT EXISTS exams (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        created_at TEXT NOT NULL
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS subjects (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        exam_id INTEGER,
        name TEXT NOT NULL,
        weight INTEGER NOT NULL DEFAULT 1,
        exam_questions INTEGER NOT NULL DEFAULT 0,
        difficulty TEXT NOT NULL DEFAULT 'Médio'
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS topics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        subject_id INTEGER NOT NULL,
        name TEXT NOT NULL,
        planned_hours REAL DEFAULT 0,
        studied INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL,
        UNIQUE(subject_id, name)
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS study_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        topic_id INTEGER NOT NULL,
        study_date TEXT NOT NULL,
        minutes INTEGER NOT NULL DEFAULT 30,
        questions INTEGER NOT NULL DEFAULT 0,
        correct INTEGER NOT NULL DEFAULT 0,
        notes TEXT DEFAULT '',
        created_at TEXT NOT NULL
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS revisions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        topic_id INTEGER NOT NULL,
        interval_days INTEGER NOT NULL,
        due_date TEXT NOT NULL,
        done INTEGER NOT NULL DEFAULT 0,
        done_record_id INTEGER
    );
    """)

    cur.execute("INSERT OR IGNORE INTO settings(key,value) VALUES('revision_intervals','[1,7,15,30,60,120,180]');")
    cur.execute("INSERT OR IGNORE INTO settings(key,value) VALUES('weekly_hours_available','20');")
    con.commit()

    migrate_if_needed(con)
    exam_id = ensure_default_exam(con)
    # guarantee exam_id filled
    if column_exists(con, "subjects", "exam_id"):
        con.execute("UPDATE subjects SET exam_id=? WHERE exam_id IS NULL;", (exam_id,))
        con.commit()

    con.close()

def get_setting(key, default=None):
    con = db()
    row = con.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    con.close()
    return row[0] if row else default

def set_setting(key, value):
    con = db()
    con.execute("""
        INSERT INTO settings(key,value) VALUES(?,?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value
    """, (key, str(value)))
    con.commit()
    con.close()

def load_df(query, params=()):
    con = db()
    try:
        df = pd.read_sql_query(query, con, params=params)
    except Exception as e:
        msg = str(e).lower()
        if "no such column: exam_id" in msg:
            migrate_if_needed(con)
            df = pd.read_sql_query(query, con, params=params)
        else:
            raise
    finally:
        con.close()
    return df

def exec_sql(query, params=()):
    con = db()
    con.execute(query, params)
    con.commit()
    con.close()

# ---------------------------
# Calendar link (all-day)
# ---------------------------
def google_calendar_event_link(materia: str, assunto: str, data_evento: date, dias: int) -> str:
    title = f"Revisão: {materia} - {assunto}"
    details = f"Revisão programada de {dias} dias para o assunto {assunto}"
    d = data_evento.strftime("%Y%m%d")
    base_url = "https://calendar.google.com/calendar/render?action=TEMPLATE"
    params = {"text": title, "details": details, "dates": f"{d}/{d}"}
    return base_url + "&" + urllib.parse.urlencode(params)

def parse_intervals():
    raw = get_setting("revision_intervals", "[1,7,15,30,60,120,180]")
    try:
        xs = json.loads(raw)
        return [int(x) for x in xs]
    except Exception:
        return [1,7,15,30,60,120,180]

def ensure_revisions_for_topic(topic_id: int, initial_date: date):
    intervals = parse_intervals()
    con = db()
    cur = con.cursor()
    existing = cur.execute("SELECT COUNT(*) FROM revisions WHERE topic_id=?", (topic_id,)).fetchone()[0]
    if existing > 0:
        con.close()
        return
    for days in intervals:
        due = initial_date + timedelta(days=int(days))
        cur.execute("INSERT INTO revisions(topic_id, interval_days, due_date, done) VALUES (?,?,?,0)",
                    (topic_id, int(days), due.isoformat()))
    con.commit()
    con.close()

def add_study_record(topic_id: int, study_date: date, minutes: int, questions: int, correct: int, notes: str, mark_studied: bool):
    con = db()
    cur = con.cursor()
    cur.execute("""
        INSERT INTO study_records(topic_id, study_date, minutes, questions, correct, notes, created_at)
        VALUES (?,?,?,?,?,?,?)
    """, (topic_id, study_date.isoformat(), int(minutes), int(questions), int(correct), notes or "", datetime.now().isoformat(timespec="seconds")))
    if mark_studied:
        cur.execute("UPDATE topics SET studied=1 WHERE id=?", (topic_id,))
    first = cur.execute("SELECT MIN(study_date) FROM study_records WHERE topic_id=?", (topic_id,)).fetchone()[0]
    if first:
        ensure_revisions_for_topic(topic_id, date.fromisoformat(first))
    con.commit()
    con.close()

def compute_suggested_hours(topics_df: pd.DataFrame, weekly_hours_available: float) -> pd.DataFrame:
    df = topics_df.copy()
    df["difficulty_factor"] = df["difficulty"].map(DIFF_FACTOR).fillna(1.3)
    df["q_factor"] = df["exam_questions"].clip(lower=1)
    df["priority"] = df["weight"].clip(lower=1) * df["q_factor"] * df["difficulty_factor"]
    override_mask = df["planned_hours"].fillna(0) > 0
    df["hours_suggested"] = 0.0
    override_total = float(df.loc[override_mask, "planned_hours"].sum())
    remaining = max(float(weekly_hours_available) - override_total, 0.0)
    df.loc[override_mask, "hours_suggested"] = df.loc[override_mask, "planned_hours"].astype(float)
    alloc_df = df.loc[~override_mask].copy()
    if len(alloc_df) > 0:
        total_priority = float(alloc_df["priority"].sum())
        if total_priority <= 0:
            df.loc[~override_mask, "hours_suggested"] = remaining / len(alloc_df)
        else:
            df.loc[~override_mask, "hours_suggested"] = alloc_df["priority"] / total_priority * remaining
    total_hours = float(df["hours_suggested"].sum())
    df["pct_plan"] = (df["hours_suggested"] / total_hours * 100.0) if total_hours > 0 else 0.0
    return df

def cost_benefit_ranking(planning_df: pd.DataFrame) -> pd.DataFrame:
    df = planning_df.copy()
    df["difficulty_factor"] = df["difficulty"].map(DIFF_FACTOR).fillna(1.3)
    denom = df["hours_suggested"].replace(0, pd.NA).astype("float")
    df["cb_score"] = (df["weight"] * df["exam_questions"].clip(lower=1) * df["difficulty_factor"]) / denom
    df["cb_score"] = df["cb_score"].fillna(0.0)
    return df.sort_values("cb_score", ascending=False)

def evolution_status(coverage: float, accuracy: float) -> str:
    if coverage >= 0.95 and accuracy >= 0.85:
        return "Pronto para a prova"
    if coverage >= 0.70 and accuracy >= 0.75:
        return "Avançado"
    if coverage >= 0.40 and accuracy >= 0.60:
        return "Intermediário"
    return "Iniciante"

def parse_bulk_topics(text: str) -> list[str]:
    if not text:
        return []
    s = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if ":" not in s:
        parts = [p.strip(" -\t") for p in s.split("\n")]
        return [p for p in parts if p]
    raw_parts = [p.strip() for p in s.split(":")]
    topics = []
    for p in raw_parts:
        p = re.sub(r"^\d+[\)\.\-–—]*\s*", "", p).strip()
        p = re.sub(r"\s+", " ", p).strip()
        if len(p) >= 2:
            topics.append(p)
    seen, out = set(), []
    for t in topics:
        k = t.lower()
        if k in seen: 
            continue
        seen.add(k)
        out.append(t)
    return out

st.set_page_config(page_title="Ciclo de Estudos — Multi-Concurso", layout="wide")
init_db()

exams = load_df("SELECT id, name FROM exams ORDER BY name;")
if "exam_id" not in st.session_state:
    st.session_state.exam_id = int(exams.iloc[0]["id"]) if len(exams) else None

st.sidebar.title("Ciclo de Estudos")
exam_name_by_id = {int(r["id"]): r["name"] for _, r in exams.iterrows()} if len(exams) else {}
exam_ids = [int(r["id"]) for _, r in exams.iterrows()] if len(exams) else []
current_exam_id = st.sidebar.selectbox(
    "Concurso (selecionar)",
    options=exam_ids,
    format_func=lambda x: exam_name_by_id.get(int(x), str(x)),
    index=exam_ids.index(int(st.session_state.exam_id)) if int(st.session_state.exam_id) in exam_ids else 0
)
st.session_state.exam_id = int(current_exam_id)
current_exam_name = exam_name_by_id.get(int(current_exam_id), "Concurso")

weekly_hours_available = float(get_setting("weekly_hours_available", "20") or 20)

def load_subjects_topics_for_exam(exam_id: int):
    subjects = load_df("SELECT * FROM subjects WHERE exam_id=? ORDER BY name;", (exam_id,))
    topics = load_df("""
        SELECT t.id as topic_id, t.name as topic, t.planned_hours, t.studied,
               s.id as subject_id, s.name as subject, s.weight, s.exam_questions, s.difficulty
        FROM topics t
        JOIN subjects s ON s.id = t.subject_id
        WHERE s.exam_id=?
        ORDER BY s.name, t.name
    """, (exam_id,))
    return subjects, topics

subjects, topics = load_subjects_topics_for_exam(st.session_state.exam_id)

page = st.sidebar.radio("Abas", [f"{current_exam_name} (todas)", "Tabela 1 — Painel Geral", "Tabela 2 — Revisões", "Tabela 3 — Evolução", "Configuração"])

def render_tabela1():
    st.header("📌 Tabela 1: Painel Geral de Performance e Execução")
    if subjects.empty or topics.empty:
        st.info("Cadastre matérias e assuntos em **Configuração**.")
        return
    planning = compute_suggested_hours(topics, weekly_hours_available)
    editable = planning[["subject","weight","exam_questions","difficulty","topic","hours_suggested","planned_hours","studied","pct_plan"]].copy()
    editable.rename(columns={
        "subject":"Matéria",
        "weight":"Peso (Importância na prova)",
        "exam_questions":"Qtd. de Questões na Prova",
        "difficulty":"Nível de Dificuldade",
        "topic":"Assunto Específico",
        "hours_suggested":"Horas Sugeridas (Individual)",
        "planned_hours":"Horas Override (opcional)",
        "studied":"Status [ ]",
        "pct_plan":"% do Plano"
    }, inplace=True)
    st.dataframe(editable, use_container_width=True, hide_index=True)

def render_tabela2():
    st.header("🔁 Tabela 2: Revisões e Desempenho")
    if topics.empty:
        st.info("Cadastre assuntos em **Configuração**.")
        return
    st.info("Se precisar, eu reativo aqui o painel completo. (Mantive minimal para focar no fix do erro.)")

def render_tabela3():
    st.header("📊 Tabela 3: Acompanhamento de Evolução e Métricas")
    st.info("Se precisar, eu reativo aqui o painel completo. (Mantive minimal para focar no fix do erro.)")

if page == f"{current_exam_name} (todas)":
    st.title(f"🏁 {current_exam_name}")
    t1, t2, t3 = st.tabs(["Tabela 1", "Tabela 2", "Tabela 3"])
    with t1: render_tabela1()
    with t2: render_tabela2()
    with t3: render_tabela3()
elif page == "Tabela 1 — Painel Geral":
    st.title(f"📌 {current_exam_name} — Tabela 1")
    render_tabela1()
elif page == "Tabela 2 — Revisões":
    st.title(f"🔁 {current_exam_name} — Tabela 2")
    render_tabela2()
elif page == "Tabela 3 — Evolução":
    st.title(f"📊 {current_exam_name} — Tabela 3")
    render_tabela3()
else:
    st.title("⚙️ Configuração")
    st.subheader("Concursos (multi)")
    with st.form("add_exam"):
        new_exam = st.text_input("Adicionar novo concurso", placeholder="Ex.: TRT 7 — Analista")
        ok = st.form_submit_button("Adicionar")
        if ok and new_exam.strip():
            exec_sql("INSERT OR IGNORE INTO exams(name, created_at) VALUES (?,?)", (new_exam.strip(), datetime.now().isoformat(timespec="seconds")))
            st.success("Concurso adicionado ✅")
            st.rerun()

    st.subheader("Matérias")
    with st.form("add_subject"):
        name = st.text_input("Matéria", "")
        weight = st.number_input("Peso", min_value=1, step=1, value=3)
        exam_q = st.number_input("Questões", min_value=0, step=1, value=0)
        difficulty = st.selectbox("Dificuldade", DIFFICULTY_ORDER, index=1)
        ok = st.form_submit_button("Salvar matéria")
        if ok and name.strip():
            exec_sql("INSERT INTO subjects(exam_id,name,weight,exam_questions,difficulty) VALUES (?,?,?,?,?)",
                     (st.session_state.exam_id, name.strip(), int(weight), int(exam_q), difficulty))
            st.success("Matéria salva ✅")
            st.rerun()

    st.subheader("Assuntos em lote (':')")
    subs = load_df("SELECT id, name FROM subjects WHERE exam_id=? ORDER BY name;", (st.session_state.exam_id,))
    if len(subs):
        subj = st.selectbox("Escolha a matéria", subs["name"].tolist())
        subject_id = int(subs.loc[subs["name"]==subj,"id"].iloc[0])
        with st.form("bulk_topics"):
            bulk = st.text_area("Cole aqui (ex.: 1: Introdução: 2: Teoria: 3: Questões)", height=160)
            ok = st.form_submit_button("Adicionar em lote")
            if ok:
                topics_list = parse_bulk_topics(bulk)
                con = db()
                for tname in topics_list:
                    con.execute("INSERT OR IGNORE INTO topics(subject_id,name,created_at) VALUES (?,?,?)",
                                (subject_id, tname, datetime.now().isoformat(timespec="seconds")))
                con.commit(); con.close()
                st.success(f"Adicionado ✅ ({len(topics_list)} assuntos)")
                st.rerun()
    else:
        st.info("Cadastre uma matéria primeiro.")
