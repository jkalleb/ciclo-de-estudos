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
DIFF_FACTOR = {"Fácil": 1.0, "Médio": 1.3, "Difícil": 1.6}

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

    # exams (concursos)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS exams (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        created_at TEXT NOT NULL
    );
    """)

    # subjects per exam
    cur.execute("""
    CREATE TABLE IF NOT EXISTS subjects (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        exam_id INTEGER NOT NULL,
        name TEXT NOT NULL,
        weight INTEGER NOT NULL DEFAULT 1,
        exam_questions INTEGER NOT NULL DEFAULT 0,
        difficulty TEXT NOT NULL DEFAULT 'Médio',
        UNIQUE(exam_id, name),
        FOREIGN KEY(exam_id) REFERENCES exams(id) ON DELETE CASCADE
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
        UNIQUE(subject_id, name),
        FOREIGN KEY(subject_id) REFERENCES subjects(id) ON DELETE CASCADE
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
        created_at TEXT NOT NULL,
        FOREIGN KEY(topic_id) REFERENCES topics(id) ON DELETE CASCADE
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS revisions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        topic_id INTEGER NOT NULL,
        interval_days INTEGER NOT NULL,
        due_date TEXT NOT NULL,
        done INTEGER NOT NULL DEFAULT 0,
        done_record_id INTEGER,
        FOREIGN KEY(topic_id) REFERENCES topics(id) ON DELETE CASCADE,
        FOREIGN KEY(done_record_id) REFERENCES study_records(id) ON DELETE SET NULL
    );
    """)

    # defaults
    cur.execute("INSERT OR IGNORE INTO settings(key,value) VALUES('revision_intervals','[1,7,15,30,60,120,180]');")
    cur.execute("INSERT OR IGNORE INTO settings(key,value) VALUES('weekly_hours_available','20');")
    con.commit()

    # Ensure at least one exam
    cur.execute("INSERT OR IGNORE INTO exams(name,created_at) VALUES(?,?)",
                ("Meu Concurso", datetime.now().isoformat(timespec="seconds")))
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
    df = pd.read_sql_query(query, con, params=params)
    con.close()
    return df

def exec_sql(query, params=()):
    con = db()
    con.execute(query, params)
    con.commit()
    con.close()

# ---------------------------
# Calendar Links (dia inteiro)
# ---------------------------
def google_calendar_event_link(materia: str, assunto: str, data_evento: date, dias: int) -> str:
    title = f"Revisão: {materia} - {assunto}"
    details = f"Revisão programada de {dias} dias para o assunto {assunto}"
    d = data_evento.strftime("%Y%m%d")
    base_url = "https://calendar.google.com/calendar/render?action=TEMPLATE"
    params = {"text": title, "details": details, "dates": f"{d}/{d}"}
    return base_url + "&" + urllib.parse.urlencode(params)

# ---------------------------
# Revision logic
# ---------------------------
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

# ---------------------------
# Planning math
# ---------------------------
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

# ---------------------------
# Bulk topic parser (":" delimiters)
# ---------------------------
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
    seen = set()
    out = []
    for t in topics:
        k = t.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(t)
    return out

# ---------------------------
# UI
# ---------------------------
st.set_page_config(page_title="Ciclo de Estudos — Multi-Concurso", layout="wide")
init_db()

exams = load_df("SELECT id, name FROM exams ORDER BY name;")
if "exam_id" not in st.session_state:
    st.session_state.exam_id = int(exams.iloc[0]["id"]) if len(exams) else None

st.sidebar.title("Ciclo de Estudos")

exam_name_by_id = {int(r["id"]): r["name"] for _, r in exams.iterrows()} if len(exams) else {}
exam_ids = [int(r["id"]) for _, r in exams.iterrows()] if len(exams) else []
if exam_ids:
    current_exam_id = st.sidebar.selectbox(
        "Concurso (selecionar)",
        options=exam_ids,
        format_func=lambda x: exam_name_by_id.get(int(x), str(x)),
        index=exam_ids.index(int(st.session_state.exam_id)) if int(st.session_state.exam_id) in exam_ids else 0
    )
    st.session_state.exam_id = int(current_exam_id)
    current_exam_name = exam_name_by_id.get(int(current_exam_id), "Concurso")
else:
    current_exam_name = "Concurso"

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
    if subjects.empty:
        st.info("Cadastre suas matérias em **Configuração**.")
        return
    if topics.empty:
        st.info("Cadastre seus assuntos em **Configuração**.")
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

    edited = st.data_editor(
        editable,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Status [ ]": st.column_config.CheckboxColumn("Status [ ]"),
            "% do Plano": st.column_config.ProgressColumn("% do Plano", min_value=0, max_value=100, format="%.1f%%"),
            "Horas Override (opcional)": st.column_config.NumberColumn("Horas Override (opcional)", min_value=0, step=0.5),
            "Horas Sugeridas (Individual)": st.column_config.NumberColumn("Horas Sugeridas (Individual)", min_value=0, step=0.25, format="%.2f"),
        },
        disabled=[
            "Matéria","Peso (Importância na prova)","Qtd. de Questões na Prova","Nível de Dificuldade",
            "Assunto Específico","Horas Sugeridas (Individual)","% do Plano"
        ]
    )

    if st.button("💾 Salvar alterações do plano"):
        merged = planning.merge(
            edited[["Matéria","Assunto Específico","Horas Override (opcional)","Status [ ]"]],
            left_on=["subject","topic"],
            right_on=["Matéria","Assunto Específico"],
            how="left"
        )
        for _, r in merged.iterrows():
            exec_sql("UPDATE topics SET studied=?, planned_hours=? WHERE id=?",
                     (int(bool(r["Status [ ]"])), float(r["Horas Override (opcional)"] or 0), int(r["topic_id"])))
        st.success("Alterações salvas ✅")
        st.rerun()

    st.subheader("Totais")
    total_questions = int(subjects["exam_questions"].sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Total de Horas Semanais Sugeridas", f"{planning['hours_suggested'].sum():.1f} h/sem")
    c2.metric("Total de Questões da Prova", f"{total_questions}")
    c3.metric("Assuntos estudados (checkbox)", f"{int(planning['studied'].sum())}/{len(planning)}")

    st.subheader("📈 Análise de Custo-Benefício")
    cb = cost_benefit_ranking(planning)
    cb_view = cb[["subject","topic","weight","exam_questions","difficulty","hours_suggested","pct_plan","cb_score"]].copy()
    cb_view.rename(columns={
        "subject":"Matéria","topic":"Assunto","weight":"Peso","exam_questions":"Questões","difficulty":"Dificuldade",
        "hours_suggested":"Horas/sem (sug.)","pct_plan":"% do plano","cb_score":"Score CxB"
    }, inplace=True)
    st.dataframe(cb_view.head(12), use_container_width=True, hide_index=True)

def render_tabela2():
    st.header("🔁 Tabela 2: Revisões e Desempenho")
    if topics.empty:
        st.info("Cadastre matérias e assuntos em **Configuração**.")
        return

    intervals = parse_intervals()
    perf = load_df("""
        SELECT t.id as topic_id, s.name as subject, t.name as topic,
               MIN(r.study_date) as first_study_date,
               COALESCE(SUM(r.questions),0) as questions_done,
               COALESCE(SUM(r.correct),0) as correct
        FROM topics t
        JOIN subjects s ON s.id = t.subject_id
        LEFT JOIN study_records r ON r.topic_id = t.id
        WHERE s.exam_id=?
        GROUP BY t.id, s.name, t.name
        ORDER BY s.name, t.name
    """, (st.session_state.exam_id,))
    perf["pct_correct"] = perf.apply(lambda x: (x["correct"]/x["questions_done"]*100.0) if x["questions_done"] else 0.0, axis=1)

    for _, row in perf.iterrows():
        if row["first_study_date"]:
            ensure_revisions_for_topic(int(row["topic_id"]), date.fromisoformat(row["first_study_date"]))

    revs = load_df("""
        SELECT rv.topic_id, rv.interval_days, rv.due_date,
               s.name as subject, t.name as topic
        FROM revisions rv
        JOIN topics t ON t.id = rv.topic_id
        JOIN subjects s ON s.id = t.subject_id
        WHERE s.exam_id=?
        ORDER BY s.name, t.name, rv.interval_days
    """, (st.session_state.exam_id,))
    if not revs.empty:
        revs["due_date"] = pd.to_datetime(revs["due_date"]).dt.date

    if revs.empty:
        wide = perf[["subject","topic"]].copy()
        for d in intervals:
            wide[d] = pd.NaT
    else:
        wide = revs.pivot_table(index=["subject","topic"], columns="interval_days", values="due_date", aggfunc="min").reset_index()
        for d in intervals:
            if d not in wide.columns:
                wide[d] = pd.NaT
        wide = wide[["subject","topic"] + intervals]

    perf_view = perf.copy()
    perf_view["first_study_date"] = pd.to_datetime(perf_view["first_study_date"]).dt.date
    perf_view.rename(columns={
        "subject":"Matéria/Assunto",
        "topic":"Assunto",
        "first_study_date":"Data do Estudo Inicial",
        "questions_done":"Qtd. de Questões Feitas",
        "correct":"Qtd. de Acertos",
        "pct_correct":"% de Acerto"
    }, inplace=True)
    wide.rename(columns={"subject":"Matéria/Assunto","topic":"Assunto"}, inplace=True)
    df2 = perf_view.merge(wide, on=["Matéria/Assunto","Assunto"], how="left")
    st.dataframe(df2, use_container_width=True, hide_index=True)

    st.subheader("📅 Google Calendar — criar tudo de uma vez")
    if "show_all_batch" not in st.session_state:
        st.session_state.show_all_batch = False
    if st.button("⚡ Gerar painel de botões para TODOS os assuntos"):
        st.session_state.show_all_batch = True

    labels = perf.apply(lambda r: f"{r['subject']} — {r['topic']}", axis=1).tolist()
    if labels:
        chosen = st.selectbox("Escolha um assunto para gerar TODAS as revisões", labels)
        chosen_row = perf.iloc[labels.index(chosen)]
        if chosen_row["first_study_date"]:
            base_date = date.fromisoformat(chosen_row["first_study_date"])
            materia = chosen_row["subject"]
            assunto = chosen_row["topic"]
            if st.button("📌 Gerar botões deste assunto"):
                cols = st.columns(len(intervals))
                for i, d in enumerate(intervals):
                    rev_date = base_date + timedelta(days=int(d))
                    link = google_calendar_event_link(materia, assunto, rev_date, int(d))
                    cols[i].link_button(f"{d}d", link, use_container_width=True)
        else:
            st.warning("Sem data do estudo inicial. Registre um estudo em Configuração.")

    if st.session_state.show_all_batch:
        st.subheader("Painel em lote — todos os assuntos com estudo inicial")
        for _, row in perf.iterrows():
            if not row["first_study_date"]:
                continue
            materia = row["subject"]
            assunto = row["topic"]
            base_date = date.fromisoformat(row["first_study_date"])
            with st.expander(f"{materia} — {assunto} (base: {base_date.strftime('%d/%m/%Y')})"):
                cols = st.columns(len(intervals))
                for i, d in enumerate(intervals):
                    rev_date = base_date + timedelta(days=int(d))
                    link = google_calendar_event_link(materia, assunto, rev_date, int(d))
                    cols[i].link_button(f"{d}d", link, use_container_width=True)

def render_tabela3():
    st.header("📊 Tabela 3: Acompanhamento de Evolução e Métricas")
    if topics.empty:
        st.info("Cadastre matérias e assuntos em **Configuração**.")
        return

    topic_minutes = load_df("""
        SELECT t.id as topic_id,
               COALESCE(SUM(r.minutes),0) as minutes_done,
               COALESCE(SUM(r.questions),0) as q_done,
               COALESCE(SUM(r.correct),0) as correct
        FROM topics t
        LEFT JOIN study_records r ON r.topic_id = t.id
        GROUP BY t.id
    """)
    base = topics.merge(topic_minutes, on="topic_id", how="left").fillna({"minutes_done":0,"q_done":0,"correct":0})
    planning = compute_suggested_hours(base, weekly_hours_available)

    subj = planning.groupby("subject").agg(
        total_topics=("topic_id","count"),
        studied_topics=("studied","sum"),
        minutes_done=("minutes_done","sum"),
        hours_suggested=("hours_suggested","sum"),
        q_done=("q_done","sum"),
        correct=("correct","sum"),
    ).reset_index()

    subj["coverage_pct"] = subj.apply(lambda r: (r["studied_topics"]/r["total_topics"]*100.0) if r["total_topics"] else 0.0, axis=1)
    subj["accuracy_pct"] = subj.apply(lambda r: (r["correct"]/r["q_done"]*100.0) if r["q_done"] else 0.0, axis=1)
    subj["status"] = subj.apply(lambda r: evolution_status(r["coverage_pct"]/100.0, r["accuracy_pct"]/100.0), axis=1)

    subj["hours_done"] = subj["minutes_done"] / 60.0
    subj["hours_remaining"] = (subj["hours_suggested"] - subj["hours_done"]).clip(lower=0.0)
    subj["o_que_falta"] = subj.apply(lambda r: f"{int(r['total_topics']-r['studied_topics'])} assuntos + {r['hours_remaining']:.1f}h", axis=1)

    view = subj[["subject","coverage_pct","accuracy_pct","status","o_que_falta"]].copy()
    view.rename(columns={
        "subject":"Matéria",
        "coverage_pct":"% de Cobertura do Edital",
        "accuracy_pct":"Média de Acertos (%) Geral",
        "status":"Status de Evolução",
        "o_que_falta":"O que falta"
    }, inplace=True)

    st.dataframe(
        view,
        use_container_width=True,
        hide_index=True,
        column_config={
            "% de Cobertura do Edital": st.column_config.ProgressColumn("% de Cobertura do Edital", min_value=0, max_value=100, format="%.1f%%"),
            "Média de Acertos (%) Geral": st.column_config.ProgressColumn("Média de Acertos (%) Geral", min_value=0, max_value=100, format="%.1f%%"),
        }
    )

    st.subheader("Resumo de Evolução Total (Final)")
    total_topics = int(subj["total_topics"].sum())
    total_studied = int(subj["studied_topics"].sum())
    overall_coverage = (total_studied/total_topics) if total_topics else 0.0
    st.progress(overall_coverage)

    last7 = load_df("""
        SELECT study_date, SUM(minutes) as minutes
        FROM study_records
        WHERE study_date >= ?
        GROUP BY study_date
    """, ((date.today() - timedelta(days=6)).isoformat(),))
    pace = float(last7["minutes"].mean()) if len(last7) else 0.0

    remaining_hours = float(subj["hours_remaining"].sum())
    remaining_minutes = remaining_hours * 60.0

    c1, c2, c3 = st.columns(3)
    c1.metric("Barra de progresso geral do edital", f"{overall_coverage*100:.1f}%")
    c2.metric("Ritmo médio (últimos 7 dias)", f"{pace:.0f} min/dia" if pace else "—")
    if pace > 0:
        eta_days = remaining_minutes / pace
        eta_date = date.today() + timedelta(days=int(round(eta_days)))
        c3.metric("Estimativa de tempo para fechar", f"~{eta_days:.1f} dias", help=f"Data estimada: {eta_date.strftime('%d/%m/%Y')}")
    else:
        c3.metric("Estimativa de tempo para fechar", "Sem dados", help="Registre estudos para calcular o ritmo.")

# Pages
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
        if ok:
            if not new_exam.strip():
                st.error("Digite um nome de concurso.")
            else:
                exec_sql("INSERT OR IGNORE INTO exams(name, created_at) VALUES (?,?)",
                         (new_exam.strip(), datetime.now().isoformat(timespec="seconds")))
                st.success("Concurso adicionado ✅")
                st.rerun()

    st.divider()
    st.subheader("Parâmetros do planejamento")
    c1, c2 = st.columns(2)
    with c1:
        weekly = st.number_input("Horas semanais disponíveis (para distribuir no plano)", min_value=1.0, step=1.0,
                                 value=float(get_setting("weekly_hours_available","20")))
    with c2:
        intervals_txt = st.text_input("Intervalos de revisão (dias) separados por vírgula",
                                      value=",".join(map(str, parse_intervals())))
    if st.button("Salvar parâmetros"):
        set_setting("weekly_hours_available", weekly)
        xs = []
        for p in intervals_txt.split(","):
            p = p.strip()
            if not p:
                continue
            try:
                xs.append(int(p))
            except:
                pass
        if xs:
            set_setting("revision_intervals", json.dumps(xs))
        st.success("Salvo ✅")

    st.divider()
    st.subheader(f"Cadastros — {current_exam_name}")

    with st.expander("➕ Cadastrar/editar matérias"):
        with st.form("add_subject"):
            name = st.text_input("Matéria", "")
            weight = st.number_input("Peso (importância)", min_value=1, step=1, value=3)
            exam_q = st.number_input("Qtd. de questões na prova", min_value=0, step=1, value=0)
            difficulty = st.selectbox("Nível de dificuldade", DIFFICULTY_ORDER, index=1)
            ok = st.form_submit_button("Salvar matéria")
            if ok:
                if not name.strip():
                    st.error("Digite o nome da matéria.")
                else:
                    con = db()
                    con.execute("""
                        INSERT INTO subjects(exam_id,name,weight,exam_questions,difficulty)
                        VALUES (?,?,?,?,?)
                        ON CONFLICT(exam_id,name) DO UPDATE SET weight=excluded.weight, exam_questions=excluded.exam_questions, difficulty=excluded.difficulty
                    """, (st.session_state.exam_id, name.strip(), int(weight), int(exam_q), difficulty))
                    con.commit()
                    con.close()
                    st.success("Matéria salva ✅")
                    st.rerun()

        st.dataframe(load_df("SELECT name as Matéria, weight as Peso, exam_questions as Questões, difficulty as Dificuldade FROM subjects WHERE exam_id=? ORDER BY name;",
                             (st.session_state.exam_id,)),
                     use_container_width=True, hide_index=True)

    with st.expander("➕ Assuntos (com estilo de lote por ':')"):
        subjects2 = load_df("SELECT * FROM subjects WHERE exam_id=? ORDER BY name;", (st.session_state.exam_id,))
        if subjects2.empty:
            st.info("Cadastre ao menos 1 matéria primeiro.")
        else:
            subj = st.selectbox("Escolha a matéria", subjects2["name"].tolist())
            subject_id = int(subjects2.loc[subjects2["name"]==subj, "id"].iloc[0])

            st.markdown("**Adicionar vários assuntos de uma vez (cole texto separado por ':' )**")
            with st.form("add_topic_bulk"):
                bulk = st.text_area("Cole aqui (ex.: 1: Introdução: 2: Teoria: 3: Questões)", height=160)
                planned_hours_bulk = st.number_input("Horas override para todos (opcional)", min_value=0.0, step=0.5, value=0.0)
                ok2 = st.form_submit_button("Adicionar em lote")
                if ok2:
                    topics_list = parse_bulk_topics(bulk)
                    if not topics_list:
                        st.error("Não identifiquei assuntos. Use ':' (dois pontos) ou 1 por linha.")
                    else:
                        con = db()
                        for tname in topics_list:
                            con.execute("""
                                INSERT OR IGNORE INTO topics(subject_id,name,planned_hours,studied,created_at)
                                VALUES (?,?,?,?,?)
                            """, (subject_id, tname, float(planned_hours_bulk), 0, datetime.now().isoformat(timespec="seconds")))
                        con.commit()
                        con.close()
                        st.success(f"Assuntos adicionados ✅ ({len(topics_list)} detectados)")
                        st.rerun()

            st.caption("Assuntos cadastrados nessa matéria:")
            st.dataframe(load_df("""
                SELECT t.name as Assunto, t.planned_hours as HorasOverride, t.studied as Estudado
                FROM topics t JOIN subjects s ON s.id=t.subject_id
                WHERE s.exam_id=? AND s.name=?
                ORDER BY t.name
            """, (st.session_state.exam_id, subj)), use_container_width=True, hide_index=True)

    with st.expander("🕒 Registrar estudo (gera revisões automaticamente)"):
        topics3 = load_df("""
            SELECT t.id as topic_id, t.name as topic, s.name as subject
            FROM topics t JOIN subjects s ON s.id=t.subject_id
            WHERE s.exam_id=?
            ORDER BY s.name, t.name
        """, (st.session_state.exam_id,))
        if topics3.empty:
            st.info("Cadastre assuntos primeiro.")
        else:
            labels = topics3.apply(lambda r: f"{r['subject']} — {r['topic']}", axis=1).tolist()
            chosen = st.selectbox("Assunto", labels)
            chosen_row = topics3.iloc[labels.index(chosen)]
            topic_id = int(chosen_row["topic_id"])

            with st.form("study_form"):
                d = st.date_input("Data do estudo", value=date.today())
                minutes = st.number_input("Minutos", min_value=5, step=5, value=30)
                q = st.number_input("Qtd. de questões feitas", min_value=0, step=5, value=0)
                c = st.number_input("Qtd. de acertos", min_value=0, step=1, value=0)
                notes = st.text_input("Notas (opcional)", "")
                mark = st.checkbox("Marcar este assunto como estudado (checkbox do plano)", value=True)
                ok = st.form_submit_button("Registrar")
                if ok:
                    add_study_record(topic_id, d, int(minutes), int(q), int(c), notes, bool(mark))
                    st.success("Estudo registrado! Revisões geradas ✅")
                    st.rerun()
