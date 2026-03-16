"""
🎓 Dashboard — Previsão de Evasão Escolar no Ensino Médio
=========================================================
Execute com:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import joblib

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÃO DA PÁGINA
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Evasão Escolar · Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

COR_ALTO     = "#E74C3C"
COR_MEDIO    = "#F39C12"
COR_BAIXO    = "#27AE60"
COR_PRIMARIA = "#1A2E4A"
COR_ACCENT   = "#2980B9"

TEMPLATE = dict(
    plot_bgcolor="white", paper_bgcolor="white",
    font=dict(family="DM Sans, sans-serif", color="#2C3E50"),
    margin=dict(t=40, b=30, l=20, r=20),
    title_font=dict(color="#1A2E4A", size=14),
    legend=dict(font=dict(color="#2C3E50")),
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background-color: #F0F3F8; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1A2E4A 0%, #16253D 100%);
}
[data-testid="stSidebar"] > div * { color: #D0D9E8 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #FFFFFF !important; }

[data-testid="stTabs"] button p { color: #1A2E4A !important; font-weight: 600; }
[data-testid="stTabs"] button[aria-selected="true"] p { color: #2980B9 !important; }
.main h1, .main h2, .main h3 { color: #1A2E4A !important; }

.kpi-card {
    background: white; border-radius: 12px; padding: 20px 24px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08); border-left: 4px solid #2980B9;
    margin-bottom: 8px;
}
.kpi-card.danger  { border-left-color: #E74C3C; }
.kpi-card.warning { border-left-color: #F39C12; }
.kpi-card.success { border-left-color: #27AE60; }
.kpi-label { font-size:0.78rem; font-weight:600; color:#7F8C9A;
             text-transform:uppercase; letter-spacing:0.06em; }
.kpi-value { font-size:2.0rem; font-weight:700; color:#1A2E4A;
             line-height:1.1; margin:4px 0; }
.kpi-delta { font-size:0.82rem; color:#7F8C9A; }
.section-title {
    font-size:1.05rem; font-weight:700; color:#1A2E4A;
    padding:0 0 6px 0; margin-top:6px; border-bottom:2px solid #E8ECF0;

    /* Títulos das abas */
[data-testid="stTabs"] button p {
    color: #1A2E4A !important;
    font-weight: 600;
}
[data-testid="stTabs"] button[aria-selected="true"] p {
    color: #2980B9 !important;
}

/* Títulos h1, h2, h3 fora da sidebar */
.main h1, .main h2, .main h3 {
    color: #1A2E4A !important;
}

/* Texto geral do corpo */
.main p, .main span, .main div {
    color: #2C3E50;
}
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DADOS E MODELO
# ─────────────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent.parent

@st.cache_data
def carregar_dados():
    df = pd.read_csv(BASE / "data" / "evasao_escolar.csv")
    df["media_notas"]      = (df["nota_portugues"] + df["nota_matematica"]) / 2
    df["baixo_desempenho"] = ((df["nota_portugues"] < 5) | (df["nota_matematica"] < 5)).astype(int)
    df["risco_faltas"]     = (df["faltas_anuais"] > 15).astype(int)
    df["risco_distorcao"]  = (df["distorcao_idade_serie"] >= 2).astype(int)
    df["engajamento"]      = df["media_notas"] / (df["faltas_anuais"] + 1)
    df["evasao_label"]     = df["evasao"].map({0: "Não Evadiu", 1: "Evadiu"})
    df["serie_label"]      = df["serie"].map({"1ano": "1º Ano", "2ano": "2º Ano", "3ano": "3º Ano"})
    df["turno_label"]      = df["turno"].str.capitalize()
    return df

@st.cache_resource
def carregar_modelo():
    try:
        pipe = joblib.load(BASE / "models" / "pipeline_rf.pkl")
        thr  = joblib.load(BASE / "models" / "threshold_rf.pkl")
        return pipe, thr
    except Exception:
        return None, 0.304

df = carregar_dados()
modelo, threshold = carregar_modelo()

NUM_FEATURES = [
    "idade","nota_portugues","nota_matematica","media_notas",
    "distorcao_idade_serie","faltas_anuais","repeticoes_anteriores",
    "renda_familiar","trabalha","escolaridade_pai","escolaridade_mae",
    "baixo_desempenho","risco_faltas","risco_distorcao","engajamento",
]
CAT_FEATURES = ["serie","turno","sexo","raca_cor"]

if modelo:
    df["score_risco"] = modelo.predict_proba(df[NUM_FEATURES + CAT_FEATURES])[:, 1]
else:
    df["score_risco"] = np.random.uniform(0, 1, len(df))

def nivel_risco(score, thr):
    if score >= thr:          return "Alto"
    elif score >= thr * 0.65: return "Médio"
    else:                     return "Baixo"

df["nivel_risco"] = df["score_risco"].apply(lambda s: nivel_risco(s, threshold))

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 Evasão Escolar")
    st.markdown("**Dashboard de Risco Preditivo**")
    st.markdown("---")
    st.markdown("### 🔍 Filtros")

    serie_sel = st.selectbox("Série",  ["Todas"] + sorted(df["serie_label"].unique()))
    turno_sel = st.selectbox("Turno",  ["Todos"] + sorted(df["turno_label"].unique()))
    risco_sel = st.selectbox("Nível de Risco", ["Todos", "Alto", "Médio", "Baixo"])
    sexo_sel  = st.selectbox("Sexo",   ["Todos", "M", "F"])

    st.markdown("---")
    st.markdown("### 📊 Sobre o Modelo")
    st.markdown(
        "**Random Forest** treinado com dados sintéticos "
        "baseados no Censo Escolar (INEP) e SAEB.\n\n"
        "🎯 **AUC-ROC:** 0.970  \n"
        "📢 **Recall:** 94.7%  \n"
        f"⚖️ **Threshold:** {threshold:.3f}"
    )
    st.markdown("---")
    st.caption("Portfólio · Ciência de Dados")

# Filtros
dff = df.copy()
if serie_sel != "Todas": dff = dff[dff["serie_label"] == serie_sel]
if turno_sel != "Todos": dff = dff[dff["turno_label"] == turno_sel]
if risco_sel != "Todos": dff = dff[dff["nivel_risco"] == risco_sel]
if sexo_sel  != "Todos": dff = dff[dff["sexo"]        == sexo_sel]

# ─────────────────────────────────────────────────────────────────────────────
# ABAS
# ─────────────────────────────────────────────────────────────────────────────
aba1, aba2, aba3, aba4 = st.tabs([
    "📊  Visão Geral",
    "👤  Perfil de Risco",
    "🤖  Desempenho do Modelo",
    "🔎  Prever Aluno",
])

# ══════════════════════════════════════════════════════════
# ABA 1 — VISÃO GERAL
# ══════════════════════════════════════════════════════════
with aba1:
    st.markdown('<h2 style="color:#1A2E4A;">Visão Geral da Evasão Escolar</h2>', unsafe_allow_html=True)
    st.caption(f"{len(dff):,} alunos com os filtros selecionados")

    c1, c2, c3, c4, c5 = st.columns(5)
    taxa_ev  = dff["evasao"].mean() * 100
    n_alto   = (dff["nivel_risco"] == "Alto").sum()
    pct_alto = n_alto / len(dff) * 100 if len(dff) else 0
    m_faltas = dff["faltas_anuais"].mean()
    m_notas  = dff["media_notas"].mean()
    pct_trab = dff["trabalha"].mean() * 100

    for col, cls, label, val, delta in [
        (c1, "danger",  "Taxa de Evasão",      f"{taxa_ev:.1f}%",  f"{dff['evasao'].sum():,} alunos"),
        (c2, "danger",  "Alto Risco",           f"{n_alto:,}",      f"{pct_alto:.1f}% do total"),
        (c3, "warning", "Média de Faltas",      f"{m_faltas:.1f}",  "dias por ano"),
        (c4, "success", "Média de Notas",       f"{m_notas:.1f}",   "de 0 a 10"),
        (c5, "",        "Alunos que Trabalham", f"{pct_trab:.1f}%", f"{dff['trabalha'].sum():,} alunos"),
    ]:
        with col:
            st.markdown(f"""
            <div class="kpi-card {cls}">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{val}</div>
                <div class="kpi-delta">{delta}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-title">Taxa de Evasão por Série</div>', unsafe_allow_html=True)
        t = (dff.groupby("serie_label")["evasao"].mean() * 100).reset_index()
        t.columns = ["Série", "Taxa (%)"]
        fig = px.bar(t.sort_values("Série"), x="Série", y="Taxa (%)",
                     color="Taxa (%)",
                     color_continuous_scale=["#27AE60", "#F39C12", "#E74C3C"],
                     text=t.sort_values("Série")["Taxa (%)"].round(1).astype(str) + "%")
        fig.update_traces(textposition="outside", marker_line_width=0)
        fig.update_layout(**TEMPLATE,
            coloraxis_showscale=False,
            yaxis=dict(title="Taxa (%)", title_font=dict(color="#2C3E50"), tickfont=dict(color="#2C3E50")),
            xaxis=dict(title="", tickfont=dict(color="#2C3E50")),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-title">Taxa de Evasão por Turno</div>', unsafe_allow_html=True)
        t = (dff.groupby("turno_label")["evasao"].mean() * 100).reset_index()
        t.columns = ["Turno", "Taxa (%)"]
        fig = px.bar(t.sort_values("Taxa (%)"), x="Taxa (%)", y="Turno", orientation="h",
                     color="Turno",
                     color_discrete_map={"Matutino": COR_BAIXO, "Vespertino": COR_MEDIO, "Noturno": COR_ALTO},
                     text=t.sort_values("Taxa (%)")["Taxa (%)"].round(1).astype(str) + "%")
        fig.update_traces(textposition="outside", marker_line_width=0)
        fig.update_layout(**TEMPLATE,
            showlegend=False,
            xaxis=dict(title="Taxa (%)", title_font=dict(color="#2C3E50"), tickfont=dict(color="#2C3E50")),
            yaxis=dict(title="", tickfont=dict(color="#2C3E50")),
        )
        st.plotly_chart(fig, use_container_width=True)

    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown('<div class="section-title">Evasão por Distorção Idade-Série</div>', unsafe_allow_html=True)
        dff2 = dff.copy()
        dff2["cat_dist"] = pd.cut(dff2["distorcao_idade_serie"],
                                   bins=[-0.1, 0, 1, 2, 5],
                                   labels=["Adequado", "Pequeno (1a)", "Médio (2a)", "Grande (3a+)"])
        t = (dff2.groupby("cat_dist", observed=True)["evasao"].mean() * 100).reset_index()
        t.columns = ["Distorção", "Taxa (%)"]
        fig = px.bar(t, x="Distorção", y="Taxa (%)",
                     color="Taxa (%)",
                     color_continuous_scale=["#27AE60", "#F39C12", "#E74C3C", "#922B21"],
                     text=t["Taxa (%)"].round(1).astype(str) + "%")
        fig.update_traces(textposition="outside", marker_line_width=0)
        fig.update_layout(**TEMPLATE,
            coloraxis_showscale=False,
            yaxis=dict(title="Taxa (%)", title_font=dict(color="#2C3E50"), tickfont=dict(color="#2C3E50")),
            xaxis=dict(title="", tickfont=dict(color="#2C3E50")),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_d:
        st.markdown('<div class="section-title">Faltas Anuais × Média de Notas</div>', unsafe_allow_html=True)
        sample = dff.sample(min(1500, len(dff)), random_state=42)
        fig = px.scatter(sample, x="faltas_anuais", y="media_notas",
                         color="evasao_label",
                         color_discrete_map={"Não Evadiu": COR_BAIXO, "Evadiu": COR_ALTO},
                         opacity=0.55,
                         labels={"faltas_anuais": "Faltas Anuais",
                                 "media_notas": "Média de Notas",
                                 "evasao_label": "Status"})
        fig.update_traces(marker=dict(size=5))
        fig.update_layout(**TEMPLATE,
            xaxis=dict(title="Faltas Anuais", title_font=dict(color="#2C3E50"), tickfont=dict(color="#2C3E50")),
            yaxis=dict(title="Média de Notas", title_font=dict(color="#2C3E50"), tickfont=dict(color="#2C3E50")),
        )
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════
# ABA 2 — PERFIL DE RISCO
# ══════════════════════════════════════════════════════════
with aba2:
    st.markdown("## Perfil dos Alunos por Nível de Risco")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown('<div class="section-title">Distribuição de Risco</div>', unsafe_allow_html=True)
        dist = dff["nivel_risco"].value_counts().reindex(["Alto", "Médio", "Baixo"]).reset_index()
        dist.columns = ["Nível", "Alunos"]
        fig = px.pie(dist, values="Alunos", names="Nível",
                     color="Nível",
                     color_discrete_map={"Alto": COR_ALTO, "Médio": COR_MEDIO, "Baixo": COR_BAIXO},
                     hole=0.48)
        fig.update_traces(textposition="outside", textinfo="percent+label",
                          marker=dict(line=dict(color="white", width=2)))
        fig.update_layout(**TEMPLATE, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">Distribuição dos Scores de Risco</div>', unsafe_allow_html=True)
        fig = go.Figure()
        for nivel, cor in [("Baixo", COR_BAIXO), ("Médio", COR_MEDIO), ("Alto", COR_ALTO)]:
            sub = dff[dff["nivel_risco"] == nivel]["score_risco"]
            fig.add_trace(go.Histogram(x=sub, name=nivel, marker_color=cor,
                                       opacity=0.75, nbinsx=40, marker_line_width=0))
        fig.add_vline(x=threshold, line_dash="dash", line_color="#1A2E4A", line_width=2,
                      annotation_text=f"Threshold ({threshold:.2f})",
                      annotation_position="top right")
        fig.update_layout(**TEMPLATE,
            barmode="overlay",
            xaxis=dict(title="Score de Risco", title_font=dict(color="#2C3E50"), tickfont=dict(color="#2C3E50")),
            yaxis=dict(title="Frequência", title_font=dict(color="#2C3E50"), tickfont=dict(color="#2C3E50")),
            legend_title="Nível",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Perfil Médio: Alto Risco vs Baixo Risco</div>', unsafe_allow_html=True)
    alto  = dff[dff["nivel_risco"] == "Alto"]
    baixo = dff[dff["nivel_risco"] == "Baixo"]

    metricas = [
        ("Nota Port.",  "nota_portugues",       "0.2f", True),
        ("Nota Mat.",   "nota_matematica",       "0.2f", True),
        ("Faltas",      "faltas_anuais",         "0.1f", False),
        ("Distorção",   "distorcao_idade_serie", "0.2f", False),
        ("Reprovações", "repeticoes_anteriores", "0.2f", False),
        ("Renda (R$)",  "renda_familiar",        ",.0f", True),
    ]
    cols = st.columns(6)
    for col, (label, campo, fmt, inv) in zip(cols, metricas):
        va = alto[campo].mean()  if len(alto)  else 0
        vb = baixo[campo].mean() if len(baixo) else 0
        with col:
            st.metric(label, f"{va:{fmt}}", f"{va-vb:+{fmt}} vs baixo",
                      delta_color="inverse" if inv else "normal")

    st.markdown("<br>", unsafe_allow_html=True)
    col_e, col_f = st.columns(2)

    with col_e:
        st.markdown('<div class="section-title">Boxplot de Notas por Nível de Risco</div>', unsafe_allow_html=True)
        fig = go.Figure()
        for nivel, cor in [("Baixo", COR_BAIXO), ("Médio", COR_MEDIO), ("Alto", COR_ALTO)]:
            sub = dff[dff["nivel_risco"] == nivel]
            for disc, campo in [("Port.", "nota_portugues"), ("Mat.", "nota_matematica")]:
                fig.add_trace(go.Box(y=sub[campo], name=f"{nivel}·{disc}",
                                     marker_color=cor, opacity=0.8, boxmean=True,
                                     showlegend=(disc == "Port.")))
        fig.update_layout(**TEMPLATE,
            yaxis=dict(title="Nota (0-10)", title_font=dict(color="#2C3E50"), tickfont=dict(color="#2C3E50")),
            xaxis=dict(title="", tickfont=dict(color="#2C3E50")),
            legend_title="Nível",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_f:
        st.markdown('<div class="section-title">Média de Faltas: Turno × Nível de Risco</div>', unsafe_allow_html=True)
        t_heat = dff.groupby(["turno_label", "nivel_risco"])["faltas_anuais"].mean().unstack()
        t_heat = t_heat.reindex(columns=["Baixo", "Médio", "Alto"])
        fig = px.imshow(t_heat, color_continuous_scale=["#EBF5FB", "#F39C12", "#E74C3C"],
                        text_auto=".1f",
                        labels=dict(x="Nível de Risco", y="Turno", color="Faltas"),
                        aspect="auto")
        fig.update_layout(**TEMPLATE,
            xaxis=dict(title="Nível de Risco", title_font=dict(color="#2C3E50"), tickfont=dict(color="#2C3E50")),
            yaxis=dict(title="Turno", title_font=dict(color="#2C3E50"), tickfont=dict(color="#2C3E50")),
        )
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════
# ABA 3 — DESEMPENHO DO MODELO
# ══════════════════════════════════════════════════════════
with aba3:
    st.markdown("## Desempenho do Modelo Preditivo")

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, cls, label, val, desc in [
        (c1, "success", "AUC-ROC",   "0.9700", "Capacidade de separação"),
        (c2, "success", "Recall",    "94.7%",  "Alunos em risco identificados"),
        (c3, "warning", "Precisão",  "67.6%",  "Acertos entre os sinalizados"),
        (c4, "success", "F2-Score",  "0.8769", "Recall com peso 2×"),
        (c5, "",        "Threshold", f"{threshold:.3f}", "Ponto de corte ótimo"),
    ]:
        with col:
            st.markdown(f"""
            <div class="kpi-card {cls}">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value" style="font-size:1.7rem">{val}</div>
                <div class="kpi-delta">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Importância das Features (Random Forest)</div>', unsafe_allow_html=True)
    fi = pd.DataFrame({
        "Feature": ["distorcao_idade_serie","faltas_anuais","engajamento",
                    "nota_matematica","media_notas","nota_portugues",
                    "risco_distorcao","risco_faltas","renda_familiar",
                    "repeticoes_anteriores","baixo_desempenho","idade",
                    "trabalha","escolaridade_mae","escolaridade_pai"],
        "Importância": [0.182,0.154,0.119,0.108,0.097,0.089,
                        0.071,0.063,0.041,0.033,0.018,0.012,
                        0.006,0.004,0.003],
        "Categoria": ["Escolar","Escolar","Engenharia",
                      "Acadêmico","Engenharia","Acadêmico",
                      "Engenharia","Engenharia","Socioeconômico",
                      "Escolar","Engenharia","Demográfico",
                      "Socioeconômico","Socioeconômico","Socioeconômico"],
    })
    fig = px.bar(fi, x="Importância", y="Feature", orientation="h",
                 color="Categoria",
                 color_discrete_map={"Escolar": COR_ALTO, "Acadêmico": COR_MEDIO,
                                     "Engenharia": COR_ACCENT, "Socioeconômico": "#8E44AD",
                                     "Demográfico": "#7F8C8D"},
                 text=fi["Importância"].apply(lambda v: f"{v:.3f}"))
    fig.update_traces(textposition="outside", marker_line_width=0)
    fig.update_layout(**TEMPLATE,
        yaxis=dict(categoryorder="total ascending",
                   title="", title_font=dict(color="#2C3E50"), tickfont=dict(color="#2C3E50")),
        xaxis=dict(title="Importância Relativa", title_font=dict(color="#2C3E50"), tickfont=dict(color="#2C3E50")),
        legend_title="Categoria", height=480,
    )
    st.plotly_chart(fig, use_container_width=True)

    col_g, col_h = st.columns(2)

    with col_g:
        st.markdown('<div class="section-title">Taxa Real de Evasão por Nível de Risco</div>', unsafe_allow_html=True)
        taxa = dff.groupby("nivel_risco")["evasao"].mean().reindex(["Alto","Médio","Baixo"]) * 100
        fig = px.bar(taxa.reset_index(), x="nivel_risco", y="evasao",
                     color="nivel_risco",
                     color_discrete_map={"Alto": COR_ALTO, "Médio": COR_MEDIO, "Baixo": COR_BAIXO},
                     text=taxa.values.round(1),
                     labels={"nivel_risco": "Nível", "evasao": "Taxa Real (%)"})
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", marker_line_width=0)
        fig.update_layout(**TEMPLATE,
            showlegend=False,
            yaxis=dict(title="Taxa Real (%)", title_font=dict(color="#2C3E50"), tickfont=dict(color="#2C3E50")),
            xaxis=dict(title="", tickfont=dict(color="#2C3E50")),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_h:
        st.markdown('<div class="section-title">Por que Recall é a métrica principal?</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="background:white;border-radius:12px;padding:20px;box-shadow:0 1px 4px rgba(0,0,0,0.08)">
        <p style="color:#1A2E4A;font-size:0.95rem;line-height:1.8">
        Neste problema, <b>falsos negativos têm custo muito maior</b> do que falsos positivos:
        </p>
        <table style="width:100%;font-size:0.88rem;border-collapse:collapse">
          <tr style="background:#F4F6F9">
            <th style="padding:8px;text-align:left">Tipo de Erro</th>
            <th style="padding:8px;text-align:left">Significado</th>
            <th style="padding:8px;text-align:left">Custo</th>
          </tr>
          <tr>
            <td style="padding:8px;color:#E74C3C;font-weight:600">Falso Negativo</td>
            <td style="padding:8px">Aluno em risco <b>não identificado</b></td>
            <td style="padding:8px;color:#E74C3C">🔴 Alto — evasão ocorre</td>
          </tr>
          <tr style="background:#F4F6F9">
            <td style="padding:8px;color:#F39C12;font-weight:600">Falso Positivo</td>
            <td style="padding:8px">Aluno saudável <b>sinalizado como risco</b></td>
            <td style="padding:8px;color:#F39C12">🟡 Baixo — atenção extra</td>
          </tr>
        </table>
        <p style="color:#1A2E4A;font-size:0.9rem;margin-top:12px">
        Por isso usamos o <b>F2-Score</b> — que pondera o Recall com peso 2× —
        para selecionar o threshold ótimo em vez do F1 padrão.
        </p>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# ABA 4 — PREVER ALUNO
# ══════════════════════════════════════════════════════════
with aba4:
    st.markdown("## 🔎 Calcular Score de Risco para um Aluno")
    st.markdown("Preencha os dados e clique em **Calcular** para obter o risco previsto pelo modelo.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**📚 Dados Escolares**")
        serie_p      = st.selectbox("Série",  ["1ano","2ano","3ano"])
        turno_p      = st.selectbox("Turno",  ["matutino","vespertino","noturno"])
        faltas_p     = st.slider("Faltas Anuais", 0, 60, 10)
        distorcao_p  = st.slider("Distorção Idade-Série (anos)", 0, 5, 0)
        repeticoes_p = st.slider("Reprovações Anteriores", 0, 3, 0)

    with col2:
        st.markdown("**🎓 Desempenho Acadêmico**")
        nota_port_p = st.slider("Nota de Português", 0.0, 10.0, 6.0, 0.5)
        nota_mat_p  = st.slider("Nota de Matemática", 0.0, 10.0, 6.0, 0.5)
        st.markdown("**👤 Demográfico**")
        sexo_p  = st.selectbox("Sexo", ["M", "F"])
        raca_p  = st.selectbox("Raça/Cor", ["Branca","Parda","Preta","Amarela","Indígena"])
        idade_p = st.number_input("Idade", 14, 24, 16)

    with col3:
        st.markdown("**💰 Socioeconômico**")
        trabalha_p  = st.radio("Trabalha?", [0, 1], format_func=lambda x: "Sim" if x else "Não")
        renda_p     = st.number_input("Renda Familiar (R$)", 500, 20000, 1500, 100)
        escol_pai_p = st.selectbox("Escolaridade Pai", [0,1,2,3],
                                    format_func=lambda x: ["Fundamental","Médio","Superior","Pós"][x])
        escol_mae_p = st.selectbox("Escolaridade Mãe", [0,1,2,3],
                                    format_func=lambda x: ["Fundamental","Médio","Superior","Pós"][x])

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔍  Calcular Score de Risco", use_container_width=True, type="primary"):
        if modelo:
            media_p      = (nota_port_p + nota_mat_p) / 2
            baixo_d_p    = int(nota_port_p < 5 or nota_mat_p < 5)
            risco_f_p    = int(faltas_p > 15)
            risco_dist_p = int(distorcao_p >= 2)
            eng_p        = media_p / (faltas_p + 1)

            entrada = pd.DataFrame([{
                "idade": idade_p, "nota_portugues": nota_port_p,
                "nota_matematica": nota_mat_p, "media_notas": media_p,
                "distorcao_idade_serie": distorcao_p, "faltas_anuais": faltas_p,
                "repeticoes_anteriores": repeticoes_p, "renda_familiar": renda_p,
                "trabalha": trabalha_p, "escolaridade_pai": escol_pai_p,
                "escolaridade_mae": escol_mae_p, "baixo_desempenho": baixo_d_p,
                "risco_faltas": risco_f_p, "risco_distorcao": risco_dist_p,
                "engajamento": eng_p, "serie": serie_p, "turno": turno_p,
                "sexo": sexo_p, "raca_cor": raca_p,
            }])

            score = modelo.predict_proba(entrada)[0, 1]
            nivel = "Alto" if score >= threshold else ("Médio" if score >= threshold * 0.65 else "Baixo")
            cor   = COR_ALTO if nivel == "Alto" else (COR_MEDIO if nivel == "Médio" else COR_BAIXO)
            cls   = "danger" if nivel == "Alto" else ("warning" if nivel == "Médio" else "success")

            st.markdown("---")
            r1, r2, r3 = st.columns([1, 1, 2])

            with r1:
                st.markdown(f"""
                <div class="kpi-card {cls}">
                    <div class="kpi-label">Score de Risco</div>
                    <div class="kpi-value" style="color:{cor}">{score:.1%}</div>
                    <div class="kpi-delta">probabilidade de evasão</div>
                </div>""", unsafe_allow_html=True)

            with r2:
                st.markdown(f"""
                <div class="kpi-card {cls}">
                    <div class="kpi-label">Nível de Risco</div>
                    <div class="kpi-value" style="color:{cor};font-size:1.6rem">{nivel}</div>
                    <div class="kpi-delta">threshold: {threshold:.3f}</div>
                </div>""", unsafe_allow_html=True)

            with r3:
                fatores = []
                if distorcao_p >= 2:  fatores.append("🔴 Distorção idade-série ≥ 2 anos")
                if faltas_p > 15:     fatores.append("🔴 Faltas anuais acima de 15 dias")
                if nota_mat_p < 5:    fatores.append("🟡 Nota de Matemática abaixo de 5")
                if nota_port_p < 5:   fatores.append("🟡 Nota de Português abaixo de 5")
                if trabalha_p:        fatores.append("🟡 Aluno trabalhador")
                if repeticoes_p >= 1: fatores.append("🟡 Possui reprovações anteriores")
                if fatores:
                    st.markdown("**Fatores de risco identificados:**")
                    for f in fatores:
                        st.markdown(f"- {f}")
                else:
                    st.success("✅ Nenhum fator de risco crítico identificado.")

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score * 100,
                number={"suffix": "%", "font": {"size": 32, "color": cor}},
                gauge={
                    "axis": {"range": [0, 100], "ticksuffix": "%"},
                    "bar": {"color": cor, "thickness": 0.3},
                    "steps": [
                        {"range": [0,               threshold * 65],  "color": "#E8F8EE"},
                        {"range": [threshold * 65,  threshold * 100], "color": "#FEF3E0"},
                        {"range": [threshold * 100, 100],             "color": "#FDE8E6"},
                    ],
                    "threshold": {
                        "line": {"color": "#1A2E4A", "width": 3},
                        "thickness": 0.8,
                        "value": threshold * 100,
                    },
                },
            ))
            fig.update_layout(
                height=260, margin=dict(t=20, b=10, l=30, r=30),
                paper_bgcolor="white",
                font=dict(family="DM Sans, sans-serif", color="#2C3E50"),
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("⚠️ Modelo não encontrado. Execute `03_modelagem.ipynb` para gerar `models/pipeline_rf.pkl`.")
