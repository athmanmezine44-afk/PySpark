import os
import streamlit as st
import pandas as pd
import plotly.express as px

# ─── CONFIGURATION DE LA PAGE ────────────────────────────────────────────────
st.set_page_config(page_title="Dashboard Analytics", page_icon="📊", layout="wide")

# ─── CHEMINS COMPATIBLES ─────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "cleanedData.csv")


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


df = load_data()

# ─── DESIGN ET DESIGN SYSTEM (CSS COUTUMIER) ──────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');
  
  /* Reset et Structure de fond */
  html, body, [class*="css"] { 
      font-family: 'Inter', sans-serif; 
      background-color: #f8fafc;
  }
  h1, h2, h3, h4 { 
      font-family: 'Plus Jakarta Sans', sans-serif; 
  }

  /* Cartes d'indicateurs clés (KPI) */
  .metric-card {
      background: #ffffff;
      border: 1px solid #e2e8f0;
      border-radius: 16px;
      padding: 1.5rem;
      display: flex; 
      align-items: flex-start; 
      justify-content: space-between;
      box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.02);
      animation: fadeIn .5s ease both;
  }
  .metric-label {
      font-size: 11px; 
      font-weight: 600;
      color: #64748b; 
      text-transform: uppercase; 
      letter-spacing: .06em;
  }
  .metric-value {
      font-family: 'Plus Jakarta Sans', sans-serif;
      font-size: 2.2rem; 
      font-weight: 700; 
      color: #0f172a; 
      margin: 0.2rem 0 0.1rem;
  }
  .metric-sub { font-size: 12px; color: #64748b; font-weight: 500; }
  .metric-icon {
      width: 46px; 
      height: 46px; 
      border-radius: 12px;
      display: flex; 
      align-items: center; 
      justify-content: center;
      font-size: 1.4rem; 
      flex-shrink: 0;
  }

  /* Cartes contenant les graphiques */
  .chart-card {
      background: #ffffff; 
      border: 1px solid #e2e8f0;
      border-radius: 16px; 
      padding: 1.5rem;
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.02);
      animation: fadeIn .5s ease both;
  }
  .chart-title {
      font-family: 'Plus Jakarta Sans', sans-serif;
      font-size: 1.1rem; 
      font-weight: 700; 
      color: #0f172a; 
      margin-bottom: 2px;
  }
  .chart-sub { font-size: 12px; color: #64748b; margin-bottom: 1.2rem; }
  
  @keyframes fadeIn {
      from { opacity: 0; transform: translateY(10px); }
      to { opacity: 1; transform: translateY(0); }
  }
</style>
""", unsafe_allow_html=True)

# ─── HEADER ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="animation: fadeIn .4s ease both; margin-bottom: 1.8rem;">
  <h1 style="font-size: 1.9rem; font-weight: 700; margin-bottom: 4px; color: #0f172a;">
    📊 Dashboard Analytics
  </h1>
  <p style="color: #64748b; font-size: 0.95rem; margin: 0;">
    Suivi en temps réel de la performance client et de la distribution du parc abonnés
  </p>
</div>
""", unsafe_allow_html=True)

# ─── CALCUL DES MÉTRIQUES EN TEMPS RÉEL ───────────────────────────────────────
total_clients = len(df)
churn_count = int((df["Churn"] == "Yes").sum())
active_count = total_clients - churn_count
churn_rate = churn_count / total_clients * 100
retention_rate = active_count / total_clients * 100

metrics = [
    ("👥", "Total Clients", f"{total_clients:,}".replace(",", " "), "Base complète brute", "#f0fdf4", "#10b981"),
    ("📉", "Taux de Churn", f"{churn_rate:.1f} %", f"{churn_count:,} clients désabonnés", "#fff5f5", "#f43f5e"),
    ("🛡️", "Clients Actifs", f"{active_count:,}".replace(",", " "), f"{retention_rate:.1f}% taux de fidélité", "#e0f2fe", "#0284c7"),
]

c1, c2, c3 = st.columns(3, gap="medium")
for col, (icon, label, value, sub, bg, color) in zip([c1, c2, c3], metrics):
    col.markdown(f"""
    <div class="metric-card">
      <div>
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-sub">{sub}</div>
      </div>
      <div class="metric-icon" style="background: {bg}; color: {color};">{icon}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

# CONFIGURATION PLOTLY PARTAGÉE
PLOTLY_THEME = dict(
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(family="Inter", size=12, color="#64748b"),
    margin=dict(t=10, b=10, l=10, r=10)
)

# ─── GRAPHIQUES LIGNE 1 ──────────────────────────────────────────────────────
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("""
    <div class="chart-card">
      <div class="chart-title">Répartition Globale du Churn</div>
      <div class="chart-sub">Volume brut de clients d'après leur statut contractuel</div>
    </div>
    """, unsafe_allow_html=True)
    
    churn_df = (
        df["Churn"]
        .value_counts()
        .rename(index={"No": "Actifs (Fidèles)", "Yes": "Désabonnés (Churn)"})
        .rename_axis("Statut")
        .reset_index(name="Clients")
    )
    
    fig1 = px.bar(
        churn_df, x="Statut", y="Clients",
        color="Statut",
        color_discrete_map={"Actifs (Fidèles)": "#10b981", "Désabonnés (Churn)": "#f43f5e"},
        text="Clients",
    )
    fig1.update_traces(textposition="outside", cliponaxis=False, marker_cornerradius=6)
    fig1.update_layout(**PLOTLY_THEME, showlegend=False, height=280)
    fig1.update_yaxes(showgrid=True, gridcolor="#f1f5f9", title="", range=[0, max(churn_df["Clients"])*1.15])
    fig1.update_xaxes(title="")
    
    st.plotly_chart(fig1, use_container_width=True, config={'displayModeBar': False})

with col2:
    st.markdown("""
    <div class="chart-card">
      <div class="chart-title">Histogramme des Charges Mensuelles</div>
      <div class="chart-sub">Analyse de la distribution volumétrique des facturations ($)</div>
    </div>
    """, unsafe_allow_html=True)
    
    fig2 = px.histogram(
        df, x="MonthlyCharges", nbins=30,
        color_discrete_sequence=["#0f172a"],
    )
    fig2.update_traces(marker_line_width=0, opacity=0.9)
    fig2.update_layout(**PLOTLY_THEME, height=280)
    fig2.update_yaxes(showgrid=True, gridcolor="#f1f5f9", title="Volume de clients")
    fig2.update_xaxes(showgrid=False, title="Montant mensuel facturé ($)")
    
    st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})

st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

# ─── GRAPHIQUE LIGNE 2 : DISTRIBUTION TENURE ─────────────────────────────────
st.markdown("""
<div class="chart-card">
  <div class="chart-title">Structure de l'Ancienneté du Parc</div>
  <div class="chart-sub">Segmentation des clients par cohortes de durée d'abonnement (en mois)</div>
</div>
""", unsafe_allow_html=True)

bins = [0, 12, 24, 36, 48, 60, 72]
labels = ["0 - 12 Mois", "13 - 24 Mois", "25 - 36 Mois", "37 - 48 Mois", "49 - 60 Mois", "61 - 72 Mois"]
tenure_buckets = pd.cut(df["tenure"], bins=bins, labels=labels, include_lowest=True)
tenure_df = tenure_buckets.value_counts().reindex(labels).rename_axis("Tenure").reset_index(name="Clients")

fig3 = px.bar(
    tenure_df, x="Tenure", y="Clients",
    color_discrete_sequence=["#eab308"],
    text="Clients",
)
fig3.update_traces(textposition="outside", cliponaxis=False, marker_cornerradius=6)
fig3.update_layout(**PLOTLY_THEME, showlegend=False, height=290)
fig3.update_yaxes(showgrid=True, gridcolor="#f1f5f9", title="", range=[0, max(tenure_df["Clients"])*1.15])
fig3.update_xaxes(title="")

st.plotly_chart(fig3, use_container_width=True, config={'displayModeBar': False})
