import os
import streamlit as st
import pandas as pd
import plotly.express as px

# ─── CONFIGURATION DE LA PAGE ────────────────────────────────────────────────
st.set_page_config(
    page_title="Analyse PySpark Churn", 
    page_icon="🔬", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── CHEMINS ET DOSSIERS ─────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

def load(filename):
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

# ─── DESIGN ET DESIGN SYSTEM (CSS COUTUMIER) ──────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');
  
  /* Reset et Typographie globale */
  html, body, [class*="css"] { 
      font-family: 'Inter', sans-serif; 
      background-color: #f8fafc;
  }
  h1, h2, h3, h4 { 
      font-family: 'Plus Jakarta Sans', sans-serif; 
      color: #0f172a;
  }

  /* Header container */
  .header-container {
      background: white;
      padding: 1.5rem 2rem;
      border-radius: 16px;
      border: 1px solid #e2e8f0;
      margin-bottom: 2rem;
      box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
      animation: fadeIn 0.5s ease both;
  }

  /* Cartes de Graphiques (Chart Cards) */
  .chart-card {
      background: #ffffff; 
      border: 1px solid #e2e8f0;
      border-radius: 16px; 
      padding: 1.5rem;
      margin-bottom: 1.5rem;
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.02), 0 2px 4px -1px rgba(0, 0, 0, 0.02);
      transition: transform 0.2s ease, box-shadow 0.2s ease;
      animation: fadeIn 0.6s ease both;
  }
  .chart-card:hover {
      box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05);
  }

  /* Titres des Sections */
  .section-title {
      font-family: 'Plus Jakarta Sans', sans-serif; 
      font-size: 1.15rem;
      font-weight: 700; 
      color: #1e293b; 
      margin-bottom: 4px;
  }
  .section-sub { 
      font-size: 0.85rem; 
      color: #64748b; 
      margin-bottom: 1.5rem; 
  }

  /* Cartes d'Insights (Livrables Verts) */
  .insight-card {
      background: #f0fdf4; 
      border: 1px solid #bbf7d0;
      border-radius: 12px; 
      padding: 1rem 1.25rem;
      margin-bottom: 0.75rem;
      display: flex; 
      align-items: center; 
      gap: 0.75rem;
      box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.02);
  }
  .insight-icon { font-size: 1.25rem; flex-shrink: 0; }
  .insight-text { font-size: 0.9rem; color: #166534; font-weight: 500; }

  /* Alertes / Warnings */
  .warning-card {
      background: #fefce8; 
      border: 1px solid #fef08a;
      border-radius: 12px; 
      padding: 1.25rem; 
      margin-bottom: 1.5rem;
  }
  .warning-text { font-size: 0.9rem; color: #854d0e; line-height: 1.6; }

  /* Animations */
  @keyframes fadeIn {
      from { opacity: 0; transform: translateY(8px); }
      to { opacity: 1; transform: translateY(0); }
  }
</style>
""", unsafe_allow_html=True)

# ─── HEADER ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-container">
  <h1 style="font-size: 2rem; font-weight: 700; margin: 0 0 6px 0; color: #0f172a;">
    🔬 Analyse PySpark Churn
  </h1>
  <p style="color: #64748b; font-size: 0.95rem; margin: 0;">
    Tableau de bord décisionnel — Données Big Data exportées de l'environnement PySpark
  </p>
</div>
""", unsafe_allow_html=True)

# ─── VÉRIFICATION DES FICHIERS ───────────────────────────────────────────────
required_files = [
    "analyse_churn_global.csv", "analyse_churn_contrat.csv",
    "analyse_churn_internet.csv", "analyse_churn_paiement.csv",
    "analyse_charges_churn.csv", "analyse_churn_services.csv",
    "analyse_insights.csv"
]
missing = [f for f in required_files if not os.path.exists(os.path.join(DATA_DIR, f))]

if missing:
    st.markdown(f"""
    <div class="warning-card">
      <div class="warning-text">
        🔒 <strong>Fichiers de données introuvables.</strong><br>
        Veuillez exécuter le notebook de traitement Spark afin de générer les tables requises :<br>
        <code>notebooks/analyse_pyspark.ipynb</code><br><br>
        <span style="opacity: 0.8">Fichiers manquants : {", ".join(missing)}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─── CHARGEMENT DES DONNÉES ──────────────────────────────────────────────────
df_global   = load("analyse_churn_global.csv")
df_contrat  = load("analyse_churn_contrat.csv")
df_internet = load("analyse_churn_internet.csv")
df_paiement = load("analyse_churn_paiement.csv")
df_charges  = load("analyse_charges_churn.csv")
df_services = load("analyse_churn_services.csv")
df_insights = load("analyse_insights.csv")

# ─── SECTION INSIGHTS CLÉS ───────────────────────────────────────────────────
if df_insights is not None:
    st.markdown("<h3 style='font-size: 1.3rem; font-weight:700; margin-bottom: 1rem;'>💡 Enseignements Stratégiques</h3>", unsafe_allow_html=True)
    for _, row in df_insights.iterrows():
        text = str(row["Insight"]).replace("📌 ", "")
        st.markdown(f"""
        <div class="insight-card">
          <div class="insight-icon">✨</div>
          <div class="insight-text">{text}</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("<div style='margin-bottom: 2rem;'></div>", unsafe_allow_html=True)

# CONFIGURATION COMMUNE PLOTLY
PLOTLY_THEME = dict(
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(family="Inter", size=12, color="#475569"),
    margin=dict(t=10, b=10, l=10, r=10)
)

# ─── BLOC 1 : VUE GLOBALE & CONTRATS ─────────────────────────────────────────
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Volume de Churn Global</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Répartition globale des clients actifs vs partis</div>', unsafe_allow_html=True)
    if df_global is not None:
        df_global["Statut"] = df_global["Churn"].map({"No": "Fidèle (Actif)", "Yes": "Désabonné (Churn)"})
        fig = px.pie(
            df_global, names="Statut", values="count",
            color="Statut",
            color_discrete_map={"Fidèle (Actif)": "#0ec38e", "Désabonné (Churn)": "#f43f5e"},
            hole=0.6
        )
        fig.update_traces(textinfo="percent+label", textfont_size=11, hovertemplate="%{label}<br>%{value} clients<br>%{percent}")
        fig.update_layout(**PLOTLY_THEME, showlegend=False, height=280)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Sensibilité au Type de Contrat</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">% de attrition (churn) selon la nature du contrat</div>', unsafe_allow_html=True)
    if df_contrat is not None:
        fig = px.bar(
            df_contrat.sort_values("Taux_Churn_%", ascending=True),
            x="Taux_Churn_%", y="Contract", orientation="h",
            text="Taux_Churn_%",
            color="Taux_Churn_%",
            color_continuous_scale=["#0ec38e", "#eab308", "#f43f5e"],
        )
        fig.update_traces(texttemplate=" %{text}%", textposition="outside", cliponaxis=False)
        fig.update_layout(**PLOTLY_THEME, coloraxis_showscale=False, height=280)
        fig.update_xaxes(showgrid=False, showline=False, title="", visible=False)
        fig.update_yaxes(showgrid=False, showline=False, title="", tickfont=dict(weight="bold"))
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)

# ─── BLOC 2 : INFRASTRUCTURE & FACTURATION ───────────────────────────────────
col3, col4 = st.columns(2, gap="large")

with col3:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Impact de la Technologie Internet</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Taux de churn constaté par type de connexion</div>', unsafe_allow_html=True)
    if df_internet is not None:
        fig = px.bar(
            df_internet.sort_values("Taux_Churn_%", ascending=False),
            x="InternetService", y="Taux_Churn_%",
            text="Taux_Churn_%",
            color="InternetService",
            color_discrete_sequence=["#f43f5e", "#eab308", "#0ec38e"],
        )
        fig.update_traces(texttemplate="%{text}%", textposition="outside", marker_cornerradius=6)
        fig.update_layout(**PLOTLY_THEME, showlegend=False, height=280)
        fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9", title="", range=[0, max(df_internet["Taux_Churn_%"])*1.15])
        fig.update_xaxes(title="")
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Influence du Mode de Paiement</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Corrélation entre mode de règlement et départ</div>', unsafe_allow_html=True)
    if df_paiement is not None:
        df_paiement["PaymentShort"] = df_paiement["PaymentMethod"].str.replace(" (automatic)", " (auto)", regex=False)
        fig = px.bar(
            df_paiement.sort_values("Taux_Churn_%", ascending=True),
            x="Taux_Churn_%", y="PaymentShort", orientation="h",
            text="Taux_Churn_%",
            color="Taux_Churn_%",
            color_continuous_scale=["#0ec38e", "#eab308", "#f43f5e"],
        )
        fig.update_traces(texttemplate=" %{text}%", textposition="outside", cliponaxis=False)
        fig.update_layout(**PLOTLY_THEME, coloraxis_showscale=False, height=280)
        fig.update_xaxes(showgrid=False, visible=False)
        fig.update_yaxes(showgrid=False, title="")
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)

# ─── BLOC 3 : METRIQUES FINANCIERES & PACKS ──────────────────────────────────
col5, col6 = st.columns(2, gap="large")

with col5:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Comportement Financier & Ancienneté</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Comparaison des moyennes entre clients Actifs et Churned</div>', unsafe_allow_html=True)
    if df_charges is not None:
        df_melt = df_charges.melt(
            id_vars="Churn", value_vars=["Avg_MonthlyCharges", "Avg_Tenure_Mois"],
            var_name="Métrique", value_name="Valeur"
        )
        df_melt["Métrique"] = df_melt["Métrique"].map({"Avg_MonthlyCharges": "Facture Mensuelle ($)", "Avg_Tenure_Mois": "Ancienneté (Mois)"})
        df_melt["Statut"] = df_melt["Churn"].map({"No": "Actif", "Yes": "Churned"})
        
        fig = px.bar(
            df_melt, x="Métrique", y="Valeur", color="Statut", barmode="group",
            text="Valeur",
            color_discrete_map={"Actif": "#0ec38e", "Churned": "#f43f5e"},
        )
        fig.update_traces(texttemplate="%{text:.1f}", textposition="outside", marker_cornerradius=4)
        fig.update_layout(
            **PLOTLY_THEME, 
            legend=dict(title="", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=280
        )
        fig.update_yaxes(gridcolor="#f1f5f9", title="")
        fig.update_xaxes(title="")
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)

with col6:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Risque lié à l\'Absence de Services Optionnels</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Taux de attrition chez les clients n\'ayant pas souscrit</div>', unsafe_allow_html=True)
    if df_services is not None:
        df_services_s = df_services.sort_values("Sans_service_Churn_%", ascending=True)
        df_services_s["ServiceLabel"] = df_services_s["Service"].str.replace("([A-Z])", r" \1", regex=True).str.strip()
        
        fig = px.bar(
            df_services_s, x="Sans_service_Churn_%", y="ServiceLabel", orientation="h",
            text="Sans_service_Churn_%",
            color="Sans_service_Churn_%",
            color_continuous_scale=["#eab308", "#f43f5e"],
        )
        fig.update_traces(texttemplate=" %{text}%", textposition="outside", cliponaxis=False)
        fig.update_layout(**PLOTLY_THEME, coloraxis_showscale=False, height=280)
        fig.update_xaxes(showgrid=False, visible=False)
        fig.update_yaxes(showgrid=False, title="")
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)
