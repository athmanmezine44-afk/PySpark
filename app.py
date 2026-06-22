import streamlit as st

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="TelcoChurn AI",
    page_icon="📡",
    layout="centered"
)

# ─── CSS (cohérent avec le reste de l'app) ───────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;600;700&display=swap');
  html, body, [class*="css"] { font-family:'Inter',sans-serif; }
  h1,h2,h3 { font-family:'Space Grotesk',sans-serif; }

  [data-testid="stSidebar"] { background:#0f1e1c !important; }
  [data-testid="stSidebar"] * { color:#b0c4c2 !important; }

  .hero-card {
    background:#fff;
    border:1px solid #e5e7eb;
    border-radius:20px;
    padding:2rem;
    animation: fadeIn .5s ease both;
  }
  .pill {
    display:inline-block;
    background:#ecfdf5;
    color:#059669;
    font-size:11px;
    font-weight:600;
    padding:4px 10px;
    border-radius:999px;
    margin:3px 6px 3px 0;
  }
  @keyframes fadeIn {
    from{opacity:0;transform:translateY(12px)}
    to{opacity:1;transform:translateY(0)}
  }
</style>
""", unsafe_allow_html=True)

# =========================
# CONTENU
# =========================
st.title("📡 Telco Customer Churn Prediction")

st.markdown('<div class="hero-card">', unsafe_allow_html=True)
st.write("""
### 👋 Bienvenue

Cette application permet d'analyser et de prédire le risque de départ (*churn*)
des clients d'un opérateur télécom, à partir d'un modèle de Machine Learning
entraîné sur l'historique réel des abonnés.

Utilisez le menu à gauche pour :
- **📊 Dashboard** — explorer les statistiques clients (churn, ancienneté, facturation)
- **✨ Prédiction** — estimer le risque de churn d'un client donné
""")
st.markdown("""
<div style="margin-top:1rem;">
  <span class="pill">🤖 Random Forest</span>
  <span class="pill">📈 Données réelles</span>
  <span class="pill">⚡ Streamlit</span>
</div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
st.success("✔ Application prête à être utilisée")
