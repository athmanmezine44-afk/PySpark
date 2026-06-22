import os
import streamlit as st
import pandas as pd
import joblib

# ─── CONFIGURATION DE LA PAGE ────────────────────────────────────────────────
st.set_page_config(page_title="Prédiction Churn", page_icon="✨", layout="centered")

# ─── CHEMINS & MODÈLE ────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "random_forest_churn_model.pkl")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()
MODEL_COLUMNS = list(model.feature_names_in_)

CAT_COLUMNS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]


def encode_input(raw: dict) -> pd.DataFrame:
    """Reproduit exactement le one-hot encoding utilisé à l'entraînement."""
    row = pd.DataFrame([raw])
    encoded = pd.get_dummies(row, columns=CAT_COLUMNS)
    encoded = encoded.reindex(columns=MODEL_COLUMNS, fill_value=0)
    return encoded


# ─── DESIGN ET DESIGN SYSTEM (CSS COUTUMIER) ──────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');
  
  /* Reset et Typographie globale */
  html, body, [class*="css"] { 
      font-family: 'Inter', sans-serif; 
      background-color: #f8fafc;
  }
  h1, h2, h3, h4 { 
      font-family: 'Plus Jakarta Sans', sans-serif; 
      color: #0f172a;
  }

  /* Formulaire principal */
  .form-card {
      background: #ffffff;
      border: 1px solid #e2e8f0;
      border-radius: 20px;
      padding: 2rem;
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.02), 0 2px 4px -1px rgba(0, 0, 0, 0.02);
      animation: fadeIn 0.4s ease both;
  }

  /* Libellés de sections internes */
  .section-label {
      font-family: 'Plus Jakarta Sans', sans-serif;
      font-size: 13px; 
      font-weight: 700; 
      text-transform: uppercase;
      letter-spacing: .06em; 
      color: #0ea5e9; 
      margin: 1.5rem 0 0.8rem;
      border-bottom: 1px solid #f1f5f9;
      padding-bottom: 4px;
  }
  .section-label:first-child { margin-top: 0; }

  /* Hack pour harmoniser les inputs Streamlit nativement */
  .stSelectbox label, .stNumberInput label {
      font-size: 11px !important; 
      font-weight: 600 !important;
      text-transform: uppercase !important; 
      letter-spacing: .05em !important;
      color: #64748b !important;
  }

  /* Bouton d'action principal */
  .stButton > button[kind="primary"] {
      background: #0f172a !important; 
      border: none !important; 
      border-radius: 12px !important;
      font-family: 'Plus Jakarta Sans', sans-serif;
      font-weight: 600 !important; 
      height: 3rem;
      font-size: 1rem !important;
      box-shadow: 0 4px 12px rgba(15, 23, 42, 0.15); 
      transition: all .2s ease;
  }
  .stButton > button[kind="primary"]:hover { 
      background: #1e293b !important; 
      transform: translateY(-1px);
      box-shadow: 0 6px 20px rgba(15, 23, 42, 0.2);
  }

  /* Bloc Résultat : Alerte Churn */
  .result-churn {
      background: #fff5f5; 
      border: 1px solid #feb2b2;
      border-radius: 16px; 
      padding: 1.5rem; 
      animation: popIn 0.4s cubic-bezier(.34,1.56,.64,1) both;
  }

  /* Bloc Résultat : Client Stable */
  .result-safe {
      background: #f0fdf4; 
      border: 1px solid #bbf7d0;
      border-radius: 16px; 
      padding: 1.5rem; 
      animation: popIn 0.4s cubic-bezier(.34,1.56,.64,1) both;
  }

  /* Composants du résultat */
  .result-icon {
      width: 56px; 
      height: 56px; 
      border-radius: 14px;
      display: flex; 
      align-items: center; 
      justify-content: center;
      font-size: 1.6rem; 
      flex-shrink: 0;
  }
  .result-title { 
      font-family: 'Plus Jakarta Sans', sans-serif; 
      font-size: 1.25rem; 
      font-weight: 700; 
      margin: 0 0 4px; 
  }
  .result-score { font-size: .9rem; color: #475569; margin-bottom: 6px; }
  .result-desc  { font-size: .88rem; color: #64748b; line-height: 1.6; }
  
  /* Jauge de probabilité */
  .score-bar-bg { background: #e2e8f0; border-radius: 999px; height: 10px; margin: 1rem 0 .5rem; overflow: hidden; }
  .score-bar-fill { height: 100%; border-radius: 999px; transition: width .8s ease; }

  /* Animations */
  @keyframes fadeIn { from{opacity:0;transform:translateY(10px)} to{opacity:1;transform:translateY(0)} }
  @keyframes popIn { from{opacity:0;transform:scale(.97)} to{opacity:1;transform:scale(1)} }
</style>
""", unsafe_allow_html=True)

# ─── HEADER ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="animation: fadeIn .4s ease both; margin-bottom: 1.5rem;">
  <h1 style="font-size: 2rem; font-weight: 700; margin-bottom: 6px; color: #0f172a;">
    ✨ Prédiction du Risque Churn
  </h1>
  <p style="color: #64748b; font-size: 0.95rem; margin: 0;">
    Saisissez le profil et les données de consommation du client pour évaluer sa probabilité d'attrition.
  </p>
</div>
""", unsafe_allow_html=True)

# ─── FORMULAIRE CLIENT ───────────────────────────────────────────────────────
st.markdown('<div class="form-card">', unsafe_allow_html=True)

# Section : Identité & Démographie
st.markdown('<div class="section-label">👤 Profil Démographique</div>', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)
with col1:
    gender = st.selectbox("Genre", ["Male", "Female"], format_func=lambda x: {"Male": "Homme", "Female": "Femme"}[x])
with col2:
    senior = st.selectbox("Senior (≥ 65 ans)", ["No", "Yes"], format_func=lambda x: {"Yes": "Oui", "No": "Non"}[x])
with col3:
    partner = st.selectbox("En couple", ["No", "Yes"], format_func=lambda x: {"Yes": "Oui", "No": "Non"}[x])
with col4:
    dependents = st.selectbox("Personnes à charge", ["No", "Yes"], format_func=lambda x: {"Yes": "Oui", "No": "Non"}[x])

# Section : Situation Financière
st.markdown('<div class="section-label">💳 Informations du Compte & Contrat</div>', unsafe_allow_html=True)
col5, col6, col7 = st.columns(3)
with col5:
    tenure = st.number_input("Ancienneté (mois)", 0, 100, 12)
with col6:
    monthly = st.number_input("Charges mensuelles ($)", 0.0, 200.0, 70.0, 0.01, format="%.2f")
with col7:
    total = st.number_input("Charges totales ($)", 0.0, 10000.0, 840.0, 0.01, format="%.2f")

col8, col9, col10 = st.columns(3)
with col8:
    contract = st.selectbox(
        "Type de contrat", ["Month-to-month", "One year", "Two year"],
        format_func=lambda x: {"Month-to-month": "Mensuel engagé", "One year": "Engagé 1 an", "Two year": "Engagé 2 ans"}[x])
with col9:
    paperless = st.selectbox("Facture dématérialisée", ["No", "Yes"], format_func=lambda x: {"Yes": "Oui", "No": "Non"}[x])
with col10:
    payment = st.selectbox(
        "Mode de règlement",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        format_func=lambda x: {
            "Electronic check": "Prélèvement unique", "Mailed check": "Chèque postal",
            "Bank transfer (automatic)": "Virement auto.", "Credit card (automatic)": "Carte bancaire auto.",
        }[x])

# Section : Services Souscrits
st.markdown('<div class="section-label">🌐 Services & Abonnements</div>', unsafe_allow_html=True)
col11, col12 = st.columns(2)
with col11:
    phone_service = st.selectbox("Abonnement Téléphonique", ["Yes", "No"], format_func=lambda x: {"Yes": "Oui", "No": "Non"}[x])

if phone_service == "No":
    multiple_lines = "No phone service"
else:
    with col12:
        multiple_lines = st.selectbox("Lignes téléphoniques multiples", ["No", "Yes"], format_func=lambda x: {"Yes": "Oui", "No": "Non"}[x])

internet_service = st.selectbox("Type d'accès Internet", ["DSL", "Fiber optic", "No"],
                                 format_func=lambda x: {"DSL": "Ligne DSL", "Fiber optic": "Fibre optique", "No": "Aucun abonnement internet"}[x])

if internet_service == "No":
    online_security = online_backup = device_protection = "No internet service"
    tech_support = streaming_tv = streaming_movies = "No internet service"
else:
    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
    col13, col14, col15 = st.columns(3)
    with col13:
        online_security = st.selectbox("Sécurité Internet optionnelle", ["No", "Yes"], format_func=lambda x: {"Yes": "Inclus", "No": "Non"}[x])
    with col14:
        online_backup = st.selectbox("Sauvegarde Cloud", ["No", "Yes"], format_func=lambda x: {"Yes": "Inclus", "No": "Non"}[x])
    with col15:
        device_protection = st.selectbox("Assurance Appareil", ["No", "Yes"], format_func=lambda x: {"Yes": "Inclus", "No": "Non"}[x])

    col16, col17, col18 = st.columns(3)
    with col16:
        tech_support = st.selectbox("Assistance technique dédiée", ["No", "Yes"], format_func=lambda x: {"Yes": "Inclus", "No": "Non"}[x])
    with col17:
        streaming_tv = st.selectbox("Service TV par internet", ["No", "Yes"], format_func=lambda x: {"Yes": "Inclus", "No": "Non"}[x])
    with col18:
        streaming_movies = st.selectbox("Catalogue Films VOD", ["No", "Yes"], format_func=lambda x: {"Yes": "Inclus", "No": "Non"}[x])

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

# Bouton de soumission
predict_btn = st.button("✨ Calculer la probabilité de risque", type="primary", use_container_width=True)

# ─── CALCUL & RÉSULTAT DE LA PRÉDICTION ─────────────────────────────────────
if predict_btn:
    raw_input = {
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "gender": gender,
        "Partner": partner,
        "Dependents": dependents,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
    }

    with st.spinner("Modèle prédictif en action..."):
        X = encode_input(raw_input)
        prediction = model.predict(X)[0]
        score = float(model.predict_proba(X)[0][1])

    churn = prediction == 1
    pct = int(round(score * 100))
    
    # Configuration des indicateurs dynamiques
    bar_color = "#f43f5e" if churn else "#10b981"
    icon_bg = "#fee2e2" if churn else "#d1fae5"
    icon = "🔥" if churn else "🛡️"
    title = "Alerte : Risque Critique de Churn" if churn else "Profil Client Stable"
    desc = ("Ce client affiche des signaux forts de désengagement. Il est fortement conseillé de lui proposer un geste de rétention commercial ou de l'appeler rapidement."
            if churn else
            "La probabilité de rupture de contrat de ce client est particulièrement basse. Le client est en phase avec vos services.")
    title_color = "#991b1b" if churn else "#065f46"

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="{'result-churn' if churn else 'result-safe'}">
      <div style="display: flex; align-items: flex-start; gap: 1.25rem;">
        <div class="result-icon" style="background: {icon_bg};">{icon}</div>
        <div style="flex: 1;">
          <div class="result-title" style="color: {title_color};">{title}</div>
          <div class="result-score">Score de probabilité d'attrition : <strong style="color: #0f172a; font-size: 1rem;">{score:.2f}</strong></div>

          <div class="score-bar-bg">
            <div class="score-bar-fill" style="width: {pct}%; background: {bar_color};"></div>
          </div>
          <div style="display: flex; justify-content: space-between; font-size: 11px; color: #64748b; font-weight: 500;">
            <span>0.0 (Fidèle)</span><span>Risque calculé : {pct}%</span><span>1.0 (Désabonné)</span>
          </div>

          <div class="result-desc" style="margin-top: 0.8rem; border-top: 1px solid rgba(0,0,0,0.04); padding-top: 8px;">{desc}</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
