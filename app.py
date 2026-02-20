import streamlit as st
import pdfplumber
import os
from openai import OpenAI

st.set_page_config(
    page_title="CV Adapté à l’Offre",
    page_icon="📄",
    layout="centered"
)

# =========================
# HEADER PROPRE
# =========================

st.markdown("""
<style>

/* Fond général */
body {
    background-color: #f9fafc;
}

/* Container principal plus propre */
.block-container {
    padding-top: 4rem;
    padding-bottom: 2rem;
}

/* Titres */
.main-title {
    font-size: 34px;
    font-weight: 700;
    color: #111827;
    text-align: center;
}

.subtitle {
    font-size: 16px;
    color: #6b7280;
    text-align: center;
    margin-bottom: 0.5rem;
}

.beta {
    font-size: 14px;
    color: #f59e0b;
    text-align: center;
    margin-bottom: 2rem;
}

/* Boutons */
.stButton > button {
    background-color: #2563eb;
    color: white;
    border-radius: 8px;
    padding: 0.6rem 1.2rem;
    font-weight: 600;
    border: none;
}

.stButton > button:hover {
    background-color: #1e40af;
}

/* Séparateurs */
hr {
    margin-top: 2rem;
    margin-bottom: 2rem;
}

/* Supprimer footer Streamlit */
footer {visibility: hidden;}

</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">CV Adapté à l’Offre d’Emploi</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Optimise ton CV, ta lettre et ton mail en quelques secondes.</p>', unsafe_allow_html=True)
st.markdown('<p class="beta">🚀 Version bêta – usage limité gratuit</p>', unsafe_allow_html=True)

st.markdown("---")

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])



# -----------------------------
# Initialisation du session_state
# -----------------------------
if "paid" not in st.session_state:
    st.session_state.paid = True

if "premium_cv_used" not in st.session_state:
    st.session_state.premium_cv_used = False

if "premium_lm_used" not in st.session_state:
    st.session_state.premium_lm_used = False

if "premium_mail_used" not in st.session_state:
    st.session_state.premium_mail_used = False

if "cv_status" not in st.session_state:
    st.session_state.cv_status = "idle"

if "lm_status" not in st.session_state:
    st.session_state.lm_status = "idle"

if "mail_status" not in st.session_state:
    st.session_state.mail_status = "idle"

if "cv_result" not in st.session_state:
    st.session_state.cv_result = ""
if "lm_result" not in st.session_state:
    st.session_state.lm_result = ""
if "mail_result" not in st.session_state:
    st.session_state.mail_result = ""

# ----------------------------
# CONFIGURATION GLOBALE
# ----------------------------
API_ACTIVE = True        # IA activée
FREE_ACCESS = True       # accès gratuit temporaire (2 semaines)
PAYMENT_ENABLED = False # paiement désactivé pendant la période gratuite

st.set_page_config(page_title="CV adapté à l’offre", layout="centered")

# ----------------------------
# CONFIGURATION PAIEMENT
# ----------------------------
PAYMENT_ENABLED = False  # Passera à True quand Maishapay sera actif

# ============================
# STATE
# ============================
if "step" not in st.session_state:
    st.session_state.step = 1

if "job_offer_text" not in st.session_state:
    st.session_state.job_offer_text = ""

# ============================
# FUNCTIONS
# ============================
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

def analyze_cv_vs_offer(cv_text: str, offer_text: str):
    cv = cv_text.lower()
    offer = offer_text.lower()

    words = offer.split()
    keywords = list(set([w for w in words if len(w) > 4]))

    matched = [kw for kw in keywords if kw in cv]
    missing = [kw for kw in keywords if kw not in cv]

    score = int((len(matched) / len(keywords)) * 100) if keywords else 0

    vague_terms = [
        "participation", "aide", "assistance", "contribution",
        "responsable de", "chargé de", "support", "collaboration",
        "implication", "gestion de"
    ]

    risk_zones = []
    for line in cv_text.split("\n"):
        l = line.lower()
        if any(term in l for term in vague_terms) and not any(c.isdigit() for c in l):
            risk_zones.append(line.strip())

    return score, matched[:10], missing[:10], risk_zones[:10]

# ============================
# UI
# ============================
st.title("CV adapté à l’offre d’emploi")

# ----------------------------
# ÉTAPE 1 — OFFRE
# ----------------------------
st.markdown("## 🔹 Étape 1 — Offre d’emploi (obligatoire)")

st.info(
    "👉 *Copie-colle le texte complet de l’offre d’emploi*.\n\n"
    "Tu peux le copier depuis LinkedIn, un site d’entreprise ou un PDF.\n"
    "⚠️ Il n’est pas nécessaire de tout réécrire à la main."
)

job_input = st.text_area(
    "Texte de l’offre d’emploi",
    height=260,
    disabled=(st.session_state.step > 1)
)

col1, col2 = st.columns(2)

with col1:
    if st.button("✅ Valider l’offre", disabled=(st.session_state.step > 1)):
        if job_input.strip():
            st.session_state.job_offer_text = job_input.strip()
            st.session_state.step = 2
            st.rerun()
        else:
            st.error("Le texte de l’offre est obligatoire.")

with col2:
    if st.button("🔁 Modifier l’offre", disabled=(st.session_state.step == 1)):
        st.session_state.step = 1
        st.session_state.job_offer_text = ""
        st.rerun()

# ----------------------------
# ÉTAPE 2 — CV
# ----------------------------
st.markdown("## 🔹 Étape 2 — CV")

if st.session_state.step < 2:
    st.info("Valide d’abord l’offre pour continuer.")
else:
    uploaded_cv = st.file_uploader("Téléverse ton CV (PDF uniquement)", type=["pdf"])

    if uploaded_cv:
        with st.spinner("Lecture du CV..."):
            cv_text = extract_text_from_pdf(uploaded_cv)

        if cv_text:
            st.session_state.cv_text = cv_text
            st.success("CV analysé avec succès ✅")

            with st.expander("👀 Aperçu du texte du CV"):
                st.write(cv_text[:1500] + ("..." if len(cv_text) > 1500 else ""))

import os
import openai

def generate_ai_analysis(job_offer_text: str, cv_text: str):
    """
    Génère l'analyse gratuite via IA (FR uniquement).
    Si aucune clé OpenAI n'est définie, retourne None.
    """

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    openai.api_key = api_key

    prompt = f"""
Tu es un recruteur professionnel expérimenté, habitué à analyser des CV
et à les comparer précisément à des offres d’emploi.

Ta mission est d’analyser UNIQUEMENT le CV et l’offre fournis ci-dessous.
Tu ne dois jamais faire de généralités.
Chaque remarque doit être directement liée à CETTE offre et à CE CV.

IMPORTANT :
- Ta réponse doit être entièrement en FRANÇAIS.
- Tu dois expliquer clairement tes constats comme si tu parlais à un candidat.
- Tu dois être honnête, pédagogique et constructif.
- Tu ne dois jamais lister de mots isolés : uniquement des phrases complètes.
- Tu ne dois PAS réécrire le CV.
- Tu ne dois PAS proposer de version adaptée du CV.

---

OFFRE D’EMPLOI :
{job_offer_text}

---

CV DU CANDIDAT :
{cv_text}

---

STRUCTURE OBLIGATOIRE DE TA RÉPONSE :

1. Score global de compatibilité (0–100 %)
Explique en une ou deux phrases comment ce score a été estimé.

2. Analyse des compétences et critères de l’offre

3. Analyse de l’expérience et des missions

4. Clarté du CV – zones floues ou à risque

5. Opportunité d'optimisation premium : 
Cette section doit :
- Suggérer qu’une optimisation stratégique du CV est possible
- Mentionner l’amélioration du score ATS et de l’alignement avec l’offre
- Ne donner aucun conseil concret, aucun exemple, ni mot-clé précis
- Créer un sentiment de potentiel inexploité
- Inciter subtilement à activer le mode Premium

Le ton doit être professionnel, crédible et orienté performance.

Ne rajoute aucune section.
Ne conclus pas avec une phrase commerciale.
"""

def generate_ai_analysis(job_offer_text, cv_text):
    prompt = f"""
Tu es un recruteur expérimenté.

Analyse le CV ci-dessous par rapport à l’offre d’emploi.

OFFRE :
{job_offer_text}

CV :
{cv_text}

STRUCTURE OBLIGATOIRE DE TA RÉPONSE :

1. Score global de compatibilité (0–100 %)
   Explique brièvement en une ou deux phrases comment ce score a été estimé.

2. Analyse des compétences et critères de l’offre

3. Analyse de l’expérience et des missions

4. Clarté du CV – zones floues ou à risque

5. Opportunité d'optimisation premium :
Cette section doit :
- Suggérer qu’une optimisation stratégique du CV est possible
- Mentionner l’amélioration du score ATS et de l’alignement avec l’offre
- Ne donner aucun conseil concret, aucun exemple, ni mot-clé précis
- Créer un sentiment de potentiel inexploité
- Inciter subtilement à activer le mode Premium

Le ton doit être professionnel, crédible et orienté performance.

Ne rajoute aucune section.
Ne conclus pas avec une phrase commerciale.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.4
    )

    return response.choices[0].message.content
    
def generate_premium_cv(job_offer_text, cv_text, output_language):
    import time 
    start_time = time.time()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    openai.api_key = api_key

    prompt = f"""
Tu es un recruteur professionnel et un expert en rédaction de CV optimisés
pour les processus de recrutement modernes.

La langue de sortie doit être : {output_language}.

OFFRE D’EMPLOI :
{job_offer_text}

CV ORIGINAL :
{cv_text}

MISSION :
Proposer UNE VERSION ADAPTÉE du CV, sans inventer d’expérience,
en suivant strictement cette structure :
- Titre professionnel
- Profil professionnel
- Compétences clés
- Expériences professionnelles
- Formation
- Autres informations pertinentes

Fournis uniquement le CV adapté final.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )

    end_time = time.time()
    print("⏱️ Temps API :", round(end_time - start_time, 2), "secondes")

    return response.choices[0].message.content


def generate_premium_lm(job_offer_text, cv_text, output_language):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    openai.api_key = api_key

    prompt = f"""
Tu es un recruteur professionnel.

La langue de sortie doit être : {output_language}.

OFFRE D’EMPLOI :
{job_offer_text}

CV :
{cv_text}

MISSION :

Rédiger UNE LETTRE DE MOTIVATION professionnelle, personnalisée et crédible.

OBLIGATIONS STRICTES :

1. Utiliser des éléments précis du CV (compétences, expériences, réalisations).
2. Faire explicitement le lien entre au moins 2 exigences de l’offre et le profil du candidat.
3. Mentionner des exemples concrets (missions réalisées, résultats obtenus, outils maîtrisés).
4. Interdire toute phrase générique ou vague (ex: "je suis motivé", "je suis dynamique", etc.).
5. Ne jamais inventer de compétences absentes du CV.

Structure obligatoire :
- En-tête
- Objet
- Introduction personnalisée
- Corps structuré en 2 à 3 paragraphes argumentés
- Conclusion cohérente
- Formule de politesse professionnelle et signature

Le ton doit être professionnel, naturel et crédible.
La lettre doit sembler écrite spécifiquement pour cette offre.

Fournis uniquement la lettre finale.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )

    return response.choices[0].message.content


def generate_premium_mail(job_offer_text, cv_text, output_language):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    openai.api_key = api_key

    prompt = f"""
Tu es un recruteur professionnel.

La langue de sortie doit être : {output_language}.

OFFRE D’EMPLOI :
{job_offer_text}

CV :
{cv_text}

MISSION :
Rédiger UN MAIL DE CANDIDATURE professionnel,
avec :
- Objet
- Message clair
- Mention des pièces jointes
- Signature

Fournis uniquement le mail final.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )

    return response.choices[0].message.content

    def redirect_to_maishapay():
    # Lien Maishapay à activer plus tard
        payment_url = "https://pay.maishapay.net/checkout-placeholder"
        st.markdown(
        f"""
        <meta http-equiv="refresh" content="0;url={payment_url}">
        """,
        unsafe_allow_html=True
    )

# ----------------------------
# ÉTAPE 3 — ANALYSE GRATUITE (IA)
# ----------------------------
if st.session_state.step >= 2 and "cv_text" in st.session_state:

    st.markdown("## 🔹 Étape 3 — Analyse gratuite CV ↔️ Offre")

    st.write(
        "Cette analyse est réalisée par une intelligence artificielle, "
        "en se basant uniquement sur ton CV et sur l’offre d’emploi fournie."
    )

    # Initialisation si nécessaire
    if "analysis_status" not in st.session_state:
        st.session_state.analysis_status = "idle"
        st.session_state.analysis_result = ""

    if st.session_state.analysis_status == "done":
        st.success("Analyse terminée ✅")
        st.markdown(st.session_state.analysis_result)

    elif st.session_state.analysis_status == "processing":
        st.button("Analyse en cours…", disabled=True)

    elif st.session_state.analysis_status == "idle":
        if st.button("🔍 Lancer l’analyse", key="gen_analysis"):
            st.session_state.analysis_status = "processing"
            st.session_state.analysis_result = ""
            st.rerun()

    # Lancer génération si processing
    if st.session_state.analysis_status == "processing" and st.session_state.analysis_result == "":
        with st.spinner("Analyse en cours..."):
            analysis = generate_ai_analysis(
                st.session_state.job_offer_text,
                st.session_state.cv_text
            )

        if analysis is None:
            st.warning("Clé OpenAI manquante.")
            st.session_state.analysis_status = "idle"
        else:
            st.session_state.analysis_result = analysis
            st.session_state.analysis_status = "done"

            # Extraire score
            import re
            match = re.search(r"(\d+)\s*%", analysis)
            if match:
                st.session_state.compatibility_score = int(match.group(1))

        st.rerun()
# ----------------------------
# ÉTAPE 4 — MODE TEST GRATUIT
# ----------------------------

# --- Langue des documents générés (sorties uniquement) ---
st.markdown("### 🌍 Langue des documents générés :")

if "output_language" not in st.session_state:
    st.session_state.output_language = "Français"

st.caption(
    f"Langue actuelle : {st.session_state.output_language} "
    "(cela affecte uniquement les documents générés)"
)

col1, col2, col3 = st.columns([1, 1, 4])  # boutons plus petits

def _reset_outputs_for_regen():
    # On force UNIQUEMENT la régénération des documents (pas l’analyse)
    st.session_state.cv_status = "processing"
    st.session_state.cv_result = ""

    st.session_state.lm_status = "processing"
    st.session_state.lm_result = ""

    st.session_state.mail_status = "processing"
    st.session_state.mail_result = ""

with col1:
    clicked_fr = st.button("Français", key="lang_fr", use_container_width=True, type="secondary")
    if clicked_fr and st.session_state.output_language != "Français":
        st.session_state.output_language = "Français"
        _reset_outputs_for_regen()

with col2:
    clicked_en = st.button("Anglais", key="lang_en", use_container_width=True, type="secondary")
    if clicked_en and st.session_state.output_language != "Anglais":
        st.session_state.output_language = "Anglais"
        _reset_outputs_for_regen()

output_language = st.session_state.output_language

# =========================================================
# 📄 CV ADAPTÉ
# =========================================================

st.markdown("### 📄 CV adapté à l’offre")

if st.session_state.cv_status == "done":
    st.success("CV adapté généré ✅")
    st.text_area("Contenu du CV adapté", st.session_state.cv_result, height=450)

    st.download_button(
    "⬇️ Télécharger le CV adapté",
    st.session_state.cv_result,
    file_name="CV_adapte.txt",
    mime="text/plain"
    )

elif st.session_state.cv_status == "processing":
    st.button("Génération en cours…", disabled=True)

elif st.session_state.cv_status == "idle":

    st.markdown('<div class="card">', unsafe_allow_html=True)

    # Vérifier le score avant d’autoriser la génération
    if (
        "compatibility_score" in st.session_state
        and st.session_state.compatibility_score is not None
        and st.session_state.compatibility_score < 50
    ):
        st.error(
            "❗ Votre CV présente une compatibilité inférieure à 50% avec cette offre.\n\n"
            "Pour des raisons d’intégrité professionnelle, nous ne pouvons pas modifier "
            "votre CV lorsque l’écart est trop important.\n\n"
            "Nous ne pouvons ni inventer ni ajouter des compétences absentes de votre profil."
        )

        st.button("Adapter mon CV", disabled=True)

    else:

        if st.button("Adapter mon CV", key="gen_cv"):
            st.session_state.cv_status = "processing"
            st.session_state.cv_result = ""

    st.markdown('</div>', unsafe_allow_html=True)

if st.session_state.cv_status == "processing" and st.session_state.cv_result == "":
    with st.spinner("Génération du CV adapté..."):
        res = generate_premium_cv(
            st.session_state.job_offer_text,
            st.session_state.cv_text,
            output_language
        )
        st.session_state.cv_result = res
        st.session_state.cv_status = "done"
        st.rerun()


# =========================================================
# ✍️ LETTRE DE MOTIVATION
# =========================================================

st.markdown("### ✍️ Lettre de motivation")

if st.session_state.lm_status == "done":
    st.success("Lettre générée ✅")
    st.text_area("Lettre de motivation", st.session_state.lm_result, height=400)

    st.download_button(
    "⬇️ Télécharger la lettre",
    st.session_state.lm_result,
    file_name="Lettre_de_motivation.txt",
    mime="text/plain"
    )

elif st.session_state.lm_status == "processing":
    st.button("Génération en cours…", disabled=True)

elif st.session_state.lm_status == "idle":

    st.markdown('<div class="card">', unsafe_allow_html=True)

    if (
        "compatibility_score" in st.session_state
        and st.session_state.compatibility_score is not None
        and st.session_state.compatibility_score < 50
    ):
        st.error(
            "❗ Votre CV présente une compatibilité inférieure à 50% avec cette offre.\n\n"
            "Pour des raisons d’intégrité professionnelle, nous ne pouvons pas modifier "
            "votre candidature lorsque l’écart est trop important.\n\n"
            "Nous ne pouvons ni inventer ni ajouter des compétences absentes de votre profil."
        )

        st.button("Générer la lettre", disabled=True)

    else:

        if st.button("Générer la lettre", key="gen_letter"):
            st.session_state.lm_status = "processing"
            st.session_state.lm_result = ""

    st.markdown('</div>', unsafe_allow_html=True)

if st.session_state.lm_status == "processing" and st.session_state.lm_result == "":
    with st.spinner("Génération de la lettre..."):
        res = generate_premium_lm(
            st.session_state.job_offer_text,
            st.session_state.cv_text,
            output_language
        )
        st.session_state.lm_result = res
        st.session_state.lm_status = "done"
        st.rerun()


# =========================================================
# 📧 MAIL DE CANDIDATURE
# =========================================================

st.markdown("### 📧 Mail de candidature")

if st.session_state.mail_status == "done":
    st.success("Mail généré ✅")
    st.text_area("Mail de candidature", st.session_state.mail_result, height=300)

    st.download_button(
    "⬇️ Télécharger le mail",
    st.session_state.mail_result,
    file_name="Mail_candidature.txt",
    mime="text/plain"
    )

elif st.session_state.mail_status == "processing":
    st.button("Génération en cours…", disabled=True)

elif st.session_state.mail_status == "idle":

    st.markdown('<div class="card">', unsafe_allow_html=True)

    if (
        "compatibility_score" in st.session_state
        and st.session_state.compatibility_score is not None
        and st.session_state.compatibility_score < 50
    ):

        st.button("Générer le mail", disabled=True)

    else:

        if st.button("Générer le mail", key="gen_mail"):
            st.session_state.mail_status = "processing"
            st.session_state.mail_result = ""

    st.markdown('</div>', unsafe_allow_html=True)

if st.session_state.mail_status == "processing" and st.session_state.mail_result == "":
    with st.spinner("Génération du mail..."):
        res = generate_premium_mail(
            st.session_state.job_offer_text,
            st.session_state.cv_text,
            output_language
        )
        st.session_state.mail_result = res
        st.session_state.mail_status = "done"
        st.rerun()

st.markdown("---")

st.markdown(
    "<p style='text-align:center; color:#6b7280; font-size:14px;'>©️ Katshux Group – Tous droits réservés</p>",
    unsafe_allow_html=True
)
# test commit 