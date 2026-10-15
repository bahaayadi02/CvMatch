"""Streamlit UI – CvMatch: Intelligent Resume & Job Matcher.

Run:  streamlit run app.py
"""

import io, json, sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.matcher import match, DEFAULT_WEIGHTS
from src.parser import read_pdf
from src.recommender import generate_recommendations
from src.url_scraper import extract_job_description_from_url
from src.tailor import generate_tailoring_suggestions, apply_tailored_suggestion
from src.pdf_generator import generate_ats_pdf
from src.latex_template import (
    tailor_cv_to_latex,
    compile_latex_to_pdf,
    latex_to_clean_text,
)

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="CvMatch – AI Resume & Job Fit Engine",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════
# MULTI-LANGUAGE TRANSLATIONS DICTIONARY (FR, EN, AR)
# ══════════════════════════════════════════════════════════════════════
TRANSLATIONS = {
    "FR": {
        "dir": "ltr",
        "brand_sub": "MOTEUR D'ALIGNEMENT IA CV & EMPLOI",
        "nav_weights_btn": "Pondération IA",
        "weights_title": "Pondération des critères d'évaluation",
        "weights_desc": "Modifiez l'importance relative de chaque signal pour calibrer l'évaluation.",
        "preset_balanced": "Équilibré (Standard)",
        "preset_skills": "Priorité Compétences",
        "preset_semantic": "Priorité Sémantique",
        "hero_badge": "ANALYSE DE PRÉCISION IA",
        "hero_title": "Analysez l'adéquation exacte<br>entre le CV et l'offre d'emploi.",
        "hero_sub": "Évaluez instantanément la concordance d'un profil avec les exigences du poste grâce à notre triple moteur d'analyse : compréhension sémantique profonde, validation lexicale ATS et cartographie des compétences.",
        "presets_label": "Profils types pour tester en 1 clic :",
        "presets": [
            ("Machine Learning", "ml_engineer.txt", "ml_engineer_jd.txt", ":material/memory:"),
            ("DevOps & Cloud", "devops_engineer.txt", "devops_engineer_jd.txt", ":material/cloud:"),
            ("Backend Developer", "backend_dev.txt", "backend_dev_jd.txt", ":material/terminal:"),
            ("Data Scientist", "data_scientist.txt", "data_scientist_jd.txt", ":material/insights:"),
            ("Full-Stack Dev", "fullstack_dev.txt", "fullstack_dev_jd.txt", ":material/layers:"),
            ("NLP Specialist", "nlp_researcher.txt", "nlp_engineer_jd.txt", ":material/chat:"),
        ],
        "resume_title": "Curriculum Vitae (Candidat)",
        "resume_desc": "Téléversez un CV au format PDF ou collez le texte brut complet.",
        "paste_text": "Coller le texte",
        "upload_pdf": "Importer un PDF",
        "resume_placeholder": "Expériences professionnelles, formations, projets, compétences techniques...",
        "words_detected": "mots détectés",
        "pdf_success": "PDF extrait avec succès : {count} mots analysables.",
        "jd_title": "Offre d'emploi (Entreprise)",
        "jd_desc": "Collez l'annonce du poste ou récupérez-la directement depuis une URL web.",
        "paste_jd": "Coller l'annonce",
        "fetch_url": "Extraire depuis URL",
        "jd_placeholder": "Missions, compétences indispensables, environnement technique, pré-requis...",
        "url_placeholder": "https://... (ex: offre d'emploi en ligne)",
        "btn_fetch": "Récupérer l'offre",
        "url_success": "Offre récupérée ({count} mots)",
        "btn_analyze": "Analyser la correspondance du profil",
        "computing": "Calcul des vecteurs d'embeddings et extraction des compétences...",
        "err_resume": "Veuillez renseigner le contenu du CV (texte ou PDF) avant d'analyser.",
        "err_jd": "Veuillez renseigner la description de l'offre d'emploi.",
        "verdict_strong_title": "Excellente adéquation candidat – poste",
        "verdict_strong_sub": "Le profil présente une très forte concordance avec les exigences techniques et le périmètre du poste.",
        "verdict_strong_pill": "Adéquation Élevée",
        "verdict_moderate_title": "Alignement intermédiaire avec axes d'optimisation",
        "verdict_moderate_sub": "Le candidat possède de solides atouts mais présente quelques lacunes sur des compétences requises.",
        "verdict_moderate_pill": "Adéquation Partielle",
        "verdict_weak_title": "Écart significatif avec le profil requis",
        "verdict_weak_sub": "Le profil s'éloigne des critères indispensables ou s'oriente vers une spécialité différente.",
        "verdict_weak_pill": "Adéquation Faible",
        "metric_score": "Score Global",
        "metric_score_sub": "Score synthétique pondéré",
        "metric_sem": "Compréhension Sémantique",
        "metric_sem_sub": "Alignement conceptuel profond",
        "metric_ats": "Mots-clés ATS",
        "metric_ats_sub": "Filtrage lexical & conformité",
        "metric_skills": "Compétences Clés",
        "metric_skills_sub": "Couverture des entités requises",
        "gauge_title": "Indice d'adéquation",
        "signal_contrib_title": "Contribution relative des signaux",
        "raw_score": "Score Brut",
        "weighted_impact": "Impact Pondéré",
        "signals": ["Sémantique", "Mots-clés ATS", "Compétences"],
        "skills_title": "Cartographie des compétences techniques",
        "skills_matched": "Compétences validées",
        "skills_missing": "Compétences manquantes",
        "skills_bonus": "Atouts additionnels du candidat",
        "no_matched": "Aucune compétence commune relevée.",
        "all_matched": "Toutes les compétences requises sont validées !",
        "no_bonus": "Aucun atout technique supplémentaire relevé.",
        "recs_title": "Recommandations stratégiques pour le profil",
        "badge_critical": "Prioritaire",
        "badge_advisory": "Conseil",
        "badge_positive": "Point Fort",
        "export_title": "Exporter le rapport d'analyse et détails",
        "btn_download": "Télécharger le rapport JSON",
        "sidebar_title": "Pondération IA",
        "sidebar_sub": "Calibrez l'importance des signaux",
        "sem_slider": "Similarité Sémantique",
        "tfidf_slider": "Mots-clés ATS",
        "skill_slider": "Compétences Clés",
        "weights_norm": "Poids normalisés :",
        "arch_title": "Moteur CvMatch :",
        "arch_info": "Triple moteur neuronal • Embeddings haute dimension • Analyse lexicale ATS",
        "tailor_studio_title": "Studio d'Optimisation ATS & Copilote CV",
        "tailor_studio_sub": "Suggestions d'enrichissement chirurgicales guidées par l'IA pour maximiser votre score ATS sans inventer d'expérience.",
        "btn_generate_tailor": "Analyser et générer les optimisations ATS",
        "tailor_gain_title": "Potentiel d'amélioration ATS",
        "tailor_applied": "Optimisation appliquée avec succès !",
        "btn_apply_sug": "Appliquer cette optimisation au CV",
        "sug_applied_badge": "Déjà appliqué",
        "pdf_card_title": "Exportation du CV au format ATS Standard (PDF)",
        "pdf_card_desc": "Format linéaire 1 colonne, typographie machine-readable (Helvetica/Arial), compatible à 100% avec Taleo, Workday, Greenhouse et Lever.",
        "btn_download_pdf": "Télécharger le CV optimisé (PDF ATS)",
        "before_label": "Avant (Original)",
        "after_label": "Après (Optimisé ATS)",
        "back_btn": "← Retour / Modifier les entrées",
        "btn_optimize_now": "✨ Optimiser mon CV pour cette offre",
        "optimizing_spinner": "Analyse sémantique et optimisation chirurgicale ATS en cours...",
        "optimized_preview_title": "CV Optimisé ATS (Format Prêt pour Recruteur)",
        "optimized_preview_sub": "Voici la version enrichie de votre CV avec les compétences et mots-clés de l'offre intégrés naturellement.",
        "btn_download_tailored_pdf": "📥 Télécharger le CV Optimisé (PDF Haute Fidélité)",
        "cv_editor_help": "Vous pouvez ajuster ou copier le texte du CV ci-dessous avant téléchargement.",
    },
    "EN": {
        "dir": "ltr",
        "brand_sub": "AI RESUME & JOB FIT ENGINE",
        "nav_weights_btn": "AI Weights",
        "weights_title": "Evaluation Criteria Weights",
        "weights_desc": "Fine-tune the relative importance of each signal to reflect your hiring priorities.",
        "preset_balanced": "Balanced (Default)",
        "preset_skills": "Skills-First",
        "preset_semantic": "Semantic-First",
        "hero_badge": "PRECISION AI MATCHING",
        "hero_title": "Analyze the exact fit<br>between resume and job criteria.",
        "hero_sub": "Instantly evaluate candidate-to-role fit using our triple neural engine: deep contextual semantic similarity, ATS keyword verification, and technical skill extraction.",
        "presets_label": "Quick 1-Click Sample Profiles:",
        "presets": [
            ("Machine Learning", "ml_engineer.txt", "ml_engineer_jd.txt", ":material/memory:"),
            ("DevOps & Cloud", "devops_engineer.txt", "devops_engineer_jd.txt", ":material/cloud:"),
            ("Backend Developer", "backend_dev.txt", "backend_dev_jd.txt", ":material/terminal:"),
            ("Data Scientist", "data_scientist.txt", "data_scientist_jd.txt", ":material/insights:"),
            ("Full-Stack Dev", "fullstack_dev.txt", "fullstack_dev_jd.txt", ":material/layers:"),
            ("NLP Specialist", "nlp_researcher.txt", "nlp_engineer_jd.txt", ":material/chat:"),
        ],
        "resume_title": "Candidate Resume (CV)",
        "resume_desc": "Upload a PDF resume or paste raw text below.",
        "paste_text": "Paste Text",
        "upload_pdf": "Upload PDF",
        "resume_placeholder": "Paste candidate resume text — experience, education, skills, projects...",
        "words_detected": "words detected",
        "pdf_success": "PDF parsed successfully: {count} words extracted.",
        "jd_title": "Job Specification",
        "jd_desc": "Paste the target job description or scrape it from a live URL.",
        "paste_jd": "Paste JD",
        "fetch_url": "Fetch from URL",
        "jd_placeholder": "Paste job requirements, technical qualifications, responsibilities...",
        "url_placeholder": "https://... (job listing URL)",
        "btn_fetch": "Extract Job Details",
        "url_success": "Job description loaded ({count} words)",
        "btn_analyze": "Run AI Alignment Engine",
        "computing": "Computing vector embeddings and extracting technical skills...",
        "err_resume": "Please provide candidate resume content (text or PDF) before analyzing.",
        "err_jd": "Please provide the target job description.",
        "verdict_strong_title": "Strong candidate–role alignment",
        "verdict_strong_sub": "The candidate's profile strongly satisfies the technical requirements and core responsibilities.",
        "verdict_strong_pill": "High Match",
        "verdict_moderate_title": "Moderate alignment with growth opportunities",
        "verdict_moderate_sub": "The candidate has solid fundamentals but lacks several specific required skills.",
        "verdict_moderate_pill": "Partial Match",
        "verdict_weak_title": "Low alignment with role requirements",
        "verdict_weak_sub": "The profile demonstrates significant gaps in core prerequisites or targets a different domain.",
        "verdict_weak_pill": "Low Match",
        "metric_score": "Match Score",
        "metric_score_sub": "Composite weighted score",
        "metric_sem": "Semantic Similarity",
        "metric_sem_sub": "Deep contextual vector cosine",
        "metric_ats": "ATS Keyword Overlap",
        "metric_ats_sub": "N-gram lexical compliance",
        "metric_skills": "Skill Coverage",
        "metric_skills_sub": "Percentage of required skills found",
        "gauge_title": "Alignment Index",
        "signal_contrib_title": "Signal Impact Breakdown",
        "raw_score": "Raw Score",
        "weighted_impact": "Weighted Impact",
        "signals": ["Semantic", "ATS Keywords", "Skills"],
        "skills_title": "Technical Skill Entity Breakdown",
        "skills_matched": "Matched Skills",
        "skills_missing": "Missing Requirements",
        "skills_bonus": "Candidate Differentiators",
        "no_matched": "No overlapping skills detected.",
        "all_matched": "All required skills are present in the resume!",
        "no_bonus": "No additional skills identified.",
        "recs_title": "Strategic Optimization Recommendations",
        "badge_critical": "Critical",
        "badge_advisory": "Advisory",
        "badge_positive": "Strength",
        "export_title": "Export Full Match Report & Technical Diagnostics",
        "btn_download": "Download JSON Report",
        "sidebar_title": "AI Weights Tuning",
        "sidebar_sub": "Calibrate signal priorities",
        "sem_slider": "Semantic Weight",
        "tfidf_slider": "ATS Keywords Weight",
        "skill_slider": "Skill Entity Weight",
        "weights_norm": "Normalized Weights:",
        "arch_title": "CvMatch Engine:",
        "arch_info": "Tri-Signal Pipeline • Dense Vector Embeddings • Multi-Gram ATS Lexical Analysis",
        "tailor_studio_title": "ATS Optimization Studio & CV Copilot",
        "tailor_studio_sub": "Surgical, AI-driven bullet enhancements to maximize your ATS match score with authentic experience.",
        "btn_generate_tailor": "Analyze & Generate ATS Optimizations",
        "tailor_gain_title": "Score Improvement Potential",
        "tailor_applied": "Optimization applied successfully!",
        "btn_apply_sug": "Apply this Optimization to CV",
        "sug_applied_badge": "Already Applied",
        "pdf_card_title": "Export Resume in Standard ATS Format (PDF)",
        "pdf_card_desc": "Single-column linear layout, machine-readable typography (Helvetica/Arial), 100% compliant with Taleo, Workday, Greenhouse, and Lever.",
        "btn_download_pdf": "Download Tailored Resume (ATS PDF)",
        "before_label": "Before (Original)",
        "after_label": "After (ATS-Optimized)",
        "back_btn": "← Back / Modify inputs",
        "btn_optimize_now": "✨ Tailor CV for this Job",
        "optimizing_spinner": "Semantic gap analysis and ATS optimization in progress...",
        "optimized_preview_title": "ATS-Optimized Resume Preview (Recruiter-Ready)",
        "optimized_preview_sub": "Here is your tailored resume with target JD keywords authentically incorporated.",
        "btn_download_tailored_pdf": "📥 Download Tailored Resume (High-Fidelity PDF)",
        "cv_editor_help": "You can edit or copy your resume text below before downloading.",
    },
    "AR": {
        "dir": "rtl",
        "brand_sub": "محرك الذكاء الاصطناعي لمطابقة السير الذاتية والوظائف",
        "nav_weights_btn": "أوزان التقييم",
        "weights_title": "معايرة أوزان التقييم الذكي",
        "weights_desc": "حدد الأهمية النسبية لكل معيار لملاءمة أسلوب التقييم مع متطلباتك.",
        "preset_balanced": "متوازن (افتراضي)",
        "preset_skills": "أولوية المهارات",
        "preset_semantic": "أولوية الدلالة",
        "hero_badge": "دقة مطابقة بالذكاء الاصطناعي",
        "hero_title": "حلل بدقة مدى توافق<br>السيرة الذاتية مع الوظيفة المطلوبة.",
        "hero_sub": "طابق السيرة الذاتية مع متطلبات الوظيفة فورياً عبر المحرك العصبي الثلاثي: الفهم الدلالي العميق، فحص الكلمات المفتاحية ATS، واستخراج المهارات التقنية الأساسية.",
        "presets_label": "نماذج جاهزة للاختبار بنقرة واحدة:",
        "presets": [
            ("تعلم آلي (ML)", "ml_engineer.txt", "ml_engineer_jd.txt", ":material/memory:"),
            ("DevOps سحابي", "devops_engineer.txt", "devops_engineer_jd.txt", ":material/cloud:"),
            ("مطور Backend", "backend_dev.txt", "backend_dev_jd.txt", ":material/terminal:"),
            ("عالم بيانات", "data_scientist.txt", "data_scientist_jd.txt", ":material/insights:"),
            ("مطور Full-Stack", "fullstack_dev.txt", "fullstack_dev_jd.txt", ":material/layers:"),
            ("خبير NLP", "nlp_researcher.txt", "nlp_engineer_jd.txt", ":material/chat:"),
        ],
        "resume_title": "السيرة الذاتية (المترشح)",
        "resume_desc": "قم برفع ملف PDF أو الصق نص السيرة الذاتية أدناه.",
        "paste_text": "لصق النص",
        "upload_pdf": "رفع PDF",
        "resume_placeholder": "الصق نص السيرة الذاتية — الخبرات، المشاريع، المهارات التقنية، التعليم...",
        "words_detected": "كلمة مكتشفة",
        "pdf_success": "تم استخراج ملف PDF بنجاح: {count} كلمة قابلة للتحليل.",
        "jd_title": "الوصف الوظيفي (الشركة)",
        "jd_desc": "الصق متطلبات الوظيفة أو استخرجها مباشرة من رابط إلكتروني.",
        "paste_jd": "لصق الإعلان",
        "fetch_url": "استخراج من رابط",
        "jd_placeholder": "المسؤوليات، المهارات الإلزامية، البيئة التقنية، المؤهلات المطلوبة...",
        "url_placeholder": "https://... (رابط عرض العمل)",
        "btn_fetch": "جلب تفاصيل الوظيفة",
        "url_success": "تم جلب تفاصيل الوظيفة ({count} كلمة)",
        "btn_analyze": "تحليل مدى التوافق والمهارات",
        "computing": "جاري حساب المتجهات واستخراج الكيانات والمهارات...",
        "err_resume": "يرجى إدخال محتوى السيرة الذاتية (نص أو ملف PDF) قبل التحليل.",
        "err_jd": "يرجى إدخال متطلبات الوصف الوظيفي.",
        "verdict_strong_title": "توافق ممتاز بين المترشح والوظيفة",
        "verdict_strong_sub": "السيرة الذاتية تلبي بامتياز المتطلبات والمهارات التقنية المطلوبة للدور.",
        "verdict_strong_pill": "توافق عالٍ",
        "verdict_moderate_title": "توافق متوسط مع فرص للتطوير",
        "verdict_moderate_sub": "يمتلك المترشح أساسيات قوية ولكن تنقصه بعض المهارات التقنية المحددة.",
        "verdict_moderate_pill": "توافق جزئي",
        "verdict_weak_title": "فجوة ملحوظة مع متطلبات الوظيفة",
        "verdict_weak_sub": "المترشح يفتقر للعديد من الشروط الأساسية أو يختص بمجال مغاير.",
        "verdict_weak_pill": "توافق منخفض",
        "metric_score": "نسبة التوافق الإجمالية",
        "metric_score_sub": "المعدل التراكمي الموزون",
        "metric_sem": "الفهم الدلالي",
        "metric_sem_sub": "التطابق المفاهيمي العميق",
        "metric_ats": "كلمات ATS المفتاحية",
        "metric_ats_sub": "التطابق المعجمي واللفظي",
        "metric_skills": "تغطية المهارات",
        "metric_skills_sub": "نسبة المهارات المطلوبة المتوفرة",
        "gauge_title": "مؤشر التوافق الوظيفي",
        "signal_contrib_title": "مساهمة المحاور في التقييم",
        "raw_score": "النتيجة الأولية",
        "weighted_impact": "الأثر الموزون",
        "signals": ["الدلالي", "كلمات ATS", "المهارات"],
        "skills_title": "خريطة المهارات والكيانات التقنية",
        "skills_matched": "المهارات المتطابقة",
        "skills_missing": "المهارات الناقصة المطلوبة",
        "skills_bonus": "مهارات إضافية يتميز بها المترشح",
        "no_matched": "لم يتم رصد مهارات مشتركة.",
        "all_matched": "جميع المهارات المطلوبة متوفرة بالكامل!",
        "no_bonus": "لا توجد مهارات إضافية مرصودة.",
        "recs_title": "توصيات استراتيجية لتحسين السيرة الذاتية",
        "badge_critical": "أولوية قصوى",
        "badge_advisory": "نصيحة",
        "badge_positive": "نقطة قوة",
        "export_title": "تحميل التقرير الكامل وتفاصيل التحليل",
        "btn_download": "تحميل تقرير JSON",
        "sidebar_title": "معايرة أوزان الذكاء الاصطناعي",
        "sidebar_sub": "حدد أهمية كل معيار في التقييم",
        "sem_slider": "وزن التطابق الدلالي",
        "tfidf_slider": "وزن كلمات ATS",
        "skill_slider": "وزن المهارات",
        "weights_norm": "الأوزان بعد التعديل:",
        "arch_title": "محرك CvMatch :",
        "arch_info": "محرك ثلاثي متطور • تحويل نصي عالي الأبعاد • كشف تلقائي للكيانات والمهارات",
        "tailor_studio_title": "استوديو التحسين المتوافق مع ATS ومساعد السيرة الذاتية",
        "tailor_studio_sub": "اقتراحات ذكية ودقيقة لإعادة صياغة العبارات بما يرفع نسبة التوافق مع أنظمة ATS دون اختلاق أي معلومات.",
        "btn_generate_tailor": "تحليل وتوليد اقتراحات تحسين ATS",
        "tailor_gain_title": "الزيادة المتوقعة في نسبة التوافق",
        "tailor_applied": "تم تطبيق التعديل بنجاح!",
        "btn_apply_sug": "تطبيق هذا التعديل على السيرة الذاتية",
        "sug_applied_badge": "تم التطبيق بالفعل",
        "pdf_card_title": "تصدير السيرة الذاتية بصيغة ATS القياسية (PDF)",
        "pdf_card_desc": "تصميم خطي بعمود واحد، خطوط قياسية سهلة القراءة آلياً، متوافق 100% مع أنظمة Taleo وWorkday وGreenhouse.",
        "btn_download_pdf": "تحميل السيرة الذاتية المحسنة (PDF ATS)",
        "before_label": "قبل (النص الأصلي)",
        "after_label": "بعد (الصياغة المحسنة)",
        "back_btn": "← العودة / تعديل المدخلات",
        "btn_optimize_now": "✨ تحسين السيرة الذاتية لهذا العرض",
        "optimizing_spinner": "جاري التحليل الدلالي وتوليد التحسينات المتوافقة مع ATS...",
        "optimized_preview_title": "السيرة الذاتية المحسنة المتوافقة مع ATS (جاهزة للتقديم)",
        "optimized_preview_sub": "هذه هي النسخة المحسنة من سيرتك الذاتية مع دمج الكلمات المفتاحية المطلوبة بدقة.",
        "btn_download_tailored_pdf": "📥 تحميل السيرة الذاتية المحسنة (PDF عالي الدقة)",
        "cv_editor_help": "يمكنك مراجعة أو نسخ نص السيرة الذاتية أدناه مباشرة قبل التحميل.",
    }
}

# ── Session state initialization ─────────────────────────────────────
for key, default in [
    ("lang", "FR"),
    ("resume_input", ""),
    ("jd_input", ""),
    ("result", None),
    ("recs", []),
    ("w_sem", DEFAULT_WEIGHTS["semantic"]),
    ("w_tfidf", DEFAULT_WEIGHTS["tfidf"]),
    ("w_skill", DEFAULT_WEIGHTS["skill"]),
    ("applied_sugs", set()),
    ("current_page", "input"),
    ("tailored_cv_text", None),
    ("tailor_boost_score", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

current_lang = st.session_state.lang
T = TRANSLATIONS[current_lang]
is_rtl = T["dir"] == "rtl"

# Calculate active normalized weights
tot_w = st.session_state.w_sem + st.session_state.w_tfidf + st.session_state.w_skill
active_weights = (
    {
        "semantic": round(st.session_state.w_sem / tot_w, 3),
        "tfidf": round(st.session_state.w_tfidf / tot_w, 3),
        "skill": round(st.session_state.w_skill / tot_w, 3),
    }
    if tot_w > 0
    else DEFAULT_WEIGHTS
)

# ══════════════════════════════════════════════════════════════════════
# CSS DESIGN SYSTEM – CLEAN MODERN HELLOWORK STYLE + 3D TOUCHES
# ══════════════════════════════════════════════════════════════════════
rtl_css = """
body, .stApp, .block-container, p, span, div, h1, h2, h3, h4, h5, h6, input, textarea {
    direction: rtl !important;
    text-align: right !important;
}
.hw-logo-wrap, .hw-navbar, .hw-card-header, .hw-verdict-box {
    flex-direction: row-reverse !important;
}
.hw-tag {
    margin: 3px 0 5px 6px !important;
}
""" if is_rtl else ""

st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800;900&family=Cairo:wght@400;600;700;800;900&family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&display=swap');

/* ── STRICTLY HIDE STREAMLIT CHROME, DEPLOY BUTTON, RUNNING MAN, AND STOP BUTTON ── */
.stDeployButton,
div[data-testid="stToolbar"],
div[data-testid="stStatusWidget"],
div[data-testid="stDecoration"],
button[data-testid="stStopButton"],
header[data-testid="stHeader"],
#MainMenu,
footer {{
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    pointer-events: none !important;
    height: 0 !important;
    width: 0 !important;
}}

/* ── Prevent broken Material Symbols font overriding ── */
.material-symbols-rounded,
.material-symbols-outlined,
[data-testid="stIconMaterial"],
button[data-testid="collapsedControl"] span,
[data-testid="stSidebarCollapseButton"] span,
[data-testid="stExpander"] summary span:first-child,
[data-testid="stFileUploader"] span,
button span[data-testid="stIconMaterial"] {{
    font-family: 'Material Symbols Rounded', 'Material Symbols Outlined' !important;
    display: inline-block !important;
}}

/* ── Clean floating Sidebar toggle ── */
button[data-testid="collapsedControl"] {{
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    background: #ffffff !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08) !important;
    top: 1.2rem !important;
    left: 1.2rem !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
}}
button[data-testid="collapsedControl"]:hover {{
    background: #f8fafc !important;
    border-color: #0f172a !important;
    transform: scale(1.05) !important;
}}

/* ── Typography & Direction ── */
html, body, p, li, h1, h2, h3, h4, h5, h6, input, textarea, select, label {{
    font-family: {'"Cairo", ' if is_rtl else ''}'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif !important;
}}

{rtl_css}

/* ── 3D & Entrance Animations ── */
@keyframes float3D {{
    0%, 100% {{ transform: translateY(0px) rotateX(4deg) rotateY(-6deg); }}
    50%      {{ transform: translateY(-8px) rotateX(-2deg) rotateY(6deg); }}
}}
@keyframes fadeInScale {{
    from {{ opacity: 0; transform: translateY(12px) scale(0.98); }}
    to   {{ opacity: 1; transform: translateY(0) scale(1); }}
}}

/* ── Base App Container ── */
.stApp {{
    background-color: #f8fafc;
    color: #0f172a;
}}
.block-container {{
    padding-top: 1rem !important;
    padding-bottom: 3.5rem !important;
    max-width: 1240px !important;
}}

/* ── Top Navbar ── */
.hw-navbar {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.8rem 0;
    border-bottom: 1px solid #e2e8f0;
    margin-bottom: 1.5rem;
}}
.hw-logo-wrap {{
    display: flex;
    align-items: baseline;
    gap: 3px;
}}
.hw-logo-main {{
    font-size: 1.75rem;
    font-weight: 900;
    letter-spacing: -0.04em;
    color: #0f172a;
    line-height: 1;
}}
.hw-logo-accent {{
    font-size: 1.75rem;
    font-weight: 900;
    letter-spacing: -0.04em;
    color: #2563eb;
    line-height: 1;
}}
.hw-logo-badge {{
    font-size: 0.68rem;
    font-weight: 800;
    color: #2563eb;
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    padding: 3px 9px;
    border-radius: 9999px;
    margin-left: 8px;
    letter-spacing: 0.04em;
}}

/* ── Hero Banner with Modern 3D Hologram Orb ── */
.hw-hero {{
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border-radius: 24px;
    padding: 2.8rem 3rem;
    color: #ffffff;
    margin-bottom: 2rem;
    box-shadow: 0 15px 35px -8px rgba(15, 23, 42, 0.2);
    position: relative;
    overflow: hidden;
    perspective: 1000px;
    animation: fadeInScale 0.5s ease both;
}}
.hw-hero::after {{
    content: "";
    position: absolute;
    top: -50%; right: -15%;
    width: 480px; height: 480px;
    border-radius: 9999px;
    background: radial-gradient(circle, rgba(37, 99, 235, 0.25) 0%, rgba(124, 58, 237, 0.12) 50%, transparent 70%);
    pointer-events: none;
}}
.hw-3d-orb {{
    position: absolute;
    right: 3.5rem;
    top: 2.3rem;
    width: 92px;
    height: 92px;
    border-radius: 24px;
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.8) 100%);
    border: 1.5px solid rgba(255,255,255,0.25);
    box-shadow: 0 12px 30px rgba(0,0,0,0.35), inset 0 0 20px rgba(96, 165, 250, 0.2);
    display: flex;
    align-items: center;
    justify-content: center;
    backdrop-filter: blur(12px);
    animation: float3D 6s ease-in-out infinite;
}}
@media (max-width: 900px) {{
    .hw-3d-orb {{ display: none; }}
}}
.hw-hero-tag {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(255, 255, 255, 0.12);
    border: 1px solid rgba(255, 255, 255, 0.22);
    border-radius: 9999px;
    padding: 4px 14px;
    font-size: 0.74rem;
    font-weight: 800;
    letter-spacing: 0.05em;
    color: #e2e8f0;
    margin-bottom: 1.1rem;
    backdrop-filter: blur(8px);
}}
.hw-hero-title {{
    font-size: 2.3rem;
    font-weight: 900;
    line-height: 1.2;
    letter-spacing: -0.035em;
    color: #ffffff;
    margin: 0 0 0.8rem 0;
}}
.hw-hero-subtitle {{
    font-size: 1.02rem;
    color: #cbd5e1;
    line-height: 1.6;
    max-width: 720px;
    margin: 0;
}}

/* ── Modern Cards with 3D Hover Lift ── */
.hw-card {{
    background: #ffffff;
    border: 1.5px solid #e2e8f0;
    border-radius: 20px;
    padding: 1.6rem 1.8rem;
    box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.04);
    transition: transform 0.28s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.28s ease, border-color 0.28s ease;
    height: 100%;
}}
.hw-card:hover {{
    transform: translateY(-4px);
    box-shadow: 0 12px 30px -4px rgba(15, 23, 42, 0.09);
    border-color: #cbd5e1;
}}
.hw-card-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.5rem;
}}
.hw-card-title {{
    font-size: 1.05rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: #0f172a;
    display: flex;
    align-items: center;
    gap: 8px;
}}
.hw-card-desc {{
    font-size: 0.82rem;
    color: #64748b;
    margin-bottom: 1rem;
}}

/* ── Inputs ── */
.stTextArea textarea {{
    background-color: #ffffff !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 14px !important;
    color: #0f172a !important;
    font-size: 0.88rem !important;
    line-height: 1.65 !important;
    padding: 0.9rem !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.02) !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}}
.stTextArea textarea:focus {{
    border-color: #0f172a !important;
    box-shadow: 0 0 0 3px rgba(15, 23, 42, 0.08) !important;
}}
.stTextInput input {{
    background-color: #ffffff !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 12px !important;
    color: #0f172a !important;
    font-size: 0.88rem !important;
    padding: 0.65rem 0.9rem !important;
}}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    gap: 8px !important;
    background-color: transparent !important;
    border-bottom: 2px solid #e2e8f0 !important;
    padding-bottom: 2px !important;
}}
.stTabs [data-baseweb="tab"] {{
    color: #64748b !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 0.55rem 1rem !important;
    border-radius: 8px 8px 0 0 !important;
    background: transparent !important;
    border: none !important;
    transition: all 0.2s ease !important;
}}
.stTabs [data-baseweb="tab"]:hover {{
    color: #0f172a !important;
    background: #f1f5f9 !important;
}}
.stTabs [aria-selected="true"] {{
    color: #0f172a !important;
    font-weight: 800 !important;
    border-bottom: 2.5px solid #0f172a !important;
}}

/* ── Primary Action Button ── */
div.stButton > button[kind="primary"] {{
    background-color: #0f172a !important;
    color: #ffffff !important;
    font-weight: 800 !important;
    font-size: 1rem !important;
    padding: 0.85rem 2.4rem !important;
    border-radius: 9999px !important;
    border: none !important;
    box-shadow: 0 6px 20px rgba(15, 23, 42, 0.22) !important;
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
    cursor: pointer !important;
}}
div.stButton > button[kind="primary"]:hover {{
    background-color: #000000 !important;
    transform: translateY(-3px) scale(1.01) !important;
    box-shadow: 0 12px 28px rgba(15, 23, 42, 0.3) !important;
}}

/* ── Preset Buttons ── */
div.stButton > button:not([kind="primary"]) {{
    background-color: #ffffff !important;
    color: #0f172a !important;
    font-weight: 700 !important;
    font-size: 0.86rem !important;
    padding: 0.65rem 1.1rem !important;
    border-radius: 9999px !important;
    border: 1.5px solid #e2e8f0 !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.03) !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
}}
div.stButton > button:not([kind="primary"]):hover {{
    border-color: #0f172a !important;
    background-color: #f8fafc !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 14px rgba(15, 23, 42, 0.08) !important;
}}

/* ── Verdict Card ── */
.hw-verdict-box {{
    background: #ffffff;
    border-radius: 20px;
    padding: 1.6rem 2.2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 1.2rem;
    border-left: 6px solid #0f172a;
    box-shadow: 0 6px 24px rgba(15, 23, 42, 0.06);
    margin-bottom: 1.6rem;
    animation: fadeInScale 0.4s ease both;
}}
.hw-verdict-box.strong {{
    border-left-color: #10b981;
    background: linear-gradient(90deg, #ecfdf5 0%, #ffffff 100%);
}}
.hw-verdict-box.moderate {{
    border-left-color: #f59e0b;
    background: linear-gradient(90deg, #fffbeb 0%, #ffffff 100%);
}}
.hw-verdict-box.weak {{
    border-left-color: #ef4444;
    background: linear-gradient(90deg, #fef2f2 0%, #ffffff 100%);
}}
.hw-verdict-text {{
    font-size: 1.15rem;
    font-weight: 800;
    color: #0f172a;
}}
.hw-verdict-sub {{
    font-size: 0.86rem;
    color: #475569;
    margin-top: 4px;
}}
.hw-verdict-pill {{
    padding: 6px 18px;
    border-radius: 9999px;
    font-weight: 800;
    font-size: 0.85rem;
    letter-spacing: 0.02em;
    text-transform: uppercase;
}}
.pill-strong {{ background: #d1fae5; color: #065f46; border: 1px solid #a7f3d0; }}
.pill-moderate {{ background: #fef3c7; color: #92400e; border: 1px solid #fde68a; }}
.pill-weak {{ background: #fee2e2; color: #991b1b; border: 1px solid #fecaca; }}

/* ── 4 Metric Highlights ── */
.hw-metrics-row {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 1.8rem;
    animation: fadeInScale 0.45s ease both;
}}
@media (max-width: 900px) {{
    .hw-metrics-row {{ grid-template-columns: repeat(2, 1fr); }}
}}
.hw-metric-tile {{
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 1.3rem 1.4rem;
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.03);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}}
.hw-metric-tile:hover {{
    transform: translateY(-3px);
    box-shadow: 0 10px 24px rgba(15, 23, 42, 0.08);
    border-color: #cbd5e1;
}}
.hw-tile-label {{
    font-size: 0.74rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #64748b;
    margin-bottom: 0.4rem;
}}
.hw-tile-value {{
    font-size: 2.2rem;
    font-weight: 900;
    color: #0f172a;
    letter-spacing: -0.03em;
    line-height: 1.1;
}}
.hw-tile-foot {{
    font-size: 0.74rem;
    color: #94a3b8;
    margin-top: 0.35rem;
    font-weight: 500;
}}

/* ── HelloWork Tags ── */
.hw-tag {{
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 5px 13px;
    border-radius: 9999px;
    font-size: 0.8rem;
    font-weight: 600;
    margin: 3px 6px 5px 0;
    transition: transform 0.15s ease;
}}
.hw-tag:hover {{ transform: translateY(-1px); }}
.hw-tag-matched {{ background: #ecfdf5; color: #065f46; border: 1px solid #a7f3d0; }}
.hw-tag-missing {{ background: #fef2f2; color: #991b1b; border: 1px solid #fecaca; }}
.hw-tag-bonus {{ background: #f0fdf4; color: #166534; border: 1px solid #bbf7d0; }}

/* ── Recommendation Cards ── */
.hw-rec-card {{
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 1.1rem 1.4rem;
    display: flex;
    align-items: flex-start;
    gap: 14px;
    box-shadow: 0 2px 10px rgba(15, 23, 42, 0.02);
    margin-bottom: 10px;
    transition: all 0.2s ease;
}}
.hw-rec-card:hover {{
    border-color: #cbd5e1;
    box-shadow: 0 6px 20px rgba(15, 23, 42, 0.05);
}}
.hw-rec-badge {{
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 0.68rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    white-space: nowrap;
    margin-top: 2px;
}}
.hw-badge-critical {{ background: #fee2e2; color: #991b1b; }}
.hw-badge-advisory {{ background: #fef3c7; color: #92400e; }}
.hw-badge-positive {{ background: #d1fae5; color: #065f46; }}
.hw-rec-text {{
    font-size: 0.88rem;
    color: #1e293b;
    line-height: 1.5;
}}

/* ── Popover styling ── */
div[data-testid="stPopoverBody"] {{
    padding: 1.2rem !important;
    border-radius: 18px !important;
    border: 1.5px solid #e2e8f0 !important;
    box-shadow: 0 15px 35px rgba(15, 23, 42, 0.12) !important;
}}
</style>""", unsafe_allow_html=True)


# ── Sample Data Loader ───────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent / "data"

def _load_sample(resume_fname, jd_fname):
    rp = DATA_DIR / "resumes" / resume_fname
    jp = DATA_DIR / "job_descriptions" / jd_fname
    r_text = rp.read_text(encoding="utf-8") if rp.exists() else ""
    j_text = jp.read_text(encoding="utf-8") if jp.exists() else ""
    return r_text, j_text

def _pdf_to_text(f):
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(f.read()))
    pages_text = []
    for page in reader.pages:
        txt = page.extract_text(extraction_mode="layout") or page.extract_text() or ""
        pages_text.append(txt)
    raw = "\n".join(pages_text)
    # Fix glued words produced by bad PDF OCR
    raw = re.sub(r"([a-zà-ÿ])([A-ZÀ-Ý])", r"\1 \2", raw)
    raw = re.sub(r"([a-zA-ZÀ-ÿ]),([a-zA-ZÀ-ÿ])", r"\1, \2", raw)
    raw = re.sub(r"([a-zA-ZÀ-ÿ]):([a-zA-ZÀ-ÿ])", r"\1: \2", raw)
    return raw


# ══════════════════════════════════════════════════════════════════════
# TOP NAVBAR: CvMatch BRAND + AI WEIGHTS POPOVER + LANGUAGE SWITCHER
# ══════════════════════════════════════════════════════════════════════
nav_brand, nav_weights, nav_lang = st.columns([2.3, 1.2, 0.9])

with nav_brand:
    st.markdown(f"""
    <div class="hw-logo-wrap" style="padding-top: 6px;">
        <span class="hw-logo-main">Cv</span><span class="hw-logo-accent">Match</span>
        <span class="hw-logo-badge">{T['brand_sub']}</span>
    </div>
    """, unsafe_allow_html=True)

with nav_weights:
    with st.popover(T["nav_weights_btn"], icon=":material/tune:", use_container_width=True):
        st.markdown(f"""
        <div style="font-weight: 800; font-size: 1rem; color: #0f172a; margin-bottom: 2px;">
            {T['weights_title']}
        </div>
        <div style="font-size: 0.78rem; color: #64748b; margin-bottom: 1rem;">
            {T['weights_desc']}
        </div>
        """, unsafe_allow_html=True)

        w_sem_val = st.slider(T["sem_slider"], 0.0, 1.0, float(st.session_state.w_sem), 0.05, key="popover_w_sem")
        w_tfidf_val = st.slider(T["tfidf_slider"], 0.0, 1.0, float(st.session_state.w_tfidf), 0.05, key="popover_w_tfidf")
        w_skill_val = st.slider(T["skill_slider"], 0.0, 1.0, float(st.session_state.w_skill), 0.05, key="popover_w_skill")

        st.session_state.w_sem = w_sem_val
        st.session_state.w_tfidf = w_tfidf_val
        st.session_state.w_skill = w_skill_val

        tot = w_sem_val + w_tfidf_val + w_skill_val
        norm_sem = round(w_sem_val / tot * 100) if tot > 0 else 50
        norm_tfidf = round(w_tfidf_val / tot * 100) if tot > 0 else 20
        norm_skill = round(w_skill_val / tot * 100) if tot > 0 else 30

        st.markdown(f"""
        <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 10px 12px; margin-top: 10px; font-size: 0.78rem; color: #334155;">
            <b>{T['weights_norm']}</b><br>
            • {T['signals'][0]} : <b>{norm_sem}%</b> |
            • {T['signals'][1]} : <b>{norm_tfidf}%</b> |
            • {T['signals'][2]} : <b>{norm_skill}%</b>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div style="height: 0.5rem"></div>', unsafe_allow_html=True)
        qb1, qb2, qb3 = st.columns(3)
        with qb1:
            if st.button(T["preset_balanced"], key="q_bal", use_container_width=True):
                st.session_state.w_sem, st.session_state.w_tfidf, st.session_state.w_skill = 0.5, 0.2, 0.3
                st.rerun()
        with qb2:
            if st.button(T["preset_skills"], key="q_ski", use_container_width=True):
                st.session_state.w_sem, st.session_state.w_tfidf, st.session_state.w_skill = 0.3, 0.2, 0.5
                st.rerun()
        with qb3:
            if st.button(T["preset_semantic"], key="q_sem", use_container_width=True):
                st.session_state.w_sem, st.session_state.w_tfidf, st.session_state.w_skill = 0.6, 0.2, 0.2
                st.rerun()

with nav_lang:
    lang_labels = {"FR": "🇫🇷 FR", "EN": "🇬🇧 EN", "AR": "🇸🇦 AR"}
    chosen_lang = st.selectbox(
        "Langue",
        options=["FR", "EN", "AR"],
        format_func=lambda x: lang_labels[x],
        index=["FR", "EN", "AR"].index(st.session_state.lang),
        label_visibility="collapsed",
    )
    if chosen_lang != st.session_state.lang:
        st.session_state.lang = chosen_lang
        st.rerun()

st.markdown('<hr style="border:none; border-top: 1px solid #e2e8f0; margin: 0.4rem 0 1.5rem 0;">', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# PAGE 1: CONFIGURATION & ANALYSE
# ══════════════════════════════════════════════════════════════════════
if st.session_state.current_page == "input":
    # ══════════════════════════════════════════════════════════════════════
    st.markdown(f"""
    <div class="hw-hero">
        <div class="hw-3d-orb">
            <svg width="50" height="50" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
                <circle cx="24" cy="24" r="21" stroke="url(#orb_g1)" stroke-width="2" stroke-dasharray="3 3"/>
                <circle cx="24" cy="24" r="14" fill="url(#orb_g2)" stroke="rgba(255,255,255,0.4)" stroke-width="1.5"/>
                <circle cx="24" cy="24" r="6" fill="#ffffff" filter="drop-shadow(0 0 8px #60a5fa)"/>
                <circle cx="24" cy="24" r="2.5" fill="#2563eb"/>
                <line x1="24" y1="3" x2="24" y2="9" stroke="#93c5fd" stroke-width="2" stroke-linecap="round"/>
                <line x1="24" y1="39" x2="24" y2="45" stroke="#93c5fd" stroke-width="2" stroke-linecap="round"/>
                <line x1="3" y1="24" x2="9" y2="24" stroke="#93c5fd" stroke-width="2" stroke-linecap="round"/>
                <line x1="39" y1="24" x2="45" y2="24" stroke="#93c5fd" stroke-width="2" stroke-linecap="round"/>
                <defs>
                    <linearGradient id="orb_g1" x1="3" y1="3" x2="45" y2="45" gradientUnits="userSpaceOnUse">
                        <stop stop-color="#60a5fa"/>
                        <stop offset="1" stop-color="#c084fc"/>
                    </linearGradient>
                    <radialGradient id="orb_g2" cx="0" cy="0" r="1" gradientUnits="userSpaceOnUse" gradientTransform="translate(24 24) rotate(90) scale(14)">
                        <stop stop-color="#3b82f6" stop-opacity="0.9"/>
                        <stop offset="1" stop-color="#1e1b4b" stop-opacity="0.95"/>
                    </radialGradient>
                </defs>
            </svg>
        </div>
        <div class="hw-hero-tag">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#60a5fa" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-1px; margin-right:5px;"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
            <span>{T['hero_badge']}</span>
        </div>
        <h1 class="hw-hero-title">{T['hero_title']}</h1>
        <p class="hw-hero-subtitle">{T['hero_sub']}</p>
    </div>
    """, unsafe_allow_html=True)


    # ══════════════════════════════════════════════════════════════════════
    # QUICK PRESET PILLS (2 ROWS OF 3 COLUMNS TO PREVENT TRUNCATION)
    # ══════════════════════════════════════════════════════════════════════
    st.markdown(f"""
    <div style="font-size: 0.88rem; font-weight: 800; color: #334155; margin-bottom: 0.7rem; display: flex; align-items: center; gap: 6px;">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#2563eb" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg>
        <span>{T['presets_label']}</span>
    </div>
    """, unsafe_allow_html=True)

    presets_list = T["presets"]
    row1_cols = st.columns(3)
    for col, (label, r_file, j_file, mat_icon) in zip(row1_cols, presets_list[:3]):
        with col:
            if st.button(label, icon=mat_icon, key=f"btn_{r_file}", use_container_width=True):
                r_content, j_content = _load_sample(r_file, j_file)
                st.session_state.resume_input = r_content
                st.session_state.jd_input = j_content
                st.session_state.result = None
                st.session_state.recs = []
                st.rerun()

    st.markdown('<div style="height: 0.35rem"></div>', unsafe_allow_html=True)

    row2_cols = st.columns(3)
    for col, (label, r_file, j_file, mat_icon) in zip(row2_cols, presets_list[3:]):
        with col:
            if st.button(label, icon=mat_icon, key=f"btn_{r_file}", use_container_width=True):
                r_content, j_content = _load_sample(r_file, j_file)
                st.session_state.resume_input = r_content
                st.session_state.jd_input = j_content
                st.session_state.result = None
                st.session_state.recs = []
                st.rerun()

    st.markdown('<div style="height: 1.2rem"></div>', unsafe_allow_html=True)


    # ══════════════════════════════════════════════════════════════════════
    # DUAL INPUT CARDS (RESUME VS JOB SPEC) WITH MODERN VECTOR ICONS
    # ══════════════════════════════════════════════════════════════════════
    left_col, right_col = st.columns(2, gap="medium")

    # ── Left: Candidate Resume ───────────────────────────────────────────
    with left_col:
        st.markdown(f"""
        <div class="hw-card-header">
            <div class="hw-card-title">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#2563eb" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-3px; margin-right:8px;"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>
                <span>{T['resume_title']}</span>
            </div>
        </div>
        <div class="hw-card-desc">{T['resume_desc']}</div>
        """, unsafe_allow_html=True)

        tab_r_text, tab_r_pdf = st.tabs([T["paste_text"], T["upload_pdf"]])

        with tab_r_text:
            st.session_state.resume_input = st.text_area(
                "r_text_area",
                value=st.session_state.resume_input,
                height=250,
                placeholder=T["resume_placeholder"],
                label_visibility="collapsed",
            )
            r_word_count = len(st.session_state.resume_input.split()) if st.session_state.resume_input else 0
            st.markdown(
                f'<div style="text-align:right; font-size:0.75rem; color:#64748b; margin-top:4px;"><b>{r_word_count}</b> {T["words_detected"]}</div>',
                unsafe_allow_html=True,
            )

        with tab_r_pdf:
            uploaded_pdf = st.file_uploader(
                "PDF",
                type=["pdf"],
                label_visibility="collapsed",
            )
            if uploaded_pdf is not None:
                extracted_text = _pdf_to_text(uploaded_pdf)
                st.session_state.resume_input = extracted_text
                st.success(T["pdf_success"].format(count=len(extracted_text.split())))

    # ── Right: Job Description ───────────────────────────────────────────
    with right_col:
        st.markdown(f"""
        <div class="hw-card-header">
            <div class="hw-card-title">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#059669" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-3px; margin-right:8px;"><rect x="2" y="7" width="20" height="14" rx="2" ry="2"/><path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"/></svg>
                <span>{T['jd_title']}</span>
            </div>
        </div>
        <div class="hw-card-desc">{T['jd_desc']}</div>
        """, unsafe_allow_html=True)

        tab_j_text, tab_j_url = st.tabs([T["paste_jd"], T["fetch_url"]])

        with tab_j_text:
            st.session_state.jd_input = st.text_area(
                "jd_text_area",
                value=st.session_state.jd_input,
                height=250,
                placeholder=T["jd_placeholder"],
                label_visibility="collapsed",
            )
            j_word_count = len(st.session_state.jd_input.split()) if st.session_state.jd_input else 0
            st.markdown(
                f'<div style="text-align:right; font-size:0.75rem; color:#64748b; margin-top:4px;"><b>{j_word_count}</b> {T["words_detected"]}</div>',
                unsafe_allow_html=True,
            )

        with tab_j_url:
            target_url = st.text_input(
                "URL",
                placeholder=T["url_placeholder"],
                label_visibility="collapsed",
            )
            if st.button(T["btn_fetch"], icon=":material/download:", key="btn_fetch_url", use_container_width=True):
                if target_url.strip():
                    with st.spinner(T["computing"]):
                        success, content = extract_job_description_from_url(target_url.strip())
                        if success:
                            st.session_state.jd_input = content
                            st.success(T["url_success"].format(count=len(content.split())))
                            st.rerun()
                        else:
                            st.error(content)


    # ══════════════════════════════════════════════════════════════════════
    # MAIN ACTION BUTTON
    # ══════════════════════════════════════════════════════════════════════
    st.markdown('<div style="height: 1.4rem"></div>', unsafe_allow_html=True)

    _, btn_center, _ = st.columns([1, 2, 1])
    with btn_center:
        compute_btn = st.button(
            T["btn_analyze"],
            type="primary",
            icon=":material/bolt:",
            use_container_width=True,
        )


    # ══════════════════════════════════════════════════════════════════════
    # EXECUTION PIPELINE (Fixed internal neural hybrid model)
    # ══════════════════════════════════════════════════════════════════════
    if compute_btn:
        r_clean = st.session_state.resume_input.strip()
        j_clean = st.session_state.jd_input.strip()

        if not r_clean:
            st.error(T["err_resume"])
            st.stop()
        if not j_clean:
            st.error(T["err_jd"])
            st.stop()

        with st.spinner(T["computing"]):
            match_res = match(r_clean, j_clean, weights=active_weights, skill_method="hybrid")
            recs_list = generate_recommendations(match_res)
            st.session_state.result = match_res
            st.session_state.recs = recs_list
            st.session_state.tailored_cv_text = None
            st.session_state.tailor_boost_score = None
            st.session_state.applied_sugs = set()
            st.session_state.current_page = "results"
            st.rerun()



# ══════════════════════════════════════════════════════════════════════
# PAGE 2: RÉSULTATS & STUDIO D'OPTIMISATION
# ══════════════════════════════════════════════════════════════════════
elif st.session_state.current_page == "results":
    if not st.session_state.result:
        st.session_state.current_page = "input"
        st.rerun()

    # Back navigation button to Page 1
    back_col, _ = st.columns([1.6, 4.4])
    with back_col:
        if st.button(T["back_btn"], icon=":material/arrow_back:", key="btn_nav_back", use_container_width=True):
            st.session_state.current_page = "input"
            st.rerun()

    st.markdown('<div style="height: 0.6rem"></div>', unsafe_allow_html=True)
    res = st.session_state.result
    recs = st.session_state.recs
    score = res["match_percentage"]

    st.markdown('<div style="height: 1.5rem"></div>', unsafe_allow_html=True)
    st.markdown('<hr style="border: none; border-top: 1px solid #e2e8f0; margin-bottom: 2rem;">', unsafe_allow_html=True)

    # ── Verdict Banner ────────────────────────────────────────────────
    if score >= 75:
        box_class, pill_class = "strong", "pill-strong"
        badge_text = T["verdict_strong_pill"]
        v_title = T["verdict_strong_title"]
        v_desc = T["verdict_strong_sub"]
    elif score >= 55:
        box_class, pill_class = "moderate", "pill-moderate"
        badge_text = T["verdict_moderate_pill"]
        v_title = T["verdict_moderate_title"]
        v_desc = T["verdict_moderate_sub"]
    else:
        box_class, pill_class = "weak", "pill-weak"
        badge_text = T["verdict_weak_pill"]
        v_title = T["verdict_weak_title"]
        v_desc = T["verdict_weak_sub"]

    st.markdown(f"""
    <div class="hw-verdict-box {box_class}">
        <div>
            <div class="hw-verdict-text">{v_title} ({score}%)</div>
            <div class="hw-verdict-sub">{v_desc}</div>
        </div>
        <div class="hw-verdict-pill {pill_class}">{badge_text}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── 4 Metric Tiles ────────────────────────────────────────────────
    cov_pct = round(res["skill_overlap"] * 100, 1)
    st.markdown(f"""
    <div class="hw-metrics-row">
        <div class="hw-metric-tile">
            <div class="hw-tile-label">{T['metric_score']}</div>
            <div class="hw-tile-value" style="color: #0f172a;">{score}%</div>
            <div class="hw-tile-foot">{T['metric_score_sub']}</div>
        </div>
        <div class="hw-metric-tile">
            <div class="hw-tile-label">{T['metric_sem']}</div>
            <div class="hw-tile-value" style="color: #2563eb;">{res['semantic_similarity']}</div>
            <div class="hw-tile-foot">{T['metric_sem_sub']}</div>
        </div>
        <div class="hw-metric-tile">
            <div class="hw-tile-label">{T['metric_ats']}</div>
            <div class="hw-tile-value" style="color: #7c3aed;">{res['tfidf_similarity']}</div>
            <div class="hw-tile-foot">{T['metric_ats_sub']}</div>
        </div>
        <div class="hw-metric-tile">
            <div class="hw-tile-label">{T['metric_skills']}</div>
            <div class="hw-tile-value" style="color: #059669;">{cov_pct}%</div>
            <div class="hw-tile-foot">{len(res['matched_skills'])} / {len(res['jd_skills'])} {T['metric_skills_sub']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Interactive Charts ────────────────────────────────────────────
    chart_col1, chart_col2 = st.columns([1, 1.3], gap="medium")

    with chart_col1:
        st.markdown(f"""
        <div style="font-size: 0.95rem; font-weight: 800; color: #0f172a; margin-bottom: 0.4rem;">
            {T['gauge_title']}
        </div>
        """, unsafe_allow_html=True)

        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=score,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": T["gauge_title"], "font": {"size": 13, "color": "#64748b", "family": "Plus Jakarta Sans"}},
                number={"suffix": "%", "font": {"size": 42, "color": "#0f172a", "family": "Plus Jakarta Sans"}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#cbd5e1"},
                    "bar": {"color": "#0f172a", "thickness": 0.28},
                    "bgcolor": "#ffffff",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 55], "color": "#fee2e2"},
                        {"range": [55, 75], "color": "#fef3c7"},
                        {"range": [75, 100], "color": "#d1fae5"},
                    ],
                },
            )
        )
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"family": "Plus Jakarta Sans"},
            height=260,
            margin=dict(l=20, r=20, t=30, b=10),
        )
        st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})

    with chart_col2:
        st.markdown(f"""
        <div style="font-size: 0.95rem; font-weight: 800; color: #0f172a; margin-bottom: 0.4rem;">
            {T['signal_contrib_title']}
        </div>
        """, unsafe_allow_html=True)

        categories = T["signals"]
        raw_values = [res["semantic_similarity"], res["tfidf_similarity"], res["skill_overlap"]]
        weighted_values = [
            round(res["semantic_similarity"] * active_weights["semantic"], 3),
            round(res["tfidf_similarity"] * active_weights["tfidf"], 3),
            round(res["skill_overlap"] * active_weights["skill"], 3),
        ]

        fig_bar = go.Figure()
        fig_bar.add_trace(
            go.Bar(
                name=T["raw_score"],
                x=categories,
                y=raw_values,
                marker_color="#cbd5e1",
                text=[f"{v:.2f}" for v in raw_values],
                textposition="auto",
            )
        )
        fig_bar.add_trace(
            go.Bar(
                name=T["weighted_impact"],
                x=categories,
                y=weighted_values,
                marker_color="#2563eb",
                text=[f"{v:.2f}" for v in weighted_values],
                textposition="auto",
            )
        )
        fig_bar.update_layout(
            barmode="group",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"family": "Plus Jakarta Sans", "color": "#0f172a"},
            height=260,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=30, b=20),
            yaxis=dict(gridcolor="#f1f5f9", range=[0, 1.1]),
            xaxis=dict(gridcolor="#f1f5f9"),
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

    # ── Skills Matrix with Vector Badges ──────────────────────────────
    st.markdown('<div style="height: 1.2rem"></div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size: 1.15rem; font-weight: 800; color: #0f172a; margin-bottom: 0.6rem;">
        {T['skills_title']}
    </div>
    """, unsafe_allow_html=True)

    sk_col1, sk_col2, sk_col3 = st.columns(3, gap="medium")

    check_svg = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#065f46" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-1px; margin-right:4px;"><polyline points="20 6 9 17 4 12"/></svg>'
    cross_svg = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#991b1b" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-1px; margin-right:4px;"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>'
    plus_svg = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#166534" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:-1px; margin-right:4px;"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>'

    with sk_col1:
        st.markdown(f"""
        <div style="font-size: 0.86rem; font-weight: 700; color: #065f46; margin-bottom: 8px;">
            {check_svg} {T['skills_matched']} ({len(res['matched_skills'])})
        </div>
        """, unsafe_allow_html=True)
        if res["matched_skills"]:
            tags_html = "".join([f'<span class="hw-tag hw-tag-matched">{check_svg} {s}</span>' for s in sorted(res["matched_skills"])])
            st.markdown(f'<div>{tags_html}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="font-size:0.8rem; color:#94a3b8;">{T["no_matched"]}</div>', unsafe_allow_html=True)

    with sk_col2:
        st.markdown(f"""
        <div style="font-size: 0.86rem; font-weight: 700; color: #991b1b; margin-bottom: 8px;">
            {cross_svg} {T['skills_missing']} ({len(res['missing_skills'])})
        </div>
        """, unsafe_allow_html=True)
        if res["missing_skills"]:
            tags_html = "".join([f'<span class="hw-tag hw-tag-missing">{cross_svg} {s}</span>' for s in sorted(res["missing_skills"])])
            st.markdown(f'<div>{tags_html}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="font-size:0.8rem; color:#059669;">{T["all_matched"]}</div>', unsafe_allow_html=True)

    with sk_col3:
        bonus_skills = [s for s in res["resume_skills"] if s not in set(res["matched_skills"])]
        st.markdown(f"""
        <div style="font-size: 0.86rem; font-weight: 700; color: #166534; margin-bottom: 8px;">
            {plus_svg} {T['skills_bonus']} ({len(bonus_skills)})
        </div>
        """, unsafe_allow_html=True)
        if bonus_skills:
            tags_html = "".join([f'<span class="hw-tag hw-tag-bonus">{plus_svg} {s}</span>' for s in sorted(bonus_skills[:12])])
            st.markdown(f'<div>{tags_html}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="font-size:0.8rem; color:#94a3b8;">{T["no_bonus"]}</div>', unsafe_allow_html=True)

    # ── Strategic Recommendations ─────────────────────────────────────
    st.markdown('<div style="height: 1.5rem"></div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size: 1.15rem; font-weight: 800; color: #0f172a; margin-bottom: 0.6rem;">
        {T['recs_title']}
    </div>
    """, unsafe_allow_html=True)

    if recs:
        for rec in recs:
            rec_l = rec.lower()
            if "high-priority" in rec_l or "deal-breaker" in rec_l or "weak match" in rec_l:
                b_class = "hw-badge-critical"
                b_text = T["badge_critical"]
            elif "moderate" in rec_l or "other missing" in rec_l or "keyword" in rec_l:
                b_class = "hw-badge-advisory"
                b_text = T["badge_advisory"]
            else:
                b_class = "hw-badge-positive"
                b_text = T["badge_positive"]

            st.markdown(f"""
            <div class="hw-rec-card">
                <span class="hw-rec-badge {b_class}">{b_text}</span>
                <span class="hw-rec-text">{rec}</span>
            </div>
            """, unsafe_allow_html=True)

    # ── ATS OPTIMIZATION STUDIO & COPILOT ─────────────────────────────
    st.markdown('<div style="height: 1.8rem"></div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); border-radius: 20px; padding: 2rem 2.2rem; color: #ffffff; margin-bottom: 1.8rem; box-shadow: 0 10px 30px rgba(15,23,42,0.12); position: relative; overflow: hidden;">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 1rem; margin-bottom: 0.6rem;">
            <div style="font-size: 1.3rem; font-weight: 900; letter-spacing: -0.03em; display: flex; align-items: center; gap: 8px;">
                <span>✨</span>
                <span>{T['tailor_studio_title']}</span>
            </div>
            <div style="background: rgba(37, 99, 235, 0.25); border: 1px solid rgba(147, 197, 253, 0.35); border-radius: 9999px; padding: 4px 14px; font-size: 0.74rem; font-weight: 800; color: #93c5fd; letter-spacing: 0.04em;">
                ATS ENGINE OPTIMIZER v2.0
            </div>
        </div>
        <div style="font-size: 0.9rem; color: #cbd5e1; line-height: 1.6; max-width: 820px;">
            {T['tailor_studio_sub']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    potential_score = min(98.5, round(score + max(14.0, (100.0 - score) * 0.58), 1))

    if st.session_state.tailored_cv_text is None:
        st.markdown(f"""
        <div style="background: #ffffff; border: 1.5px solid #e2e8f0; border-radius: 16px; padding: 1.4rem 1.6rem; margin-bottom: 1.6rem; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 1rem; box-shadow: 0 4px 14px rgba(15,23,42,0.03);">
            <div>
                <div style="font-size: 0.82rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.06em; color: #64748b;">
                    {T['tailor_gain_title']}
                </div>
                <div style="font-size: 1.05rem; font-weight: 800; color: #0f172a; margin-top: 3px;">
                    Score actuel : <b>{score}%</b> ➔ Score projeté après optimisations : <b style="color: #059669;">{potential_score}%</b>
                </div>
            </div>
            <div style="background: #ecfdf5; color: #065f46; border: 1px solid #a7f3d0; border-radius: 9999px; padding: 6px 16px; font-size: 0.85rem; font-weight: 800;">
                +{round(potential_score - score, 1)}% GAIN ATS PROJETÉ
            </div>
        </div>
        """, unsafe_allow_html=True)

        _, opt_btn_col, _ = st.columns([1, 2.5, 1])
        with opt_btn_col:
            if st.button(
                T["btn_optimize_now"],
                icon=":material/auto_fix_high:",
                type="primary",
                key="btn_run_cv_opt",
                use_container_width=True,
            ):
                with st.spinner(T["optimizing_spinner"]):
                    tailored_latex = tailor_cv_to_latex(st.session_state.resume_input, st.session_state.jd_input, res)
                    st.session_state.tailored_cv_text = tailored_latex

                    clean_txt = latex_to_clean_text(tailored_latex)
                    new_eval = match(clean_txt, st.session_state.jd_input, weights=active_weights, skill_method="hybrid")
                    st.session_state.tailor_boost_score = new_eval["match_percentage"]
                    st.toast(T["tailor_applied"], icon="🎉")
                    st.rerun()

    else:
        # Optimization already generated
        boosted_score = st.session_state.tailor_boost_score or potential_score
        gain_val = round(boosted_score - score, 1)

        st.markdown(f"""
        <div style="background: #ecfdf5; border: 1.5px solid #10b981; border-radius: 16px; padding: 1.2rem 1.6rem; margin-bottom: 1.5rem; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 1rem; box-shadow: 0 4px 18px rgba(16,185,129,0.08);">
            <div>
                <div style="font-size: 0.8rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.05em; color: #047857;">
                    ✓ {T['tailor_applied']}
                </div>
                <div style="font-size: 1.1rem; font-weight: 800; color: #065f46; margin-top: 3px;">
                    Score initial : <b>{score}%</b> ➔ Nouveau Score Boosté : <b style="color: #059669; font-size: 1.25rem;">{boosted_score}%</b>
                </div>
            </div>
            <div style="background: #059669; color: #ffffff; border-radius: 9999px; padding: 6px 18px; font-size: 0.88rem; font-weight: 800;">
                +{gain_val}% GAIN ATS RÉALISÉ
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── THE RECTANGLE DISPLAYING THE OPTIMIZED CV ──
        word_count_opt = len(st.session_state.tailored_cv_text.split())
        st.markdown(f"""
        <div style="background: #ffffff; border: 2px solid #2563eb; border-radius: 16px; padding: 1.4rem 1.6rem 0.8rem 1.6rem; box-shadow: 0 10px 30px rgba(37,99,235,0.08); margin-bottom: 1rem;">
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px; margin-bottom: 8px; border-bottom: 1px solid #e2e8f0; padding-bottom: 10px;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#2563eb" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>
                    <span style="font-size: 1.05rem; font-weight: 900; color: #0f172a;">{T['optimized_preview_title']}</span>
                </div>
                <div style="display: flex; gap: 6px;">
                    <span style="background: #eff6ff; color: #1d4ed8; font-size: 0.74rem; font-weight: 800; padding: 3px 10px; border-radius: 9999px; border: 1px solid #bfdbfe;">✓ ATS Optimisé</span>
                    <span style="background: #f1f5f9; color: #475569; font-size: 0.74rem; font-weight: 700; padding: 3px 10px; border-radius: 9999px;">{word_count_opt} {T['words_detected']}</span>
                </div>
            </div>
            <div style="font-size: 0.82rem; color: #64748b; margin-bottom: 8px;">{T['optimized_preview_sub']}</div>
        </div>
        """, unsafe_allow_html=True)

        tab_tex, tab_preview = st.tabs(["📄 Code Source LaTeX (.tex)", "👁️ Aperçu Texte Nettoyé"])
        with tab_tex:
            updated_tailored = st.text_area(
                "optimized_cv_box_tex",
                value=st.session_state.tailored_cv_text,
                height=380,
                help=T["cv_editor_help"],
                label_visibility="collapsed",
            )
            if updated_tailored != st.session_state.tailored_cv_text:
                st.session_state.tailored_cv_text = updated_tailored
        with tab_preview:
            st.text_area(
                "optimized_cv_box_preview",
                value=latex_to_clean_text(st.session_state.tailored_cv_text),
                height=380,
                disabled=True,
                label_visibility="collapsed",
            )

        # ── HIGH-FIDELITY PDF EXPORT CARD ──
        st.markdown('<div style="height: 0.8rem"></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background: #ffffff; border: 1.5px solid #e2e8f0; border-radius: 18px; padding: 1.4rem 1.8rem; box-shadow: 0 4px 16px rgba(15,23,42,0.03); margin-bottom: 1.2rem;">
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 1rem; margin-bottom: 0.4rem;">
                <div style="font-size: 1.1rem; font-weight: 900; color: #0f172a; display: flex; align-items: center; gap: 8px;">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#2563eb" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
                    <span>{T['pdf_card_title']}</span>
                </div>
                <div style="display: flex; gap: 6px; flex-wrap: wrap;">
                    <span style="background: #eff6ff; color: #1d4ed8; font-size: 0.72rem; font-weight: 800; padding: 3px 9px; border-radius: 9999px; border: 1px solid #bfdbfe;">pdflatex natif</span>
                    <span style="background: #f1f5f9; color: #334155; font-size: 0.72rem; font-weight: 700; padding: 3px 9px; border-radius: 9999px;">1 Page Exacte (A4)</span>
                    <span style="background: #f1f5f9; color: #334155; font-size: 0.72rem; font-weight: 700; padding: 3px 9px; border-radius: 9999px;">Latin Modern Roman</span>
                </div>
            </div>
            <div style="font-size: 0.83rem; color: #64748b; line-height: 1.5;">
                {T['pdf_card_desc']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        pdf_bytes = compile_latex_to_pdf(st.session_state.tailored_cv_text)

        dl_col1, dl_col2, dl_col3 = st.columns([2.5, 1.5, 1])
        with dl_col1:
            st.download_button(
                label=T["btn_download_tailored_pdf"],
                icon=":material/picture_as_pdf:",
                data=pdf_bytes,
                file_name="cv_optimise_ats.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True,
            )
        with dl_col2:
            st.download_button(
                label="📄 Source LaTeX (.tex)",
                icon=":material/code:",
                data=st.session_state.tailored_cv_text,
                file_name="cv_optimise_ats.tex",
                mime="text/x-tex",
                use_container_width=True,
            )
        with dl_col3:
            if st.button("🔄 " + T.get("reset_opt", "Réinitialiser"), key="btn_reset_tailor", use_container_width=True):
                st.session_state.tailored_cv_text = None
                st.session_state.tailor_boost_score = None
                st.rerun()


    # ── Export JSON ───────────────────────────────────────────────────
    st.markdown('<div style="height: 0.8rem"></div>', unsafe_allow_html=True)
    with st.expander(T["export_title"]):
        report_data = {
            "overall_match_score": res["match_percentage"],
            "signals": {
                "semantic_similarity": res["semantic_similarity"],
                "tfidf_similarity": res["tfidf_similarity"],
                "skill_overlap": res["skill_overlap"],
            },
            "weights_used": active_weights,
            "skills": {
                "matched": res["matched_skills"],
                "missing": res["missing_skills"],
                "candidate_total": res["resume_skills"],
                "job_total": res["jd_skills"],
            },
            "recommendations": recs,
        }
        json_str = json.dumps(report_data, indent=2, ensure_ascii=False)

        st.download_button(
            label=T["btn_download"],
            icon=":material/download:",
            data=json_str,
            file_name="cvmatch_report.json",
            mime="application/json",
            use_container_width=True,
        )

