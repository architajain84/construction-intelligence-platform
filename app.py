"""
Multimodal Construction Intelligence Platform
Skill Wallet Capstone Project 1

A Streamlit app that analyses construction site photos for PPE safety,
diagnoses concrete cracks, estimates project cost and real-estate value
in INR, and gives weather-aware daily site advice with a chat assistant.

Works in two modes:
  * Live AI mode  - uses Google Gemini (multimodal) when an API key is given.
  * Demo mode     - deterministic built-in analysis, no key needed.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import random
import urllib.parse
from datetime import datetime

import requests
import streamlit as st
from PIL import Image, ImageStat

# ----------------------------------------------------------------------------
# Page setup and styling
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Construction Intelligence Platform",
    page_icon="🏗️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main-title {font-size: 2.1rem; font-weight: 800; margin-bottom: 0;}
    .sub-title {color: #6b7280; margin-top: 0.2rem; margin-bottom: 1.2rem;}
    .pill {display:inline-block; padding:4px 12px; border-radius:999px;
           font-weight:600; font-size:0.85rem; color:white;}
    .pill-low {background:#16a34a;} .pill-medium {background:#ca8a04;}
    .pill-high {background:#ea580c;} .pill-critical {background:#dc2626;}
    .card {border:1px solid rgba(128,128,128,0.25); border-radius:12px;
           padding:14px 16px; margin-bottom:10px;}
    </style>
    """,
    unsafe_allow_html=True,
)

SEVERITY_CLASS = {"low": "pill-low", "medium": "pill-medium",
                  "high": "pill-high", "critical": "pill-critical"}


def pill(text: str) -> str:
    cls = SEVERITY_CLASS.get(str(text).lower(), "pill-medium")
    return f'<span class="pill {cls}">{text}</span>'


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input(
        "Google Gemini API key (optional)",
        type="password",
        help="Leave empty to use Demo mode. Get a free key at aistudio.google.com",
    )
    model_name = st.text_input("Gemini model", value="gemini-2.5-flash")
    LIVE = bool(api_key.strip())
    if LIVE:
        st.success("Live AI mode: Gemini multimodal analysis")
    else:
        st.info("Demo mode: built-in analysis engine (no key needed)")

    st.divider()
    st.subheader("🧭 Architecture")
    st.markdown(
        """
        - **UI:** Streamlit
        - **AI layer:** Gemini multimodal LLM via REST
        - **Vision prep:** Pillow
        - **Contracts:** strict JSON outputs
        - **Rendering:** Pollinations AI
        - **Storage:** none (stateless, cloud-ready)
        """
    )
    st.divider()
    st.subheader("📌 Modules")
    st.markdown(
        """
        1. Site Safety & PPE Audit
        2. Crack Diagnostics
        3. Planning & Valuation
        4. Site Assistant & Weather
        """
    )


# ----------------------------------------------------------------------------
# AI helpers
# ----------------------------------------------------------------------------
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"


def image_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def gemini_json(prompt: str, img: Image.Image | None = None) -> dict:
    """Call Gemini and return parsed JSON. Raises on any failure."""
    parts = [{"text": prompt}]
    if img is not None:
        parts.append({"inline_data": {"mime_type": "image/jpeg",
                                      "data": image_to_b64(img)}})
    body = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {"response_mime_type": "application/json",
                             "temperature": 0.2},
    }
    r = requests.post(GEMINI_URL.format(model=model_name.strip(), key=api_key.strip()),
                      json=body, timeout=60)
    r.raise_for_status()
    text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
    return json.loads(text)


def gemini_chat(history: list[dict], system: str) -> str:
    contents = [{"role": "user" if m["role"] == "user" else "model",
                 "parts": [{"text": m["content"]}]} for m in history]
    body = {"system_instruction": {"parts": [{"text": system}]},
            "contents": contents,
            "generationConfig": {"temperature": 0.5}}
    r = requests.post(GEMINI_URL.format(model=model_name.strip(), key=api_key.strip()),
                      json=body, timeout=60)
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]


def seeded_rng(img: Image.Image) -> random.Random:
    """Same image always gives the same demo result."""
    digest = hashlib.md5(img.convert("RGB").resize((64, 64)).tobytes()).hexdigest()
    return random.Random(int(digest[:8], 16))


def image_features(img: Image.Image) -> dict:
    gray = img.convert("L")
    stat = ImageStat.Stat(gray)
    return {"brightness": round(stat.mean[0], 1), "contrast": round(stat.stddev[0], 1),
            "width": img.width, "height": img.height}


# ----------------------------------------------------------------------------
# Demo-mode analysis engines (deterministic fallbacks)
# ----------------------------------------------------------------------------
def demo_safety(img: Image.Image) -> dict:
    rng = seeded_rng(img)
    feats = image_features(img)
    ppe_items = ["Hard hat", "High-visibility vest", "Steel-toe boots",
                 "Safety harness", "Gloves", "Eye protection"]
    checklist = {item: rng.random() > 0.3 for item in ppe_items}
    compliant = sum(checklist.values())
    hazard_pool = [
        ("Unprotected edge / fall risk", "Critical", "Install guardrails and enforce harness tie-off above 2 m."),
        ("Loose material on walkways", "Medium", "Clear debris and mark walkways; schedule hourly housekeeping."),
        ("Exposed rebar ends", "High", "Fit rebar caps on all protruding bars immediately."),
        ("Poor lighting in work zone", "Medium", "Add temporary flood lights before evening shift."),
        ("Unsecured scaffolding", "High", "Inspect and tag scaffold; tie into structure as per IS 3696."),
        ("Electrical cables on wet ground", "Critical", "Elevate cables and use RCD-protected boards."),
        ("Missing safety signage", "Low", "Place PPE-mandatory and danger signs at entry points."),
    ]
    hazards = rng.sample(hazard_pool, k=rng.randint(2, 4))
    if feats["brightness"] < 70:
        hazards.append(("Low visibility in image area", "Medium", "Improve site lighting and re-inspect."))
    penalty = sum({"Low": 3, "Medium": 7, "High": 12, "Critical": 18}[h[1]] for h in hazards)
    score = max(15, min(98, int(compliant / len(ppe_items) * 100) - penalty // 2))
    status = "Safe" if score >= 80 else "Needs Attention" if score >= 55 else "Unsafe"
    return {
        "safety_score": score,
        "overall_status": status,
        "workers_detected": rng.randint(1, 9),
        "ppe_detection": checklist,
        "hazards": [{"hazard": h, "severity": s, "remedy": r} for h, s, r in hazards],
        "standards_referenced": ["OSHA 1926", "IS 456:2000", "IS 3696"],
    }


def demo_crack(img: Image.Image) -> dict:
    rng = seeded_rng(img)
    feats = image_features(img)
    types = [
        ("Hairline", (0.1, 0.3), "Low", "Plastic shrinkage during curing",
         "Seal with low-viscosity epoxy or cementitious slurry; monitor."),
        ("Thermal Expansion", (0.3, 1.0), "Medium", "Temperature cycling without expansion joints",
         "Route-and-seal with flexible polyurethane sealant; add movement joint."),
        ("Settlement", (1.0, 3.0), "High", "Differential foundation settlement",
         "Install tell-tale monitors; investigate soil; underpin if movement continues."),
        ("Spalling", (0.5, 2.0), "High", "Rebar corrosion pushing cover concrete off",
         "Remove loose concrete, treat rebar, apply polymer-modified repair mortar."),
        ("Structural Shear", (2.0, 6.0), "Critical", "Overloading or inadequate shear reinforcement",
         "Restrict loading; structural engineer review; epoxy injection plus FRP wrapping."),
    ]
    idx = min(len(types) - 1, int(feats["contrast"] / 15) + rng.randint(0, 1))
    name, (lo, hi), urgency, cause, repair = types[idx]
    width = round(rng.uniform(lo, hi), 2)
    return {
        "defect_type": name,
        "estimated_width_mm": width,
        "severity_score": {"Low": 20, "Medium": 45, "High": 70, "Critical": 90}[urgency] + rng.randint(-5, 5),
        "urgency": urgency,
        "probable_root_causes": [cause, "Inadequate curing or poor concrete mix"],
        "recommended_repairs": [repair, "Re-inspect in 30 days and log crack width"],
        "structural_integrity": "Compromised" if urgency in ("High", "Critical") else "Sound",
    }


def demo_chat(question: str, weather: str) -> str:
    q = question.lower()
    tips = {
        "curing": "Keep concrete moist for at least 7 days (10+ days for blended cements) as per IS 456. Use wet hessian or curing compound.",
        "cement": "Typical RCC residential work uses about 0.4 bags of cement per sq ft of built-up area.",
        "steel": "Residential RCC usually needs about 3.5 to 4.5 kg of steel per sq ft of built-up area.",
        "helmet": "Every worker must wear an ISI-marked helmet (IS 2925) at all times on site.",
        "crack": "Use the Crack Diagnostics tab to upload a photo. Hairline cracks under 0.3 mm are usually cosmetic.",
        "rain": "In rain: stop concreting, cover fresh pours, and switch off temporary electrics in wet zones.",
        "wind": "Suspend crane lifts when wind exceeds about 38 km/h and secure loose sheets.",
        "heat": "In extreme heat: pour concrete early morning, use chilled water, and give workers shade and water breaks.",
    }
    for key, ans in tips.items():
        if key in q:
            return ans
    return (f"Good question. With today's weather set to **{weather}**, start with a toolbox talk, "
            "check PPE at the gate, and review the hazard list from the Safety tab. "
            "Add a Gemini API key in the sidebar for detailed AI answers.")


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown('<p class="main-title">🏗️ Multimodal Construction Intelligence Platform</p>',
            unsafe_allow_html=True)
st.markdown('<p class="sub-title">AI decision support for site safety, structural health, '
            'planning, valuation and daily operations.</p>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "🦺 Site Safety & PPE Audit",
    "🧱 Crack Diagnostics",
    "📐 Planning & Valuation",
    "🌦️ Site Assistant & Weather",
])

# ----------------------------------------------------------------------------
# Tab 1: Safety
# ----------------------------------------------------------------------------
SAFETY_PROMPT = """You are a construction safety auditor. Analyse this site photo for worker PPE
compliance and hazards using OSHA 1926 and IS 456 / IS 3696 practice. Respond ONLY with JSON:
{"safety_score": int 0-100, "overall_status": "Safe"|"Needs Attention"|"Unsafe",
 "workers_detected": int,
 "ppe_detection": {"Hard hat": bool, "High-visibility vest": bool, "Steel-toe boots": bool,
                   "Safety harness": bool, "Gloves": bool, "Eye protection": bool},
 "hazards": [{"hazard": str, "severity": "Low"|"Medium"|"High"|"Critical", "remedy": str}],
 "standards_referenced": [str]}"""

with tab1:
    st.subheader("Site Safety & PPE Compliance Audit")
    up = st.file_uploader("Upload a construction site photo", type=["jpg", "jpeg", "png"], key="safety")
    if up:
        try:
            img = Image.open(up)
            c1, c2 = st.columns([1, 1.2])
            c1.image(img, caption="Uploaded site photo")
            with st.spinner("Auditing site safety..."):
                try:
                    result = gemini_json(SAFETY_PROMPT, img) if LIVE else demo_safety(img)
                except Exception as e:
                    st.warning(f"Live AI unavailable ({e.__class__.__name__}). Showing demo analysis.")
                    result = demo_safety(img)
            with c2:
                m1, m2, m3 = st.columns(3)
                m1.metric("Safety score", f"{result['safety_score']}/100")
                m2.metric("Status", result["overall_status"])
                m3.metric("Workers", result.get("workers_detected", "-"))
                st.progress(int(result["safety_score"]) / 100)
                st.markdown("**PPE checklist**")
                for item, ok in result["ppe_detection"].items():
                    st.markdown(f"{'✅' if ok else '❌'} {item}")
            st.markdown("#### Identified hazards")
            for h in result["hazards"]:
                st.markdown(
                    f'<div class="card"><b>{h["hazard"]}</b> &nbsp; {pill(h["severity"])}'
                    f'<br><span style="opacity:.8">Remedy: {h["remedy"]}</span></div>',
                    unsafe_allow_html=True)
            result["generated_at"] = datetime.now().isoformat(timespec="seconds")
            with st.expander("View JSON data contract"):
                st.json(result)
            st.download_button("⬇️ Download JSON report", json.dumps(result, indent=2),
                               file_name="safety_report.json", mime="application/json")
        except Exception as e:
            st.error(f"Could not process this image: {e}")
    else:
        st.caption("Tip: use any photo of a building site with workers.")

# ----------------------------------------------------------------------------
# Tab 2: Cracks
# ----------------------------------------------------------------------------
CRACK_PROMPT = """You are a structural engineer. Classify the concrete/wall defect in this photo.
Respond ONLY with JSON:
{"defect_type": "Hairline"|"Settlement"|"Structural Shear"|"Thermal Expansion"|"Spalling",
 "estimated_width_mm": float, "severity_score": int 0-100,
 "urgency": "Low"|"Medium"|"High"|"Critical",
 "probable_root_causes": [str], "recommended_repairs": [str],
 "structural_integrity": "Sound"|"Compromised"}"""

with tab2:
    st.subheader("Crack Diagnostics & Structural Integrity")
    up2 = st.file_uploader("Upload a photo of a concrete or wall surface", type=["jpg", "jpeg", "png"], key="crack")
    if up2:
        try:
            img2 = Image.open(up2)
            c1, c2 = st.columns([1, 1.2])
            c1.image(img2, caption="Surface photo")
            with st.spinner("Diagnosing defect..."):
                try:
                    res = gemini_json(CRACK_PROMPT, img2) if LIVE else demo_crack(img2)
                except Exception as e:
                    st.warning(f"Live AI unavailable ({e.__class__.__name__}). Showing demo analysis.")
                    res = demo_crack(img2)
            with c2:
                m1, m2, m3 = st.columns(3)
                m1.metric("Defect type", res["defect_type"])
                m2.metric("Width (est.)", f"{res['estimated_width_mm']} mm")
                m3.metric("Severity", f"{res['severity_score']}/100")
                st.markdown(f"**Urgency:** {pill(res['urgency'])} &nbsp; "
                            f"**Structural integrity:** {res['structural_integrity']}",
                            unsafe_allow_html=True)
                st.markdown("**Probable root causes**")
                for c in res["probable_root_causes"]:
                    st.markdown(f"- {c}")
                st.markdown("**Recommended repairs**")
                for r in res["recommended_repairs"]:
                    st.markdown(f"- {r}")
            with st.expander("View JSON data contract"):
                st.json(res)
            st.download_button("⬇️ Download JSON report", json.dumps(res, indent=2),
                               file_name="crack_report.json", mime="application/json")
        except Exception as e:
            st.error(f"Could not process this image: {e}")

# ----------------------------------------------------------------------------
# Tab 3: Planning & valuation
# ----------------------------------------------------------------------------
COST_PER_SQFT = {"Residential House": 1900, "Apartment Building": 2300,
                 "Commercial Office": 2800, "Warehouse / Industrial": 1500}
LAND_RATE = {"Tier 1 (Mumbai, Delhi, Bengaluru...)": 9000,
             "Tier 2 (Indore, Jaipur, Pune...)": 4500,
             "Tier 3 (smaller cities)": 2200}


def inr(amount: float) -> str:
    if amount >= 1e7:
        return f"₹{amount / 1e7:.2f} Cr"
    if amount >= 1e5:
        return f"₹{amount / 1e5:.2f} Lakh"
    return f"₹{amount:,.0f}"


with tab3:
    st.subheader("Project Planning & Real Estate Valuation")
    c1, c2 = st.columns(2)
    with c1:
        structure = st.selectbox("Structure type", list(COST_PER_SQFT))
        sqft = st.number_input("Built-up area (sq ft)", min_value=200, max_value=500000, value=2000, step=100)
    with c2:
        tier = st.selectbox("Location tier", list(LAND_RATE))
        progress = st.slider("Completion (%)", 0, 100, 40)

    budget = sqft * COST_PER_SQFT[structure]
    spent = budget * progress / 100
    market_value = sqft * LAND_RATE[tier] * (0.6 + 0.4 * progress / 100) + spent
    cement_bags = sqft * 0.4
    steel_kg = sqft * 4.0

    m1, m2, m3 = st.columns(3)
    m1.metric("Total project budget", inr(budget))
    m2.metric("Capital spent so far", inr(spent))
    m3.metric("Current market value", inr(market_value))
    st.progress(progress / 100, text=f"Milestone progress: {progress}%")

    m4, m5, m6 = st.columns(3)
    m4.metric("Cement required", f"{cement_bags:,.0f} bags")
    m5.metric("Steel required", f"{steel_kg / 1000:,.1f} tonnes")
    phase = ("Foundation" if progress < 20 else "Superstructure" if progress < 60
             else "Finishing" if progress < 95 else "Handover")
    m6.metric("Current phase", phase)

    st.markdown("#### 🎨 Conceptual 3D rendering")
    style = st.selectbox("Architectural style", ["Modern minimalist", "Contemporary Indian",
                                                 "Glass facade", "Eco-friendly green"])
    if st.button("Generate concept render"):
        prompt = (f"{style} {structure.lower()} in an Indian city, {sqft} square feet, "
                  "photorealistic architectural 3D render, daylight")
        url = "https://image.pollinations.ai/prompt/" + urllib.parse.quote(prompt) + "?width=768&height=432&nologo=true"
        st.markdown(f'<img src="{url}" style="width:100%;border-radius:12px" alt="Concept render">',
                    unsafe_allow_html=True)
        st.caption("Rendered by Pollinations AI. It may take a few seconds to appear.")

# ----------------------------------------------------------------------------
# Tab 4: Weather + assistant
# ----------------------------------------------------------------------------
WEATHER_ALERTS = {
    "Normal": ("Low", ["All operations permitted.", "Run the daily toolbox talk and PPE check.",
                       "Continue standard concrete curing schedule."]),
    "High Wind": ("High", ["Suspend crane and hoist operations above ~38 km/h.",
                           "Secure loose sheets, formwork and scaffold netting.",
                           "No work at height without double lanyard tie-off."]),
    "Heavy Rain": ("Critical", ["Stop concreting; cover fresh pours with tarpaulin.",
                                "Isolate temporary electrical boards in wet zones.",
                                "Check excavation sides for collapse risk; pump out water."]),
    "Extreme Heat": ("Medium", ["Shift concrete pours to early morning; use chilled water.",
                                "Mandatory water and shade breaks every hour.",
                                "Extend curing; keep surfaces wet to prevent cracks."]),
}

with tab4:
    st.subheader("Daily Site Assistant & Weather-Aware Advisory")
    weather = st.radio("Today's weather", list(WEATHER_ALERTS), horizontal=True)
    level, actions = WEATHER_ALERTS[weather]
    st.markdown(f"**Risk level:** {pill(level)}", unsafe_allow_html=True)
    for a in actions:
        st.markdown(f"- {a}")

    st.divider()
    st.markdown("#### 💬 Construction engineering assistant")
    if "chat" not in st.session_state:
        st.session_state.chat = []
    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    question = st.chat_input("Ask about curing, materials, safety, cracks...")
    if question:
        st.session_state.chat.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        system = (f"You are a helpful senior site engineer in India. Today's weather: {weather}. "
                  "Answer practically and briefly, citing IS codes or OSHA where useful.")
        try:
            answer = gemini_chat(st.session_state.chat, system) if LIVE else demo_chat(question, weather)
        except Exception:
            answer = demo_chat(question, weather)
        st.session_state.chat.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)

st.divider()
st.caption("Skill Wallet Capstone Project 1 · Built with Streamlit · Estimates are indicative only.")
