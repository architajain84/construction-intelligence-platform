# 🏗️ Multimodal Construction Intelligence Platform

**Skill Wallet Capstone Project 1** · by Archita Jain ([@architajain84](https://github.com/architajain84))

An AI-powered decision-support web app for construction professionals. It analyses site photos for worker safety, diagnoses concrete cracks, estimates project cost and real-estate value in INR, and gives weather-aware daily site advice through a chat assistant.

## ✨ Features

| Module | What it does |
|---|---|
| 🦺 **Site Safety & PPE Audit** | Upload a site photo → safety score (0–100), PPE checklist (helmet, vest, boots, harness, gloves, eye protection), hazards with severity and remedies, downloadable JSON report |
| 🧱 **Crack Diagnostics** | Upload a wall/concrete photo → defect type (Hairline, Thermal, Settlement, Spalling, Structural Shear), estimated width in mm, urgency, root causes and repair methods |
| 📐 **Planning & Valuation** | Structure type, area, city tier and completion % → total budget, capital spent, market value (₹ Lakh/Crore), cement and steel quantities, AI concept render |
| 🌦️ **Site Assistant & Weather** | Weather-based safety alerts (wind, rain, heat) plus a multi-turn construction engineering chatbot |

## 🧠 How the AI works

- **Live AI mode:** paste a free Google Gemini API key in the sidebar. Photos are sent to the Gemini multimodal model, which returns strict JSON (data contracts) that the app renders.
- **Demo mode:** with no key, a built-in deterministic engine produces realistic analysis so the app always works.

## 🛠️ Tech stack

Python 3.9+ · Streamlit · Google Gemini (multimodal LLM, REST) · Pillow · Pollinations AI · JSON data contracts · No database (stateless, cloud-ready)

## 🗺️ Epic mapping

| Epic | Where in the app |
|---|---|
| Epic 1: Requirements & environment | `requirements.txt`, sidebar configuration |
| Epic 2: PPE safety & crack analysis | Tabs 1 and 2 |
| Epic 3: Planning & valuation | Tab 3 |
| Epic 4: Multimodal AI & JSON contracts | `gemini_json()` + JSON viewers/downloads |
| Epic 5: Assistant & weather advisory | Tab 4 |
| Epic 6: Interface, testing, deployment | Streamlit UI, error handling, Streamlit Cloud |

## 🚀 Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## ☁️ Live demo

Deployed free on Streamlit Community Cloud.

## ⚠️ Known limitations

- Demo mode results are simulated, not real computer-vision detections.
- Cost and valuation figures are indicative averages, not professional estimates.
- Live mode needs an internet connection and a valid Gemini API key.
