# 🍽️ Food Verse AI – Multimodal Food Intelligence Agent
Skill Wallet Capstone Project 3

Five AI-powered food modules in one Streamlit app, with no database and no setup beyond an optional API key.

| Tab | Epic | What it does |
|---|---|---|
| 🥗 Nutrition Audit | 2 | Meal photo → dish, cuisine, ingredients, portion, macros, fiber, allergens, health score |
| 🥬 Freshness | 3 | Produce/meat/dairy photo → Fresh / Moderately Fresh / Spoiled + safety advice + storage tips |
| 🧼 Kitchen Hygiene | 4 | Kitchen image/video → hygiene indicators, compliance %, violation log |
| 👩‍🍳 Recipes | 4 | Pantry list → recipes with time, difficulty, steps, substitutions |
| 📅 Meal Planner | 5 | BMR/TDEE targets → 7-day plan + grocery list |

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Modes
* **Demo mode** (no key): realistic sample outputs so every screen works offline.
* **Live mode**: paste a **Gemini** or **Groq** API key in the sidebar (or set `GEMINI_API_KEY` / `GROQ_API_KEY` in Streamlit secrets).
  * Gemini default model: `gemini-2.5-flash` (supports images and short video)
  * Groq default model: `meta-llama/llama-4-scout-17b-16e-instruct` (images only)
  * Model names are editable in the sidebar if a provider renames or retires one.

## Deploy on Streamlit Community Cloud
1. Push `app.py` and `requirements.txt` to a GitHub repo.
2. On share.streamlit.io choose **New app**, select the repo and set the main file to `app.py`.
3. (Optional) Under **Advanced settings → Secrets** add:
   ```toml
   GEMINI_API_KEY = "your-key"
   ```

## Notes
* The Meal Planner is rule-based (Mifflin-St Jeor BMR × activity factor, goal adjustment) and works fully offline.
* Outputs are informational only, not medical, dietary or regulatory advice. Visual freshness checks can't guarantee food safety.
