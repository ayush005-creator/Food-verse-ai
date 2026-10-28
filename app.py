"""
Food Verse AI - Multimodal Food Intelligence Agent
Skill Wallet Capstone Project 3

Single-file Streamlit app. No database. Works in two modes:
  * Live mode  : Gemini or Groq (REST, via `requests`) with the user's API key
  * Demo mode  : deterministic sample outputs, no key needed

Run:  streamlit run app.py
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import re
from collections import defaultdict
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from PIL import Image, ImageOps

# --------------------------------------------------------------------------- #
# Page config + styling
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Food Verse AI",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
:root { --fv-orange:#ff7a18; --fv-pink:#ff3d71; --fv-green:#16a34a; --fv-ink:#1f2937; }
.fv-hero{background:linear-gradient(120deg,#ff7a18 0%,#ff3d71 55%,#7c3aed 100%);
  padding:1.6rem 1.8rem;border-radius:20px;color:#fff;margin-bottom:1rem;
  box-shadow:0 10px 30px rgba(255,61,113,.25);}
.fv-hero h1{margin:0;font-size:2.0rem;color:#fff;}
.fv-hero p{margin:.3rem 0 0 0;opacity:.95;}
.fv-card{background:rgba(127,127,127,.08);border:1px solid rgba(127,127,127,.18);
  border-radius:16px;padding:1rem 1.2rem;margin-bottom:.8rem;}
.fv-badge{display:inline-block;padding:.25rem .8rem;border-radius:999px;font-weight:600;
  font-size:.85rem;margin:.15rem .25rem .15rem 0;color:#fff;}
.b-green{background:#16a34a}.b-amber{background:#f59e0b}.b-red{background:#dc2626}
.b-blue{background:#2563eb}.b-purple{background:#7c3aed}.b-grey{background:#6b7280}
.fv-metric{text-align:center;padding:.8rem;border-radius:14px;
  background:linear-gradient(135deg,rgba(255,122,24,.12),rgba(255,61,113,.12));
  border:1px solid rgba(255,122,24,.25);}
.fv-metric .v{font-size:1.6rem;font-weight:800;}
.fv-metric .l{font-size:.8rem;opacity:.75;text-transform:uppercase;letter-spacing:.05em;}
.stTabs [data-baseweb="tab-list"]{gap:.4rem;}
.stTabs [data-baseweb="tab"]{border-radius:12px 12px 0 0;padding:.5rem 1rem;font-weight:600;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Generic helpers
# --------------------------------------------------------------------------- #
class LLMError(Exception):
    """Raised when a live model call fails."""


def num(x: Any, default: float = 0.0) -> float:
    """Safely coerce model output to float."""
    try:
        if x is None:
            return default
        if isinstance(x, (int, float)):
            return float(x)
        m = re.search(r"-?\d+(\.\d+)?", str(x))
        return float(m.group()) if m else default
    except Exception:
        return default


def badge(text: str, color: str = "grey") -> str:
    return f'<span class="fv-badge b-{color}">{text}</span>'


def metric_card(label: str, value: str) -> str:
    return f'<div class="fv-metric"><div class="v">{value}</div><div class="l">{label}</div></div>'


def score_color(score: float) -> str:
    return "green" if score >= 70 else "amber" if score >= 45 else "red"


def digest(data: bytes) -> int:
    return int(hashlib.md5(data).hexdigest(), 16)


def prepare_image(uploaded, max_side: int = 1024) -> tuple[Image.Image, bytes, str]:
    """Open an upload, fix orientation, downscale, return (PIL, jpeg_bytes, mime)."""
    img = Image.open(uploaded)
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return img, buf.getvalue(), "image/jpeg"


def parse_json(text: str) -> dict:
    """Parse JSON from a model response, tolerating code fences and prose."""
    if not text:
        raise LLMError("Empty response from model.")
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    raise LLMError("Model did not return valid JSON. Try again or switch model.")


# --------------------------------------------------------------------------- #
# LLM layer (Gemini / Groq over REST)
# --------------------------------------------------------------------------- #
GEMINI_DEFAULT = "gemini-2.5-flash"
GROQ_DEFAULT = "meta-llama/llama-4-scout-17b-16e-instruct"


def call_llm(provider: str, api_key: str, model: str, prompt: str,
             media: list[tuple[bytes, str]] | None = None, timeout: int = 90) -> dict:
    """Send prompt (+ optional media) and return parsed JSON dict."""
    media = media or []
    try:
        if provider == "Gemini":
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
            parts: list[dict] = [{"text": prompt}]
            for data, mime in media:
                parts.append({"inline_data": {"mime_type": mime,
                                              "data": base64.b64encode(data).decode()}})
            body = {"contents": [{"parts": parts}],
                    "generationConfig": {"temperature": 0.2,
                                         "responseMimeType": "application/json"}}
            r = requests.post(url, params={"key": api_key}, json=body, timeout=timeout)
            if r.status_code != 200:
                raise LLMError(f"Gemini error {r.status_code}: {r.text[:300]}")
            try:
                text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
            except (KeyError, IndexError, TypeError):
                raise LLMError("Gemini returned no content (possibly blocked by safety filters).")
            return parse_json(text)

        if provider == "Groq":
            content: list[dict] = [{"type": "text", "text": prompt}]
            for data, mime in media:
                if not mime.startswith("image/"):
                    raise LLMError("Groq vision accepts images only.")
                uri = f"data:{mime};base64,{base64.b64encode(data).decode()}"
                content.append({"type": "image_url", "image_url": {"url": uri}})
            body = {"model": model,
                    "messages": [{"role": "user", "content": content}],
                    "temperature": 0.2,
                    "response_format": {"type": "json_object"}}
            r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                              headers={"Authorization": f"Bearer {api_key}"},
                              json=body, timeout=timeout)
            if r.status_code != 200:
                raise LLMError(f"Groq error {r.status_code}: {r.text[:300]}")
            try:
                text = r.json()["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                raise LLMError("Groq returned an unexpected response.")
            return parse_json(text)

        raise LLMError(f"Unknown provider: {provider}")
    except requests.Timeout:
        raise LLMError("The request timed out. Please try again.")
    except requests.RequestException as e:
        raise LLMError(f"Network error: {e}")


def run_task(prompt: str, media: list[tuple[bytes, str]] | None, demo_fn, seed: bytes | str = b"") -> tuple[dict, bool]:
    """Run live if configured, else (or on failure) demo. Returns (result, is_demo)."""
    cfg = st.session_state.cfg
    seed_b = seed if isinstance(seed, bytes) else seed.encode()
    if cfg["live"]:
        try:
            with st.spinner(f"Asking {cfg['provider']} ({cfg['model']})…"):
                return call_llm(cfg["provider"], cfg["key"], cfg["model"], prompt, media), False
        except LLMError as e:
            st.error(f"Live analysis failed: {e}")
            st.info("Showing demonstration output instead so you can keep exploring.")
        except Exception as e:  # noqa: BLE001
            st.error(f"Unexpected error: {e}")
    return demo_fn(seed_b), True


# --------------------------------------------------------------------------- #
# Sidebar configuration
# --------------------------------------------------------------------------- #
def secret_key(name: str) -> str:
    try:
        return st.secrets.get(name, "")
    except Exception:  # noqa: BLE001
        return ""


def sidebar() -> None:
    with st.sidebar:
        st.markdown("## 🍽️ Food Verse AI")
        st.caption("Multimodal Food Intelligence Agent")
        provider = st.selectbox("AI provider", ["Gemini", "Groq"])
        default_model = GEMINI_DEFAULT if provider == "Gemini" else GROQ_DEFAULT
        model = st.text_input("Model", value=default_model,
                              help="Must be a vision-capable model for image tabs.")
        pre = secret_key("GEMINI_API_KEY" if provider == "Gemini" else "GROQ_API_KEY")
        key = st.text_input(f"{provider} API key", type="password", value=pre,
                            placeholder="Paste key or leave empty for demo mode")
        force_demo = st.toggle("Force demo mode", value=False)
        live = bool(key.strip()) and not force_demo
        st.session_state.cfg = {"provider": provider, "model": model.strip() or default_model,
                                "key": key.strip(), "live": live}
        if live:
            st.success(f"Live mode · {provider}")
        else:
            st.warning("Demo mode · sample outputs (no API key)")
        st.divider()
        st.markdown("**Intelligence pillars**")
        st.markdown("1. 🥗 Recognition & Nutrition\n2. 🥬 Freshness & Spoilage\n"
                    "3. 🧼 Kitchen Hygiene\n4. 👩‍🍳 Recipe Assistant\n5. 📅 Meal Planner")
        st.caption("Keys are held only in this browser session and never stored.")
        st.caption("Informational only — not medical or regulatory advice.")


# --------------------------------------------------------------------------- #
# Tab 1 - Food recognition & nutrition audit
# --------------------------------------------------------------------------- #
NUTRITION_PROMPT = """You are an expert food scientist and registered dietitian.
Analyze the meal photo. Identify the dish, cuisine, visible ingredients and portion size,
then estimate nutrition for the WHOLE portion shown. Respond with ONLY valid JSON:
{
 "dish_name": str, "cuisine": str, "confidence": number 0-100,
 "portion_description": str, "estimated_weight_g": number,
 "ingredients": [{"name": str, "estimated_grams": number}],
 "nutrition": {"calories_kcal": number, "protein_g": number, "carbs_g": number,
               "fats_g": number, "fiber_g": number},
 "allergens": [str],
 "health_score": integer 1-100,
 "health_score_rationale": str,
 "health_tips": [str]
}
Health score weighs nutrient balance, fiber, processing level, sugar/sodium/saturated fat and portion size."""

DEMO_DISHES = [
    {"dish_name": "Chicken Biryani with Raita", "cuisine": "Indian (Hyderabadi)", "confidence": 88,
     "portion_description": "One large plate (~1.5 cups rice + 2 chicken pieces) with a small bowl of raita",
     "estimated_weight_g": 450,
     "ingredients": [{"name": "Basmati rice", "estimated_grams": 220}, {"name": "Chicken", "estimated_grams": 120},
                     {"name": "Fried onion", "estimated_grams": 25}, {"name": "Ghee / oil", "estimated_grams": 20},
                     {"name": "Yogurt (raita)", "estimated_grams": 60}],
     "nutrition": {"calories_kcal": 720, "protein_g": 34, "carbs_g": 88, "fats_g": 24, "fiber_g": 4},
     "allergens": ["Dairy"], "health_score": 58,
     "health_score_rationale": "Good protein, but refined rice and ghee make it energy-dense with little fiber.",
     "health_tips": ["Add a large cucumber-tomato salad for fiber.", "Prefer a smaller rice portion and extra chicken."]},
    {"dish_name": "Greek Salad with Grilled Chicken", "cuisine": "Mediterranean", "confidence": 91,
     "portion_description": "Large bowl, roughly 350 g with ~100 g grilled chicken",
     "estimated_weight_g": 350,
     "ingredients": [{"name": "Cucumber", "estimated_grams": 90}, {"name": "Tomato", "estimated_grams": 80},
                     {"name": "Feta cheese", "estimated_grams": 40}, {"name": "Grilled chicken", "estimated_grams": 100},
                     {"name": "Olives & olive oil", "estimated_grams": 30}],
     "nutrition": {"calories_kcal": 410, "protein_g": 36, "carbs_g": 12, "fats_g": 24, "fiber_g": 4},
     "allergens": ["Dairy"], "health_score": 88,
     "health_score_rationale": "High protein, low refined carbs and plenty of vegetables; feta adds sodium.",
     "health_tips": ["Go light on feta to reduce sodium.", "Add chickpeas for extra fiber."]},
    {"dish_name": "Masala Dosa with Sambar", "cuisine": "South Indian", "confidence": 90,
     "portion_description": "One large dosa with potato filling, sambar and coconut chutney",
     "estimated_weight_g": 380,
     "ingredients": [{"name": "Rice-lentil batter", "estimated_grams": 120}, {"name": "Potato masala", "estimated_grams": 150},
                     {"name": "Sambar", "estimated_grams": 80}, {"name": "Coconut chutney", "estimated_grams": 30}],
     "nutrition": {"calories_kcal": 480, "protein_g": 11, "carbs_g": 72, "fats_g": 17, "fiber_g": 6},
     "allergens": ["Mustard", "Coconut"], "health_score": 66,
     "health_score_rationale": "Fermented batter is gut-friendly and sambar adds lentils, but carbs dominate.",
     "health_tips": ["Ask for extra sambar and less potato filling.", "Pair with a protein side like curd or paneer."]},
    {"dish_name": "Margherita Pizza (2 slices)", "cuisine": "Italian", "confidence": 93,
     "portion_description": "Two medium slices, thin crust",
     "estimated_weight_g": 220,
     "ingredients": [{"name": "Pizza dough (refined flour)", "estimated_grams": 110}, {"name": "Mozzarella", "estimated_grams": 70},
                     {"name": "Tomato sauce", "estimated_grams": 35}, {"name": "Basil & olive oil", "estimated_grams": 5}],
     "nutrition": {"calories_kcal": 540, "protein_g": 22, "carbs_g": 62, "fats_g": 21, "fiber_g": 3},
     "allergens": ["Gluten", "Dairy"], "health_score": 49,
     "health_score_rationale": "Moderate protein from cheese; refined flour, low fiber and high saturated fat.",
     "health_tips": ["Choose whole-wheat crust and add vegetable toppings.", "Pair with a green salad."]},
]


def demo_nutrition(seed: bytes) -> dict:
    return DEMO_DISHES[digest(seed) % len(DEMO_DISHES)]


def render_nutrition(res: dict, is_demo: bool) -> None:
    n = res.get("nutrition", {})
    kcal, pro, carb, fat, fib = (num(n.get(k)) for k in
                                 ("calories_kcal", "protein_g", "carbs_g", "fats_g", "fiber_g"))
    score = int(max(1, min(100, num(res.get("health_score"), 50))))
    color = score_color(score)
    label = {"green": "Excellent choice", "amber": "Moderate", "red": "Indulgent"}[color]

    st.markdown(f"### {res.get('dish_name', 'Unknown dish')}")
    head = badge(res.get("cuisine", "Cuisine n/a"), "purple") + badge(f"Confidence {num(res.get('confidence')):.0f}%", "blue")
    head += badge(f"Health {score}/100 · {label}", color)
    if is_demo:
        head += badge("DEMO DATA", "grey")
    st.markdown(head, unsafe_allow_html=True)
    st.caption(f"Portion: {res.get('portion_description', 'n/a')} · ~{num(res.get('estimated_weight_g')):.0f} g")

    cols = st.columns(5)
    for c, (l, v) in zip(cols, [("Calories", f"{kcal:.0f} kcal"), ("Protein", f"{pro:.0f} g"),
                                ("Carbs", f"{carb:.0f} g"), ("Fats", f"{fat:.0f} g"), ("Fiber", f"{fib:.0f} g")]):
        c.markdown(metric_card(l, v), unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1.1, 1, 1])
    with c1:
        macro_kcal = {"Protein": pro * 4, "Carbs": carb * 4, "Fats": fat * 9}
        fig = px.pie(names=list(macro_kcal), values=list(macro_kcal.values()), hole=0.55,
                     color_discrete_sequence=["#16a34a", "#f59e0b", "#ef4444"], title="Calories by macro")
        fig.update_layout(margin=dict(t=50, b=10, l=10, r=10), height=300)
        st.plotly_chart(fig, width="stretch")
    with c2:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=score, title={"text": "Health score"},
            gauge={"axis": {"range": [0, 100]},
                   "bar": {"color": {"green": "#16a34a", "amber": "#f59e0b", "red": "#dc2626"}[color]},
                   "steps": [{"range": [0, 45], "color": "rgba(220,38,38,.15)"},
                             {"range": [45, 70], "color": "rgba(245,158,11,.15)"},
                             {"range": [70, 100], "color": "rgba(22,163,74,.15)"}]}))
        gauge.update_layout(margin=dict(t=60, b=10, l=20, r=20), height=300)
        st.plotly_chart(gauge, width="stretch")
    with c3:
        ings = res.get("ingredients", []) or []
        if ings:
            df = pd.DataFrame(ings)
            if {"name", "estimated_grams"} <= set(df.columns):
                df["estimated_grams"] = df["estimated_grams"].map(num)
                fig = px.bar(df, x="estimated_grams", y="name", orientation="h",
                             title="Ingredient weight (g)", color_discrete_sequence=["#ff7a18"])
                fig.update_layout(margin=dict(t=50, b=10, l=10, r=10), height=300,
                                  yaxis_title=None, xaxis_title=None)
                st.plotly_chart(fig, width="stretch")

    allergens = res.get("allergens") or []
    st.markdown("**Allergen alerts**")
    if allergens:
        st.markdown("".join(badge(f"⚠ {a}", "red") for a in allergens), unsafe_allow_html=True)
    else:
        st.markdown(badge("No major allergens detected", "green"), unsafe_allow_html=True)

    st.markdown(f"**Why this score:** {res.get('health_score_rationale', '')}")
    tips = res.get("health_tips") or []
    if tips:
        st.markdown("**Smart swaps & tips**")
        for t in tips:
            st.markdown(f"- {t}")


def tab_nutrition() -> None:
    st.subheader("🥗 Food Recognition & Nutrition Audit")
    up = st.file_uploader("Upload a photo of your meal", type=["jpg", "jpeg", "png", "webp"], key="nut_up")
    if not up:
        st.info("Upload a meal photo to identify the dish and audit its nutrition.")
        return
    try:
        img, data, mime = prepare_image(up)
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not read this image: {e}")
        return
    left, right = st.columns([1, 2])
    left.image(img, width="stretch")
    with left:
        go_btn = st.button("🔍 Analyze meal", type="primary", key="nut_btn", width="stretch")
    if go_btn:
        res, is_demo = run_task(NUTRITION_PROMPT, [(data, mime)], demo_nutrition, data)
        st.session_state["nut_res"] = (res, is_demo, hashlib.md5(data).hexdigest())
    saved = st.session_state.get("nut_res")
    if saved and saved[2] == hashlib.md5(data).hexdigest():
        with right:
            render_nutrition(saved[0], saved[1])


# --------------------------------------------------------------------------- #
# Tab 2 - Freshness & spoilage
# --------------------------------------------------------------------------- #
FRESH_PROMPT = """You are a food safety scientist. Inspect the photo of a perishable item
(fruit, vegetable, meat, fish, dairy or similar). Evaluate surface texture, discoloration,
wrinkling/shrivelling and visible mold or slime. Respond with ONLY valid JSON:
{
 "item_name": str, "category": str,
 "observations": {"surface_texture": str, "discoloration": str, "wrinkling": str, "mold": str},
 "freshness_stage": "Fresh" | "Moderately Fresh" | "Spoiled",
 "freshness_score": integer 0-100,
 "safety_advice": "Safe to consume" | "Consume within 24h" | "Discard immediately",
 "estimated_remaining_shelf_life": str,
 "reasoning": str,
 "storage_tips": [str]
}
Be conservative: if mold, slime or foul-looking decay is visible on meat, fish or dairy, choose Spoiled.
Visual inspection cannot guarantee safety; say so in reasoning when uncertain."""

DEMO_FRESH = [
    {"item_name": "Tomatoes", "category": "Vegetable",
     "observations": {"surface_texture": "Firm, smooth and glossy skin", "discoloration": "Uniform red, no dark patches",
                      "wrinkling": "None", "mold": "None visible"},
     "freshness_stage": "Fresh", "freshness_score": 92, "safety_advice": "Safe to consume",
     "estimated_remaining_shelf_life": "5-7 days at room temperature",
     "reasoning": "Taut skin, even colour and no soft spots indicate peak freshness.",
     "storage_tips": ["Store stem-side down at room temperature away from sunlight.",
                      "Refrigerate only once fully ripe to extend life 3-4 days.", "Keep away from ethylene producers like bananas."]},
    {"item_name": "Bananas", "category": "Fruit",
     "observations": {"surface_texture": "Soft with brown freckling", "discoloration": "Heavy brown spots over most of peel",
                      "wrinkling": "Mild peel wrinkling near the stem", "mold": "None visible"},
     "freshness_stage": "Moderately Fresh", "freshness_score": 55, "safety_advice": "Consume within 24h",
     "estimated_remaining_shelf_life": "1-2 days (ideal for baking/smoothies)",
     "reasoning": "Starches have converted to sugar; flesh is likely soft but still safe with no mold.",
     "storage_tips": ["Peel and freeze slices for smoothies or banana bread.", "Separate from other fruit to slow ripening.",
                      "Wrap the stems in cling film."]},
    {"item_name": "Strawberries", "category": "Fruit",
     "observations": {"surface_texture": "Mushy, collapsing with visible moisture", "discoloration": "Dark patches and dull colour",
                      "wrinkling": "Shrivelled calyx", "mold": "Grey fuzzy mold present"},
     "freshness_stage": "Spoiled", "freshness_score": 8, "safety_advice": "Discard immediately",
     "estimated_remaining_shelf_life": "None",
     "reasoning": "Visible mold on soft produce spreads invisibly through the fruit; do not cut around it.",
     "storage_tips": ["Discard the whole punnet and wash the container.", "Next time store unwashed in a paper-towel-lined box in the fridge.",
                      "Wash berries only right before eating."]},
]


def demo_fresh(seed: bytes) -> dict:
    return DEMO_FRESH[digest(seed) % len(DEMO_FRESH)]


def normalise_fresh(res: dict) -> dict:
    stage_map = {"fresh": "Fresh", "moderately fresh": "Moderately Fresh", "spoiled": "Spoiled"}
    stage = stage_map.get(str(res.get("freshness_stage", "")).strip().lower(), "Moderately Fresh")
    res["freshness_stage"] = stage
    default_adv = {"Fresh": "Safe to consume", "Moderately Fresh": "Consume within 24h", "Spoiled": "Discard immediately"}
    adv = str(res.get("safety_advice", "")).strip()
    if adv not in default_adv.values():
        adv = default_adv[stage]
    res["safety_advice"] = adv
    return res


def render_fresh(res: dict, is_demo: bool) -> None:
    res = normalise_fresh(res)
    stage, adv = res["freshness_stage"], res["safety_advice"]
    col = {"Fresh": "green", "Moderately Fresh": "amber", "Spoiled": "red"}[stage]
    icon = {"Fresh": "✅", "Moderately Fresh": "⚠️", "Spoiled": "🚫"}[stage]
    st.markdown(f"### {res.get('item_name', 'Item')} · {res.get('category', '')}")
    head = badge(f"{icon} {stage}", col) + badge(adv, col)
    if is_demo:
        head += badge("DEMO DATA", "grey")
    st.markdown(head, unsafe_allow_html=True)

    score = num(res.get("freshness_score"), 50)
    c1, c2 = st.columns([1, 1.4])
    with c1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=score, title={"text": "Freshness index"},
            gauge={"axis": {"range": [0, 100]},
                   "bar": {"color": {"green": "#16a34a", "amber": "#f59e0b", "red": "#dc2626"}[col]},
                   "steps": [{"range": [0, 35], "color": "rgba(220,38,38,.15)"},
                             {"range": [35, 70], "color": "rgba(245,158,11,.15)"},
                             {"range": [70, 100], "color": "rgba(22,163,74,.15)"}]}))
        fig.update_layout(height=270, margin=dict(t=60, b=10, l=20, r=20))
        st.plotly_chart(fig, width="stretch")
        st.markdown(f"**Estimated shelf life:** {res.get('estimated_remaining_shelf_life', 'n/a')}")
    with c2:
        obs = res.get("observations", {}) or {}
        rows = [("Surface texture", obs.get("surface_texture")), ("Discoloration", obs.get("discoloration")),
                ("Wrinkling", obs.get("wrinkling")), ("Mold / slime", obs.get("mold"))]
        st.table(pd.DataFrame(rows, columns=["Indicator", "Observation"]).set_index("Indicator"))
        st.markdown(f"**Assessment:** {res.get('reasoning', '')}")

    if adv == "Discard immediately":
        st.error("🚫 Discard immediately. Do not taste-test or cut away affected areas.")
    elif adv == "Consume within 24h":
        st.warning("⏱️ Consume within 24 hours, or cook / freeze it today.")
    else:
        st.success("✅ Safe to consume.")
    tips = res.get("storage_tips") or []
    if tips:
        st.markdown("**Preservation & storage tips**")
        for t in tips:
            st.markdown(f"- {t}")
    st.caption("Visual checks are indicative only. When in doubt, throw it out.")


def tab_freshness() -> None:
    st.subheader("🥬 Food Freshness & Spoilage Diagnostic")
    up = st.file_uploader("Upload a close-up of a fruit, vegetable, meat or dairy item",
                          type=["jpg", "jpeg", "png", "webp"], key="fr_up")
    if not up:
        st.info("Upload a clear, well-lit close-up for the best diagnosis.")
        return
    try:
        img, data, mime = prepare_image(up)
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not read this image: {e}")
        return
    left, right = st.columns([1, 2])
    left.image(img, width="stretch")
    with left:
        go_btn = st.button("🧪 Diagnose freshness", type="primary", key="fr_btn", width="stretch")
    if go_btn:
        res, is_demo = run_task(FRESH_PROMPT, [(data, mime)], demo_fresh, data)
        st.session_state["fr_res"] = (res, is_demo, hashlib.md5(data).hexdigest())
    saved = st.session_state.get("fr_res")
    if saved and saved[2] == hashlib.md5(data).hexdigest():
        with right:
            render_fresh(saved[0], saved[1])


# --------------------------------------------------------------------------- #
# Tab 3 - Kitchen hygiene
# --------------------------------------------------------------------------- #
HYGIENE_PROMPT = """You are a certified food safety (HACCP) auditor. Inspect this commercial kitchen image/video.
Report on each indicator with status "compliant", "violation" or "not_visible":
 - head_covering (chef hat / hairnet / cap on all visible staff)
 - gloves (gloves worn where handling ready-to-eat food)
 - sanitization (clean surfaces, tidy stations, sanitizer, no spills or clutter)
 - cross_contamination (raw poultry/meat/fish near vegetables or ready-to-eat food, shared boards/knives)
 - uniform_apron (clean apron / uniform)
Respond with ONLY valid JSON:
{
 "indicators": {
   "head_covering": {"status": str, "note": str},
   "gloves": {"status": str, "note": str},
   "sanitization": {"status": str, "note": str},
   "cross_contamination": {"status": str, "note": str},
   "uniform_apron": {"status": str, "note": str}
 },
 "violations": [{"item": str, "severity": "critical"|"major"|"minor", "description": str, "corrective_action": str}],
 "staff_visible": integer,
 "overall_notes": str
}
Never invent violations that are not visible; use "not_visible" when something cannot be assessed."""

HYGIENE_WEIGHTS = {"head_covering": 20, "gloves": 25, "sanitization": 25,
                   "cross_contamination": 30, "uniform_apron": 10}
HYGIENE_LABELS = {"head_covering": "Head covering", "gloves": "Glove usage", "sanitization": "Sanitization",
                  "cross_contamination": "Cross-contamination control", "uniform_apron": "Uniform / apron"}

DEMO_HYGIENE = [
    {"indicators": {"head_covering": {"status": "compliant", "note": "All 3 visible staff wear hairnets or caps."},
                    "gloves": {"status": "violation", "note": "One cook handles cooked chicken bare-handed."},
                    "sanitization": {"status": "compliant", "note": "Stainless surfaces are clean and uncluttered."},
                    "cross_contamination": {"status": "violation", "note": "Raw poultry tray sits beside chopped salad vegetables."},
                    "uniform_apron": {"status": "compliant", "note": "Clean aprons observed."}},
     "violations": [{"item": "Raw poultry next to vegetables", "severity": "critical",
                     "description": "Raw chicken stored on the same bench as ready-to-eat vegetables.",
                     "corrective_action": "Separate raw and RTE zones; use colour-coded boards; store raw poultry on the lowest shelf."},
                    {"item": "Bare-hand contact", "severity": "major",
                     "description": "Cooked food handled without gloves or utensils.",
                     "corrective_action": "Use tongs or single-use gloves; wash hands before changing tasks."}],
     "staff_visible": 3, "overall_notes": "Good personal grooming but critical separation failure needs immediate action."},
    {"indicators": {"head_covering": {"status": "violation", "note": "Two staff without any head covering."},
                    "gloves": {"status": "compliant", "note": "Gloves worn at the prep station."},
                    "sanitization": {"status": "violation", "note": "Visible spills and food debris on the counter."},
                    "cross_contamination": {"status": "compliant", "note": "Raw and cooked items separated."},
                    "uniform_apron": {"status": "not_visible", "note": "Cannot be assessed."}},
     "violations": [{"item": "Missing head covering", "severity": "major",
                     "description": "Staff preparing food without hairnet or cap.",
                     "corrective_action": "Mandate hair restraints at the kitchen entrance."},
                    {"item": "Unsanitized work surface", "severity": "minor",
                     "description": "Spills and debris on the prep counter.",
                     "corrective_action": "Clean and sanitize surfaces between tasks; follow a clean-as-you-go routine."}],
     "staff_visible": 2, "overall_notes": "Cross-contamination control is good; hygiene housekeeping needs improvement."},
]


def demo_hygiene(seed: bytes) -> dict:
    return DEMO_HYGIENE[digest(seed) % len(DEMO_HYGIENE)]


def hygiene_score(res: dict) -> tuple[float, pd.DataFrame]:
    rows, earned, possible = [], 0.0, 0.0
    ind = res.get("indicators", {}) or {}
    for key, w in HYGIENE_WEIGHTS.items():
        item = ind.get(key, {}) or {}
        status = str(item.get("status", "not_visible")).lower()
        if status == "compliant":
            earned += w
            possible += w
        elif status == "violation":
            possible += w
        rows.append({"Indicator": HYGIENE_LABELS[key], "Status": status.replace("_", " ").title(),
                     "Weight": w, "Note": item.get("note", "")})
    # Extra severity penalty for critical items beyond what indicators capture
    score = (earned / possible * 100) if possible else 0.0
    crit = sum(1 for v in res.get("violations", []) or [] if str(v.get("severity", "")).lower() == "critical")
    score = max(0.0, score - 5 * max(0, crit - 1))
    return round(score, 1), pd.DataFrame(rows)


def render_hygiene(res: dict, is_demo: bool) -> None:
    score, df = hygiene_score(res)
    col = score_color(score)
    verdict = {"green": "Compliant", "amber": "Needs improvement", "red": "Non-compliant"}[col]
    head = badge(f"Compliance {score:.0f}% · {verdict}", col)
    if is_demo:
        head += badge("DEMO DATA", "grey")
    st.markdown(head, unsafe_allow_html=True)
    st.caption(f"Staff visible: {res.get('staff_visible', 'n/a')} · {res.get('overall_notes', '')}")

    c1, c2 = st.columns([1, 1.3])
    with c1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=score, number={"suffix": "%"}, title={"text": "Hygiene compliance"},
            gauge={"axis": {"range": [0, 100]},
                   "bar": {"color": {"green": "#16a34a", "amber": "#f59e0b", "red": "#dc2626"}[col]},
                   "steps": [{"range": [0, 50], "color": "rgba(220,38,38,.15)"},
                             {"range": [50, 75], "color": "rgba(245,158,11,.15)"},
                             {"range": [75, 100], "color": "rgba(22,163,74,.15)"}]}))
        fig.update_layout(height=280, margin=dict(t=60, b=10, l=20, r=20))
        st.plotly_chart(fig, width="stretch")
    with c2:
        cmap = {"Compliant": "#16a34a", "Violation": "#dc2626", "Not Visible": "#9ca3af"}
        fig = px.bar(df, x="Weight", y="Indicator", color="Status", orientation="h",
                     color_discrete_map=cmap, title="Indicator status (bar = weight)")
        fig.update_layout(height=280, margin=dict(t=50, b=10, l=10, r=10), yaxis_title=None)
        st.plotly_chart(fig, width="stretch")

    st.markdown("**Indicator checklist**")
    st.dataframe(df[["Indicator", "Status", "Note"]], width="stretch", hide_index=True)

    viol = res.get("violations", []) or []
    st.markdown("**Violation log**")
    if viol:
        vdf = pd.DataFrame(viol).rename(columns={"item": "Violation", "severity": "Severity",
                                                 "description": "Description", "corrective_action": "Corrective action"})
        order = {"critical": 0, "major": 1, "minor": 2}
        vdf["_o"] = vdf["Severity"].astype(str).str.lower().map(order).fillna(3)
        vdf = vdf.sort_values("_o").drop(columns="_o")
        vdf["Severity"] = vdf["Severity"].astype(str).str.title()
        st.dataframe(vdf, width="stretch", hide_index=True)
        st.download_button("⬇️ Download violation log (CSV)", vdf.to_csv(index=False),
                           file_name="hygiene_violations.csv", mime="text/csv", key="hy_dl")
    else:
        st.success("No violations detected in the supplied media.")
    st.caption("AI screening aid — does not replace an official inspection.")


def tab_hygiene() -> None:
    st.subheader("🧼 Kitchen Hygiene & Food Safety Compliance")
    up = st.file_uploader("Upload a kitchen image or short video",
                          type=["jpg", "jpeg", "png", "webp", "mp4", "mov", "webm"], key="hy_up")
    if not up:
        st.info("Upload a photo (or a short clip with Gemini) of a kitchen to audit.")
        return
    is_video = (up.type or "").startswith("video/") or up.name.lower().endswith((".mp4", ".mov", ".webm"))
    left, right = st.columns([1, 2])
    media: list[tuple[bytes, str]] = []
    if is_video:
        raw = up.getvalue()
        left.video(raw)
        mime = up.type or "video/mp4"
        media = [(raw, mime)]
        seed = raw[:200_000]
    else:
        try:
            img, data, mime = prepare_image(up, 1280)
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not read this image: {e}")
            return
        left.image(img, width="stretch")
        media, seed = [(data, mime)], data
    with left:
        go_btn = st.button("🛡️ Run compliance audit", type="primary", key="hy_btn", width="stretch")
    if go_btn:
        cfg = st.session_state.cfg
        force_demo = False
        if is_video and cfg["live"]:
            if cfg["provider"] != "Gemini":
                st.warning("Groq vision supports images only. Showing demo output — switch to Gemini for video.")
                force_demo = True
            elif len(media[0][0]) > 18 * 1024 * 1024:
                st.warning("Video is larger than ~18 MB inline limit. Showing demo output — upload a shorter clip.")
                force_demo = True
        if force_demo:
            res, is_demo = demo_hygiene(seed), True
        else:
            res, is_demo = run_task(HYGIENE_PROMPT, media, demo_hygiene, seed)
        st.session_state["hy_res"] = (res, is_demo, hashlib.md5(seed).hexdigest())
    saved = st.session_state.get("hy_res")
    if saved and saved[2] == hashlib.md5(seed).hexdigest():
        with right:
            render_hygiene(saved[0], saved[1])


# --------------------------------------------------------------------------- #
# Tab 4 - Recipe assistant
# --------------------------------------------------------------------------- #
RECIPE_PROMPT = """You are a creative chef and nutritionist.
Pantry: {pantry}
Dietary preference: {diet}
Maximum cooking time: {time} minutes
Create {n} distinct, nutritious recipes using mostly the pantry items (basic salt, oil and water are assumed).
Respond with ONLY valid JSON:
{{
 "recipes": [{{
   "name": str, "cuisine": str, "cook_time_min": integer, "difficulty": "Easy"|"Medium"|"Hard",
   "servings": integer, "calories_per_serving": integer, "protein_g_per_serving": integer,
   "ingredients_used": [str], "extra_needed": [str],
   "steps": [str],
   "substitutions": [{{"original": str, "substitute": str, "note": str}}],
   "chef_tip": str
 }}]
}}"""


def demo_recipes(pantry: list[str], diet: str, max_time: int, n: int) -> dict:
    p = [x.title() for x in pantry] or ["Mixed vegetables"]
    low = " ".join(pantry).lower()
    has_egg = "egg" in low and diet not in ("Vegan",)
    has_rice = any(k in low for k in ("rice", "dal", "lentil", "quinoa", "oats"))
    pool: list[dict] = []
    pool.append({"name": "Rainbow Pantry Stir-Fry", "cuisine": "Fusion", "cook_time_min": min(20, max_time),
                 "difficulty": "Easy", "servings": 2, "calories_per_serving": 340, "protein_g_per_serving": 14,
                 "ingredients_used": p[:6], "extra_needed": ["Garlic", "Soy sauce or lemon juice"],
                 "steps": ["Chop all vegetables into evenly sized pieces.", "Heat 1 tbsp oil in a wide pan on high heat.",
                           "Sauté garlic 30 seconds, then add the hardest vegetables first.",
                           "Stir-fry 5-7 minutes until crisp-tender and season with salt, pepper and soy/lemon.",
                           "Serve hot with your choice of grain or protein."],
                 "substitutions": [{"original": "Soy sauce", "substitute": "Lemon juice + pinch of salt", "note": "Keeps it low sodium"}],
                 "chef_tip": "Keep the pan very hot and do not overcrowd it so vegetables sear instead of steam."})
    if has_rice:
        pool.append({"name": "One-Pot Veggie Khichdi", "cuisine": "Indian", "cook_time_min": min(30, max_time),
                     "difficulty": "Easy", "servings": 3, "calories_per_serving": 380, "protein_g_per_serving": 15,
                     "ingredients_used": p[:6], "extra_needed": ["Cumin seeds", "Turmeric", "Ghee or oil"],
                     "steps": ["Rinse the rice and dal together and soak 10 minutes.",
                               "Temper cumin and turmeric in 1 tbsp ghee/oil in a pressure cooker.",
                               "Add chopped vegetables, rice-dal mix and 4 cups water; season with salt.",
                               "Pressure cook 3 whistles (or simmer 25 minutes) until soft.",
                               "Mash lightly and serve with curd or pickle."],
                     "substitutions": [{"original": "Ghee", "substitute": "Olive oil", "note": "Makes it vegan"}],
                     "chef_tip": "Add a splash of hot water if it thickens too much while resting."})
    if has_egg:
        pool.append({"name": "Masala Veggie Scramble", "cuisine": "Indian-style", "cook_time_min": min(12, max_time),
                     "difficulty": "Easy", "servings": 2, "calories_per_serving": 290, "protein_g_per_serving": 19,
                     "ingredients_used": p[:5], "extra_needed": ["Chilli", "Black pepper"],
                     "steps": ["Dice the vegetables finely.", "Sauté in 1 tsp oil for 3 minutes.",
                               "Beat eggs with salt and pepper and pour over the vegetables.",
                               "Stir gently until just set, about 2 minutes.", "Garnish with herbs and serve with toast or roti."],
                     "substitutions": [{"original": "Eggs", "substitute": "Crumbled firm tofu + turmeric", "note": "Vegan scramble"}],
                     "chef_tip": "Pull the pan off the heat while the eggs still look slightly glossy; residual heat finishes them."})
    pool.append({"name": "Warm Pantry Grain Bowl", "cuisine": "Global", "cook_time_min": min(25, max_time),
                 "difficulty": "Medium", "servings": 2, "calories_per_serving": 420, "protein_g_per_serving": 17,
                 "ingredients_used": p[:6], "extra_needed": ["Lemon", "Olive oil", "Herbs"],
                 "steps": ["Cook your base grain according to the packet.", "Roast or pan-sear the vegetables with salt and spices.",
                           "Whisk lemon juice, olive oil and herbs into a dressing.",
                           "Assemble grain, vegetables and any protein in a bowl.", "Drizzle dressing and serve."],
                 "substitutions": [{"original": "Lemon juice", "substitute": "Vinegar + pinch of sugar", "note": "Similar acidity"}],
                 "chef_tip": "Season each component separately for a more flavourful bowl."})
    return {"recipes": pool[:n]}


def render_recipes(res: dict, is_demo: bool) -> None:
    recipes = res.get("recipes", []) or []
    if not recipes:
        st.warning("No recipes were returned. Try adding more ingredients.")
        return
    if is_demo:
        st.markdown(badge("DEMO DATA", "grey"), unsafe_allow_html=True)
    for i, r in enumerate(recipes):
        diff = str(r.get("difficulty", "Easy"))
        dcol = {"easy": "green", "medium": "amber", "hard": "red"}.get(diff.lower(), "grey")
        with st.expander(f"🍲 {r.get('name', 'Recipe')}", expanded=(i == 0)):
            st.markdown(badge(f"⏱ {num(r.get('cook_time_min')):.0f} min", "blue") + badge(diff, dcol)
                        + badge(f"{num(r.get('calories_per_serving')):.0f} kcal/serving", "purple")
                        + badge(f"{num(r.get('protein_g_per_serving')):.0f} g protein", "green")
                        + badge(f"Serves {num(r.get('servings'), 2):.0f}", "grey"), unsafe_allow_html=True)
            a, b = st.columns(2)
            with a:
                st.markdown("**From your pantry**")
                for x in r.get("ingredients_used", []) or []:
                    st.markdown(f"- ✅ {x}")
            with b:
                st.markdown("**Extra needed**")
                extra = r.get("extra_needed", []) or []
                if extra:
                    for x in extra:
                        st.markdown(f"- 🛒 {x}")
                else:
                    st.markdown("_Nothing — you have it all!_")
            st.markdown("**Method**")
            for n_, s in enumerate(r.get("steps", []) or [], 1):
                st.markdown(f"{n_}. {s}")
            subs = r.get("substitutions", []) or []
            if subs:
                st.markdown("**Ingredient substitutions**")
                st.dataframe(pd.DataFrame(subs).rename(columns=str.title), hide_index=True, width="stretch")
            if r.get("chef_tip"):
                st.info(f"👩‍🍳 Chef's tip: {r['chef_tip']}")


def tab_recipes() -> None:
    st.subheader("👩‍🍳 Smart Kitchen Recipe Assistant")
    c1, c2 = st.columns([2, 1])
    with c1:
        pantry_txt = st.text_area("What's in your pantry? (comma or line separated)",
                                  placeholder="eggs, spinach, rice, tomato, paneer, onion, chickpeas…", height=120)
    with c2:
        diet = st.selectbox("Dietary preference", ["Any", "Vegetarian", "Vegan", "Keto", "Diabetic-Friendly", "High-Protein"])
        max_time = st.slider("Max cooking time (min)", 10, 90, 40, 5)
        n = st.slider("Number of recipes", 1, 4, 3)
    pantry = [x.strip() for x in re.split(r"[,\n;]+", pantry_txt) if x.strip()]
    if st.button("✨ Generate recipes", type="primary"):
        if not pantry:
            st.warning("Please list at least one ingredient.")
        else:
            prompt = RECIPE_PROMPT.format(pantry=", ".join(pantry), diet=diet, time=max_time, n=n)
            res, is_demo = run_task(prompt, None, lambda s: demo_recipes(pantry, diet, max_time, n), "|".join(pantry))
            st.session_state["rec_res"] = (res, is_demo)
    saved = st.session_state.get("rec_res")
    if saved:
        render_recipes(*saved)


# --------------------------------------------------------------------------- #
# Tab 5 - Meal planner
# --------------------------------------------------------------------------- #
ACTIVITY = {"Sedentary (desk job)": 1.2, "Lightly active (1-3 days/wk)": 1.375,
            "Moderately active (3-5 days/wk)": 1.55, "Very active (6-7 days/wk)": 1.725,
            "Athlete / physical job": 1.9}
GOALS = {"Fat Loss": -0.20, "Muscle Gain": 0.10, "Maintain Weight": 0.0, "Improve Endurance": 0.05}
SLOT_SPLIT = {"Breakfast": 0.25, "Lunch": 0.30, "Dinner": 0.30, "Snack": 0.15}

V, VG, K, D = "vegetarian", "vegan", "keto", "diabetic"


def meal(name, kcal, p, c, f, tags, ing):
    return {"name": name, "kcal": kcal, "p": p, "c": c, "f": f, "tags": set(tags), "ing": ing}


MEALS: dict[str, list[dict]] = {
    "Breakfast": [
        meal("Moong dal chilla with mint chutney", 320, 18, 42, 8, [VG, V, D], {"Moong dal": 60, "Onion": 30, "Coriander": 10, "Cooking oil": 5, "Mint": 10}),
        meal("Vegetable poha with peanuts", 350, 8, 58, 10, [VG, V], {"Poha (flattened rice)": 70, "Peanuts": 15, "Onion": 30, "Green peas": 30, "Cooking oil": 5}),
        meal("Greek yogurt berry bowl with chia", 300, 22, 32, 9, [V, D], {"Greek yogurt": 200, "Mixed berries": 80, "Chia seeds": 10, "Almonds": 10}),
        meal("Masala omelette (3 eggs) with avocado", 420, 24, 6, 34, [V, K, D], {"Eggs": 150, "Avocado": 70, "Onion": 20, "Spinach": 30, "Butter": 5}),
        meal("Tofu scramble with spinach & avocado", 340, 22, 8, 24, [VG, V, K, D], {"Firm tofu": 150, "Spinach": 50, "Avocado": 60, "Olive oil": 5}),
        meal("Oats with banana & peanut butter", 380, 12, 56, 12, [VG, V], {"Rolled oats": 60, "Banana": 100, "Peanut butter": 15, "Soy milk": 150}),
        meal("Paneer bhurji with bell pepper", 380, 24, 8, 28, [V, K, D], {"Paneer": 120, "Bell pepper": 60, "Onion": 30, "Tomato": 40, "Cooking oil": 5}),
    ],
    "Lunch": [
        meal("Dal tadka, brown rice & cucumber salad", 480, 20, 78, 9, [VG, V], {"Toor dal": 70, "Brown rice": 60, "Cucumber": 80, "Tomato": 40, "Cooking oil": 8}),
        meal("Grilled chicken quinoa bowl", 520, 45, 48, 14, [D], {"Chicken breast": 150, "Quinoa": 60, "Broccoli": 100, "Olive oil": 8}),
        meal("Chickpea & veggie salad with tahini", 440, 18, 50, 18, [VG, V, D], {"Chickpeas (cooked)": 120, "Cucumber": 80, "Tomato": 60, "Tahini": 15, "Lettuce": 40}),
        meal("Paneer tikka salad bowl", 430, 30, 14, 28, [V, K, D], {"Paneer": 130, "Bell pepper": 60, "Lettuce": 60, "Curd": 40, "Olive oil": 5}),
        meal("Grilled chicken salad, olive-oil dressing", 460, 42, 8, 28, [K, D], {"Chicken breast": 150, "Lettuce": 100, "Parmesan": 20, "Olive oil": 15}),
        meal("Rajma with bajra roti", 490, 22, 80, 8, [VG, V, D], {"Rajma (cooked)": 120, "Bajra flour": 60, "Onion": 30, "Tomato": 50}),
        meal("Zucchini noodles with tofu & pesto", 400, 24, 10, 28, [VG, V, K, D], {"Firm tofu": 150, "Zucchini": 200, "Basil": 15, "Olive oil": 15, "Walnuts": 15}),
    ],
    "Dinner": [
        meal("Baked salmon with asparagus", 480, 38, 8, 32, [K, D], {"Salmon fillet": 150, "Asparagus": 120, "Olive oil": 10, "Lemon": 20}),
        meal("Palak paneer with 1 whole-wheat roti", 470, 26, 32, 26, [V, D], {"Paneer": 100, "Spinach": 200, "Whole-wheat flour": 40, "Cooking oil": 8, "Onion": 30}),
        meal("Lentil vegetable soup with multigrain toast", 400, 22, 60, 8, [VG, V, D], {"Red lentils": 70, "Carrot": 60, "Tomato": 50, "Multigrain bread": 40, "Onion": 30}),
        meal("Chicken curry with cauliflower rice", 450, 40, 14, 26, [K, D], {"Chicken thigh": 160, "Cauliflower": 150, "Onion": 30, "Tomato": 40, "Cooking oil": 10}),
        meal("Tofu & vegetable stir-fry with brown rice", 460, 26, 56, 14, [VG, V], {"Firm tofu": 150, "Mixed vegetables": 200, "Brown rice": 60, "Soy sauce": 10, "Cooking oil": 8}),
        meal("Mushroom-tofu coconut curry", 380, 20, 12, 28, [VG, V, K, D], {"Mushrooms": 150, "Firm tofu": 100, "Coconut milk": 60, "Spinach": 60, "Cooking oil": 5}),
        meal("Fish tikka with crunchy salad", 420, 40, 10, 24, [K, D], {"White fish fillet": 170, "Curd": 40, "Lettuce": 60, "Cucumber": 60, "Cooking oil": 8}),
    ],
    "Snack": [
        meal("Roasted chana & apple", 180, 8, 28, 3, [VG, V, D], {"Roasted chana": 30, "Apple": 100}),
        meal("Almond & walnut mix", 190, 6, 5, 17, [VG, V, K, D], {"Almonds": 15, "Walnuts": 15}),
        meal("Greek yogurt with pumpkin seeds", 160, 15, 8, 8, [V, K, D], {"Greek yogurt": 150, "Pumpkin seeds": 10}),
        meal("Hummus with cucumber & carrot sticks", 170, 6, 18, 8, [VG, V, D], {"Hummus": 60, "Cucumber": 80, "Carrot": 50}),
        meal("Two boiled eggs with pepper", 150, 12, 1, 10, [V, K, D], {"Eggs": 100}),
        meal("Peanut-butter celery sticks", 180, 6, 6, 15, [VG, V, K, D], {"Peanut butter": 25, "Celery": 80}),
    ],
}

ALLERGEN_KEYWORDS = {
    "Nuts": ["almond", "walnut", "peanut", "cashew"],
    "Dairy": ["paneer", "yogurt", "curd", "milk", "butter", "parmesan", "whey"],
    "Eggs": ["egg"],
    "Gluten": ["flour", "bread", "oats", "bajra"],
    "Soy": ["tofu", "soy"],
    "Fish": ["salmon", "fish"],
}
# 'Soy milk' contains 'milk' — treat as non-dairy
DAIRY_EXCEPTIONS = ["soy milk", "coconut milk"]

GROCERY_CATEGORIES = {
    "🥬 Produce": ["onion", "tomato", "spinach", "cucumber", "lettuce", "broccoli", "pepper", "carrot", "coriander", "mint",
                   "basil", "lemon", "avocado", "banana", "apple", "berries", "zucchini", "asparagus", "cauliflower",
                   "mushroom", "celery", "peas", "vegetables"],
    "🍗 Protein & Dairy": ["chicken", "salmon", "fish", "egg", "paneer", "tofu", "yogurt", "curd", "milk", "butter", "parmesan"],
    "🌾 Grains & Pulses": ["dal", "rice", "poha", "oats", "quinoa", "flour", "bread", "rajma", "chickpea", "chana", "lentil", "bajra"],
    "🥜 Nuts, Seeds & Pantry": ["almond", "walnut", "peanut", "chia", "seeds", "tahini", "hummus", "oil", "sauce", "coconut"],
}


def categorize(item: str) -> str:
    low = item.lower()
    for cat, kws in GROCERY_CATEGORIES.items():
        if any(k in low for k in kws):
            return cat
    return "🧂 Other"


def has_allergen(m: dict, allergens: list[str]) -> bool:
    for ing in m["ing"]:
        low = ing.lower()
        for a in allergens:
            for kw in ALLERGEN_KEYWORDS[a]:
                if kw in low:
                    if a == "Dairy" and any(ex in low for ex in DAIRY_EXCEPTIONS):
                        continue
                    return True
    return False


def calc_targets(age: int, sex: str, weight: float, height: float, activity: str, goal: str, diets: list[str]) -> dict:
    s = 5 if sex == "Male" else -161 if sex == "Female" else -78
    bmr = 10 * weight + 6.25 * height - 5 * age + s
    tdee = bmr * ACTIVITY[activity]
    target = tdee * (1 + GOALS[goal])
    floor = 1500 if sex == "Male" else 1200
    floored = target < floor
    target = max(target, floor)
    prot_per_kg = 2.0 if goal == "Fat Loss" else 1.8 if goal == "Muscle Gain" else 1.4
    protein = min(prot_per_kg * weight, target * 0.35 / 4)
    if "Keto" in diets:
        carbs = target * 0.06 / 4
        fat = (target - protein * 4 - carbs * 4) / 9
    else:
        carbs_share = 0.40 if "Diabetic-Friendly" in diets else 0.45
        carbs = target * carbs_share / 4
        fat = max((target - protein * 4 - carbs * 4) / 9, target * 0.20 / 9)
    bmi = weight / ((height / 100) ** 2)
    return {"bmr": bmr, "tdee": tdee, "target": target, "protein": protein, "carbs": carbs,
            "fat": fat, "bmi": bmi, "floored": floored}


def build_pool(slot: str, diets: list[str], allergens: list[str]) -> tuple[list[dict], bool]:
    """Return candidate meals and whether constraints had to be relaxed."""
    required = set()
    if "Vegan" in diets:
        required.add(VG)
    if "Vegetarian" in diets:
        required.add(V)
    if "Keto" in diets:
        required.add(K)
    if "Diabetic-Friendly" in diets:
        required.add(D)
    base = [m for m in MEALS[slot] if not has_allergen(m, allergens)]
    for drop in ([], [D], [D, K]):
        req = required - set(drop)
        pool = [m for m in base if req <= m["tags"]]
        if pool:
            return pool, bool(drop)
    return base or MEALS[slot], True


def build_plan(targets: dict, diets: list[str], allergens: list[str]) -> tuple[list[dict], bool]:
    relaxed_any = False
    pools = {}
    for slot in SLOT_SPLIT:
        pools[slot], rel = build_pool(slot, diets, allergens)
        relaxed_any |= rel
    days = []
    for d in range(7):
        entries = []
        for si, (slot, share) in enumerate(SLOT_SPLIT.items()):
            pool = pools[slot]
            m = pool[(d * 2 + si) % len(pool)] if len(pool) > 3 else pool[(d + si) % len(pool)]
            factor = min(2.0, max(0.6, targets["target"] * share / m["kcal"]))
            factor = round(factor * 4) / 4  # portions in quarter steps
            entries.append({"slot": slot, "meal": m, "factor": factor,
                            "kcal": m["kcal"] * factor, "p": m["p"] * factor,
                            "c": m["c"] * factor, "f": m["f"] * factor})
        days.append({"day": f"Day {d + 1}", "entries": entries})
    return days, relaxed_any


def grocery_list(days: list[dict]) -> pd.DataFrame:
    tot: dict[str, float] = defaultdict(float)
    for d in days:
        for e in d["entries"]:
            for ing, g in e["meal"]["ing"].items():
                tot[ing] += g * e["factor"]
    rows = []
    for item, g in tot.items():
        g_round = math.ceil(g / 50) * 50 if g >= 50 else math.ceil(g / 10) * 10
        qty = f"{g_round / 1000:.2f} kg".rstrip("0").rstrip(".") if g_round >= 1000 else f"{g_round} g"
        if item == "Eggs":
            qty = f"{math.ceil(g / 50)} eggs"
        rows.append({"Category": categorize(item), "Item": item, "Weekly quantity": qty, "Got it": False})
    df = pd.DataFrame(rows).sort_values(["Category", "Item"]).reset_index(drop=True)
    return df


def tab_planner() -> None:
    st.subheader("📅 Personalized Meal Planner")
    with st.form("profile"):
        c1, c2, c3, c4 = st.columns(4)
        age = c1.number_input("Age", 14, 90, 28)
        sex = c2.selectbox("Gender", ["Male", "Female", "Other"])
        weight = c3.number_input("Weight (kg)", 30.0, 250.0, 70.0, 0.5)
        height = c4.number_input("Height (cm)", 120.0, 230.0, 170.0, 0.5)
        c5, c6 = st.columns(2)
        activity = c5.selectbox("Activity level", list(ACTIVITY), index=2)
        goal = c6.selectbox("Fitness goal", list(GOALS))
        c7, c8 = st.columns(2)
        diets = c7.multiselect("Dietary constraints", ["Vegetarian", "Vegan", "Keto", "Diabetic-Friendly"])
        allergens = c8.multiselect("Avoid allergens", list(ALLERGEN_KEYWORDS))
        submitted = st.form_submit_button("🚀 Build my 7-day plan", type="primary")
    if submitted:
        try:
            t = calc_targets(int(age), sex, float(weight), float(height), activity, goal, diets)
            days, relaxed = build_plan(t, diets, allergens)
            st.session_state["plan"] = {"t": t, "days": days, "relaxed": relaxed, "diets": diets, "goal": goal}
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not build plan: {e}")
    plan = st.session_state.get("plan")
    if not plan:
        st.info("Fill in your profile and press **Build my 7-day plan**.")
        return
    t, days = plan["t"], plan["days"]

    cols = st.columns(5)
    for c, (l, v) in zip(cols, [("BMR", f"{t['bmr']:.0f} kcal"), ("TDEE", f"{t['tdee']:.0f} kcal"),
                                ("Daily target", f"{t['target']:.0f} kcal"), ("BMI", f"{t['bmi']:.1f}"),
                                ("Goal", plan["goal"])]):
        c.markdown(metric_card(l, v), unsafe_allow_html=True)
    st.markdown("")
    st.markdown(badge(f"Protein {t['protein']:.0f} g", "green") + badge(f"Carbs {t['carbs']:.0f} g", "amber")
                + badge(f"Fat {t['fat']:.0f} g", "red")
                + "".join(badge(d, "purple") for d in plan["diets"]), unsafe_allow_html=True)
    if t["floored"]:
        st.warning("Your calculated target fell below a safe minimum, so it was raised to 1,200/1,500 kcal. "
                   "Please consult a professional for aggressive deficits.")
    if plan["relaxed"]:
        st.warning("Not enough meals matched every constraint, so the planner relaxed the Diabetic-Friendly/Keto filter "
                   "for some slots. Review before following.")

    tot_rows = []
    for d in days:
        tot_rows.append({"Day": d["day"], "kcal": sum(e["kcal"] for e in d["entries"]),
                         "Protein": sum(e["p"] for e in d["entries"]), "Carbs": sum(e["c"] for e in d["entries"]),
                         "Fat": sum(e["f"] for e in d["entries"])})
    tdf = pd.DataFrame(tot_rows)
    fig = px.bar(tdf, x="Day", y="kcal", title="Planned calories vs target", color_discrete_sequence=["#ff7a18"])
    fig.add_hline(y=t["target"], line_dash="dash", line_color="#7c3aed", annotation_text="target")
    fig.update_layout(height=280, margin=dict(t=50, b=10, l=10, r=10))
    st.plotly_chart(fig, width="stretch")

    plan_tab, groc_tab = st.tabs(["🍽️ 7-day schedule", "🛒 Grocery list"])
    with plan_tab:
        for d, tr in zip(days, tot_rows):
            with st.expander(f"{d['day']} — {tr['kcal']:.0f} kcal · P {tr['Protein']:.0f} / C {tr['Carbs']:.0f} / F {tr['Fat']:.0f} g"):
                df = pd.DataFrame([{"Meal": e["slot"], "Dish": e["meal"]["name"], "Portion": f"{e['factor']:g}×",
                                    "kcal": round(e["kcal"]), "P (g)": round(e["p"]), "C (g)": round(e["c"]),
                                    "F (g)": round(e["f"])} for e in d["entries"]])
                st.dataframe(df, hide_index=True, width="stretch")
        md = ["# Food Verse AI — 7-day meal plan", f"Daily target: {t['target']:.0f} kcal\n"]
        for d in days:
            md.append(f"## {d['day']}")
            for e in d["entries"]:
                md.append(f"- **{e['slot']}**: {e['meal']['name']} ({e['factor']:g}×, {e['kcal']:.0f} kcal)")
        st.download_button("⬇️ Download plan (Markdown)", "\n".join(md), "meal_plan.md", "text/markdown")
    with groc_tab:
        gdf = grocery_list(days)
        edited = st.data_editor(gdf, hide_index=True, width="stretch",
                                disabled=["Category", "Item", "Weekly quantity"], key="groc_editor")
        st.caption(f"{len(gdf)} items for 7 days. Tick items as you shop.")
        st.download_button("⬇️ Download grocery list (CSV)", gdf.drop(columns="Got it").to_csv(index=False),
                           "grocery_list.csv", "text/csv")
    st.caption("Estimates for general wellness only. Consult a doctor or dietitian, especially for diabetes or medical conditions.")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    sidebar()
    st.markdown(
        '<div class="fv-hero"><h1>🍽️ Food Verse AI</h1>'
        "<p>Multimodal Food Intelligence Agent — recognize, audit, protect, cook and plan, all in one place.</p></div>",
        unsafe_allow_html=True)
    tabs = st.tabs(["🥗 Nutrition Audit", "🥬 Freshness", "🧼 Kitchen Hygiene", "👩‍🍳 Recipes", "📅 Meal Planner"])
    for tab, fn in zip(tabs, [tab_nutrition, tab_freshness, tab_hygiene, tab_recipes, tab_planner]):
        with tab:
            try:
                fn()
            except Exception as e:  # noqa: BLE001
                st.error(f"Something went wrong in this module: {e}")


main()
