# streamlit run phys_modeller.py
#  directory setup: cd C:\users\oakhtar\documents\pyprojs_local
import streamlit as st
import openai
import numpy as np
import plotly.graph_objects as go
import hashlib
import time
import re

# --- CONFIG ---
st.set_page_config(page_title="GenAI Physics Modeler", page_icon="atom", layout="wide")

# --- PRICING (Nov 2025) ---
PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "grok-4-1-fast-reasoning": {"input": 2.00, "output": 6.00},
}

# --- SECRETS ---
def get_key(name):
    return st.secrets.get(name) or st.secrets.get(name.upper()) or st.secrets.get(name.lower())

# --- AUTH ---
def check_password():
    pwd = get_key("app_password")
    if not pwd:
        st.error("Missing `app_password` in secrets")
        st.stop()
    def submit():
        st.session_state.auth = st.session_state.pwd == pwd
    if "auth" not in st.session_state:
        st.text_input("Password", type="password", key="pwd", on_change=submit)
        st.stop()
    if not st.session_state.auth:
        st.text_input("Password", type="password", key="pwd", on_change=submit)
        st.error("Wrong password")
        st.stop()

check_password()

# --- CACHING ---
@st.cache_data(show_spinner=False)
def get_cached_code(prompt_hash):
    return st.session_state.get(f"cache_{prompt_hash}")

def set_cached_code(prompt_hash, code):
    st.session_state[f"cache_{prompt_hash}"] = code

# --- SYSTEM PROMPT (Golden) ---
SYSTEM_PROMPT = """
You are an expert Python physics visualizer using Plotly.

RULES:
- Use only: numpy (np), plotly.graph_objects (go), streamlit (st)
- Generate exactly 100 frames of precomputed data
- Define a variable `fig` (Plotly Figure) at the end
- Include Play/Pause buttons via updatemenus
- Use smooth, continuous motion (no back-and-forth unless asked)
- All arrays must be float64
- Output RAW Python code only — no markdown, no explanation
"""

# --- MAIN ---
with st.sidebar:
    st.title("Settings")
    provider = st.radio("Model", ["OpenAI (gpt-4o-mini)", "OpenAI (gpt-4o)", "xAI Grok-4.1"])
    
    if "gpt-4o-mini" in provider:
        model = "gpt-4o-mini"
        key = get_key("openai_api_key")
        base_url = None
    elif "gpt-4o" in provider:
        model = "gpt-4o"
        key = get_key("openai_api_key")
        base_url = None
    else:
        model = "grok-4-1-fast-reasoning"
        key = get_key("xai_api_key")
        base_url = "https://api.x.ai/v1"

    if not key:
        st.error("Missing API key")
        st.stop()

    speed = st.slider("Animation Speed", 1, 100, 40)
    frame_ms = max(10, 1000 // speed)

st.title("Generative Physics Modeler")
st.caption("Describe a physics scene → get a real-time 3D animation")

user_input = st.text_area(
    "Describe the physics scene",
    height=120,
    placeholder="e.g., A red pendulum swinging inside a glowing blue wireframe cube with gravity particles falling",
    value="A red sphere orbiting a glowing yellow sun with 12 moons in elliptical paths, all inside a rotating wireframe dodecahedron"
)

if st.button("Generate Animation", type="primary"):
    if not key:
        st.error("API key missing")
        st.stop()

    prompt_hash = hashlib.md5(user_input.encode()).hexdigest()
    cached = get_cached_code(prompt_hash)
    if cached:
        st.success("Loaded from cache!")
        final_code = cached
    else:
        with st.status(f"Generating with {model}...", expanded=True) as status:
            st.write("Sending to AI...")
            client = openai.OpenAI(api_key=key, base_url=base_url)

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_input}
            ]

            for attempt in range(3):
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=0.3,
                        max_tokens=4000
                    )
                    code = response.choices[0].message.content.strip()
                    code = re.sub(r"^```python|^```|```$", "", code, flags=re.MULTILINE).strip()

                    # Test execution
                    test_env = {"np": np, "go": go, "st": st}
                    exec(code, test_env)
                    if "fig" not in test_env:
                        raise ValueError("No 'fig' defined")

                    final_code = code
                    set_cached_code(prompt_hash, code)
                    status.update(label="Success!", state="complete")
                    break

                except Exception as e:
                    error_msg = f"Attempt {attempt+1} failed: {str(e)[:200]}"
                    st.warning(error_msg)
                    messages.append({"role": "assistant", "content": code if 'code' in locals() else ""})
                    messages.append({"role": "user", "content": f"Fix this error and return only corrected code:\n{str(e)}"})
                    if attempt == 2:
                        st.error("Failed after 3 attempts")
                        st.stop()

    # --- Render ---
    st.session_state.generated_code = final_code

if "generated_code" in st.session_state:
    code = st.session_state.generated_code

    with st.expander("View & Download Code"):
        c1, c2 = st.columns([1, 4])
        with c1:
            st.download_button("Download .py", code, "physics_sim.py")
        with c2:
            st.code(code, language="python")

    try:
        env = {"np": np, "go": go, "st": st}
        exec(code, env)
        fig = env["fig"]

        # Make it beautiful
        fig.update_layout(
            height=800,
            margin=dict(l=0, r=0, t=30, b=0),
            title="AI-Generated Physics Simulation",
            title_x=0.5,
            scene=dict(aspectmode='data')
        )

        # Fix Play/Pause buttons
        if fig.layout.updatemenus:
            for btn in fig.layout.updatemenus[0].buttons:
                if btn.label == "Play" and btn.args:
                    if len(btn.args) > 1 and isinstance(btn.args[1], dict):
                        btn.args[1]["frame"]["duration"] = frame_ms

        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "scrollZoom": True})

    except Exception as e:
        st.error(f"Render failed: {e}")
        st.code(code, language="python")
