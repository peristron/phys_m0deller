# streamlit run phys_modeller.py
#  directory setup: cd C:\users\oakhtar\documents\pyprojs_local
import streamlit as st
import openai
import numpy as np
import plotly.graph_objects as go
import hashlib
import re

st.set_page_config(page_title="GenAI Physics Modeler", page_icon="atom", layout="wide")

# --- PRICING ---
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

# --- SYSTEM PROMPT ---
SYSTEM_PROMPT = """
You are an expert Python physics animator using Plotly.

MANDATORY RULES — FOLLOW EXACTLY:
1. Use only: numpy as np, plotly.graph_objects as go
2. Generate EXACTLY 100 precomputed frames using a list called `frames`
3. Define `frames = [go.Frame(data=..., name=str(i)) for i in range(100)]`
4. Create the figure with BOTH data and frames:  
   fig = go.Figure(data=initial_data, frames=frames)
5. Add Play/Pause buttons using updatemenus
6. Use smooth, continuous motion (rotation, orbit, etc.)
7. All arrays must be float64
8. Output ONLY raw, valid Python code — no markdown, no explanation
"""

# --- SIDEBAR ---
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

    speed = st.slider("Animation Speed", 1, 100, 40, help="Higher = faster")
    frame_ms = max(10, 1000 // speed)

    st.divider()
    st.subheader("Live Cost")
    if "last_cost" in st.session_state:
        st.metric("Total Cost", f"${st.session_state.last_cost:.4f}")
        st.caption(f"Using {model}")
    else:
        st.info("Cost appears after first generation")

st.title("Generative Physics Modeler")
st.caption("Describe a physics scene → get a real-time 3D animation")

user_input = st.text_area(
    "Describe the physics scene",
    height=120,
    placeholder="e.g., A red sphere orbiting a glowing yellow sun with 12 moons in elliptical paths",
    value="A red cube orbiting a glowing, large fuschia sun with 99 moons in elliptical paths, all inside a rotating wireframe dodecahedron"
)

if st.button("Generate Animation", type="primary"):
    prompt_hash = hashlib.md5(user_input.encode()).hexdigest()
    cached = get_cached_code(prompt_hash)
    if cached:
        st.success("Loaded from cache!")
        st.session_state.generated_code = cached
        st.session_state.last_cost = 0.0
    else:
        with st.status(f"Generating with {model}...", expanded=True) as status:
            st.write("Sending to AI...")
            client = openai.OpenAI(api_key=key, base_url=base_url)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_input}
            ]

            total_input = len(user_input)
            total_output = 0

            for attempt in range(4):
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=0.2 if attempt > 1 else 0.3,
                        max_tokens=4000
                    )
                    raw_code = response.choices[0].message.content.strip()
                    code = re.sub(r"^```python|^```|```$", "", raw_code, flags=re.MULTILINE).strip()

                    if not code or len(code) < 50:
                        raise ValueError("Empty or too short code")

                    test_env = {"np": np, "go": go}
                    exec(code, test_env)
                    if "fig" not in test_env:
                        raise ValueError("No 'fig' variable defined")

                    final_code = code
                    set_cached_code(prompt_hash, code)
                    total_output += len(raw_code)
                    status.update(label="Success!", state="complete")
                    break

                except Exception as e:
                    error_msg = str(e)
                    if attempt < 3:
                        st.warning(f"Attempt {attempt + 1} failed → retrying...")
                        messages.append({"role": "assistant", "content": raw_code if 'raw_code' in locals() else ""})
                        messages.append({"role": "user", "content": 
                            f"CRITICAL ERROR:\n{error_msg}\n\n"
                            "Fix the code and return ONLY valid, runnable Python. No markdown. No explanation."
                        ))
                        total_output += len(raw_code or "")
                    else:
                        st.error(f"Failed after 4 attempts. Last error: {error_msg}")
                        st.code(raw_code, language="python")
                        st.stop()

            in_cost = (total_input / 1_000_000) * PRICING[model]["input"]
            out_cost = (total_output / 1_000_000) * PRICING[model]["output"]
            st.session_state.last_cost = in_cost + out_cost

        st.session_state.generated_code = final_code

# === FINAL RENDERING: INFINITE LOOP + NO ERRORS ===
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

        # --- GUARANTEE FRAMES ---
        if not hasattr(fig, "frames") or not fig.frames:
            theta = np.linspace(0, 2*np.pi, 100)
            x = np.cos(theta * 5)
            y = np.sin(theta * 5)
            z = np.zeros_like(theta)
            frames = [go.Frame(data=[go.Scatter3d(x=[x[i]], y=[y[i]], z=[z[i]], mode='markers', marker=dict(color='red', size=12))], name=str(i)) for i in range(100)]
            fig.frames = frames

        # --- INFINITE LOOP + SPEED CONTROL ---
        if fig.layout.updatemenus:
            for btn in fig.layout.updatemenus[0].buttons:
                if btn.label == "Play":
                    btn.args = [
                        None,
                        {
                            "frame": {"duration": frame_ms, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": 0},
                            "mode": "immediate"
                        }
                    ]

        # --- BUTTONS ABOVE CHART ---
        if fig.layout.updatemenus:
            play_pause = fig.layout.updatemenus[0].to_plotly_json()
            play_pause.update({
                "y": 1.15, "x": 0.0, "xanchor": "left", "yanchor": "top",
                "bgcolor": "rgba(30,30,30,0.95)",
                "bordercolor": "#00cc99",
                "borderwidth": 2,
                "font": {"color": "white", "size": 13}
            })
            fig.update_layout(updatemenus=[play_pause])

        fig.update_layout(
            height=800,
            margin=dict(l=0, r=0, t=60, b=0),
            title="AI-Generated Physics Simulation",
            title_x=0.5,
            scene=dict(aspectmode='data')
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displaylogo": False, "scrollZoom": True}
        )

    except Exception as e:
        st.error(f"Render failed: {e}")
        st.code(code, language="python")
