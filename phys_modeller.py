# streamlit run phys_modeller.py
#  directory setup: cd C:\users\oakhtar\documents\pyprojs_local
import streamlit as st
import openai
import numpy as np
import plotly.graph_objects as go
import re
import types

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="GenAI Physics Modeler", page_icon="⚛️")

# --- Constants ---
# Updated exact pricing from xAI Docs (Nov 2025 data)
PRICING = {
    # OpenAI
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    
    # xAI (Exact values from your screenshot)
    "grok-2-1212":           {"input": 2.00, "output": 10.00}, # Standard v2
    "grok-4-0709":           {"input": 3.00, "output": 15.00}, # Standard v4
    "grok-4-1-fast-reasoning": {"input": 0.20, "output": 0.50},  # Ultra-efficient
    "grok-3":                {"input": 3.00, "output": 15.00}, 
}

SCENARIOS = {
    "Custom": "",
    "Rotating Sphere with Gas": "A wireframe sphere rotating on the Z-axis with 20 gas molecules bouncing around it inside a cubic container.",
    "Rarefied Gas Spin-Up": "Simulate the transfer of conserved momentum from a rotating solid disk to gas molecules immediately adjacent to it in a vacuum. The gas should gradually spin up due to wall collisions.",
    "Solar System w/ Comet": "A solar system simulation with a static yellow sun, 3 orbiting planets at different distances/speeds, and a comet passing through on a hyperbolic trajectory.",
    "Damped Pendulum": "A 3D visualization of a simple pendulum with damping. Show the pendulum bob swinging in 3D space and trace its path color-coded by velocity.",
    "Lorenz Attractor": "Simulate the Lorenz attractor (chaotic system). Visualize the trajectory of a point over time in 3D space, leaving a trail.",
}

# --- Session State Initialization ---
if "history" not in st.session_state: st.session_state.history = [] 
if "prompt_text" not in st.session_state: st.session_state.prompt_text = SCENARIOS["Rotating Sphere with Gas"]

# --- Helper: Safe Secret Retrieval ---
def get_secret(key_name):
    for key in [key_name, key_name.upper(), key_name.lower()]:
        if key in st.secrets: return st.secrets[key]
    return None

# --- Authentication ---
def check_password():
    stored_password = get_secret("app_password")
    if not stored_password:
        st.error("❌ Configuration Error: 'app_password' not found in Secrets.")
        st.stop()

    def password_entered():
        if st.session_state["password"] == stored_password:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Enter App Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Enter App Password", type="password", on_change=password_entered, key="password")
        st.error("😕 Password incorrect")
        return False
    return True

# --- Logic: Sandbox Execution ---
def execute_safe_code(code_str, global_vars):
    def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in ['os', 'sys', 'subprocess', 'shutil', 'requests']:
            raise ImportError(f"Importing '{name}' is not allowed for security reasons.")
        return __import__(name, globals, locals, fromlist, level)

    safe_builtins = __builtins__.copy() if isinstance(__builtins__, dict) else __builtins__.__dict__.copy()
    safe_builtins['__import__'] = restricted_import
    global_vars['__builtins__'] = safe_builtins
    
    try:
        exec(code_str, global_vars)
        return True, None
    except Exception as e:
        return False, str(e)

# --- Logic: LLM Generation ---
def clean_code(code):
    code = re.sub(r'^```python', '', code)
    code = re.sub(r'^```', '', code)
    code = re.sub(r'```$', '', code)
    return code.strip()

def estimate_cost(input_text, output_text, model):
    # Default to expensive fallback if model mismatch to warn user
    rates = PRICING.get(model, {"input": 3.00, "output": 15.00}) 
    in_tok = len(input_text) / 4
    out_tok = len(output_text) / 4
    return (in_tok/1e6)*rates["input"], (out_tok/1e6)*rates["output"]

def get_system_prompt():
    return """
    You are a Python Code Generator for a 3D Physics Visualization using `numpy` and `plotly.graph_objects`.
    
    STRICT REQUIREMENTS:
    1. **Output:** RETURN ONLY RAW PYTHON CODE. No markdown.
    2. **Libraries:** Use `import numpy as np` and `import plotly.graph_objects as go`.
    3. **Variable:** Define a final figure variable named `fig`.
    4. **Animation:** 
       - Pre-calculate exactly 60-90 frames.
       - Use `go.Frame` objects attached to `fig.frames`.
       - **DO NOT define `updatemenus` (Play buttons).** The host app will add these automatically.
    5. **Physics:** Initialize positions as floats.
    
    CRITICAL VISUALIZATION RULES:
    1. **FIXED AXIS RANGES:** You MUST set `layout.scene.xaxis.range` (and y/z) to fixed hardcoded values (e.g., [-10, 10]) covering the bounds. Do NOT use auto-scaling.
    2. **Camera:** Set a reasonable default camera eye.
    """

def call_llm(messages, key, url, model):
    client = openai.OpenAI(api_key=key, base_url=url)
    # Note: We do not use 'frequency_penalty' or 'presence_penalty' 
    # as these are not supported by Grok 4 reasoning models per documentation.
    response = client.chat.completions.create(model=model, messages=messages, temperature=0.5)
    return response.choices[0].message.content

def generate_simulation(prompt, key, url, model):
    system_prompt = get_system_prompt()
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
    
    raw_code = call_llm(messages, key, url, model)
    code = clean_code(raw_code)
    
    dummy_globals = {"np": np, "go": go, "st": st}
    success, error = execute_safe_code(code, dummy_globals)
    
    if not success:
        err_msg = f"The code raised this error: {error}. Please fix the code and return ONLY valid Python."
        messages.append({"role": "assistant", "content": code})
        messages.append({"role": "user", "content": err_msg})
        raw_code = call_llm(messages, key, url, model)
        code = clean_code(raw_code)
    
    return code, system_prompt + prompt, raw_code

def update_prompt():
    selection = st.session_state.scenario_selector
    if selection != "Custom":
        st.session_state.prompt_text = SCENARIOS[selection]

# --- Main App ---
def main_app():
    
    with st.sidebar:
        st.title("⚙️ Settings")
        
        # --- Provider Selection with Version Control ---
        st.subheader("AI Provider")
        provider = st.radio("Model Source", ["xAI (Grok)", "OpenAI"], label_visibility="collapsed")
        
        api_key = None
        base_url = None
        model_name = ""
        
        if provider == "xAI (Grok)":
            api_key = get_secret("xai_api_key")
            base_url = "https://api.x.ai/v1"
            
            # UPDATED MODEL SELECTOR based on Table Data
            model_options = {
                "Grok 4.1 Fast (Reasoning) [Best Value]": "grok-4-1-fast-reasoning",
                "Grok 4 (Standard)": "grok-4-0709",
                "Grok 2 (Legacy)": "grok-2-1212",
                "Grok 3 (Legacy)": "grok-3"
            }
            
            choice = st.selectbox("Version", list(model_options.keys()))
            model_name = model_options[choice]
            
        else: # OpenAI
            api_key = get_secret("openai_api_key")
            model_name = st.selectbox("Version", ["gpt-4o", "gpt-4o-mini"])
            
        if not api_key: st.error(f"Missing API Key for {provider}")
        st.divider()
        
        # Animation Controls
        st.subheader("🎮 Animation Controls")
        speed = st.slider("Speed Factor", 10, 200, 50)
        frame_dur = int(1000/speed)
        st.caption(f"Frame Duration: {frame_dur}ms")
        st.divider()
        
        st.subheader("📜 Session History")
        if len(st.session_state.history) > 0:
            for i, item in enumerate(reversed(st.session_state.history)):
                if i > 4: break
                if st.button(f"Load #{len(st.session_state.history)-i}: {item['model']}", key=f"hist_{i}"):
                    st.session_state["current_code"] = item['code']
                    st.session_state["prompt_text"] = item['prompt']
                    st.rerun()

    st.title("⚛️ Generative Physics Modeler")
    
    with st.container():
        c1, c2 = st.columns([3, 1])
        with c1:
            st.selectbox("📚 Scenarios", list(SCENARIOS.keys()), key="scenario_selector", on_change=update_prompt)
        
        user_prompt = st.text_area("Physics Description", value=st.session_state.prompt_text, height=100, key="prompt_input")
        st.session_state.prompt_text = user_prompt

        if st.button("🚀 Generate Simulation", type="primary", use_container_width=True, disabled=not api_key):
            with st.status(f"Simulating with {model_name}...", expanded=True) as status:
                try:
                    st.write("🧠 Thinking...")
                    final_code, in_txt, out_txt = generate_simulation(user_prompt, api_key, base_url, model_name)
                    
                    st.write("✍️ Compiling...")
                    st.session_state["current_code"] = final_code
                    c_in, c_out = estimate_cost(in_txt, out_txt, model_name)
                    
                    st.session_state.history.append({
                        "code": final_code,
                        "prompt": user_prompt,
                        "model": model_name,
                        "cost": c_in + c_out
                    })
                    
                    status.update(label="Done!", state="complete", expanded=False)
                    st.rerun()
                except Exception as e:
                    st.error(f"Generation failed: {e}")
                    status.update(label="Failed", state="error")

    if "current_code" in st.session_state and st.session_state["current_code"]:
        with st.expander("View Code & Details"):
            tab_code, tab_cost = st.tabs(["Python Code", "Cost Analysis"])
            with tab_code:
                st.code(st.session_state["current_code"], language="python")
                st.download_button("📥 Download .py", st.session_state["current_code"], "simulation.py")
            with tab_cost:
                if st.session_state.history:
                    last = st.session_state.history[-1]
                    st.metric("Est. Cost", f"${last['cost']:.4f}")

        exec_globals = {"np": np, "go": go, "st": st}
        success, error = execute_safe_code(st.session_state["current_code"], exec_globals)
        
        if success and "fig" in exec_globals:
            fig = exec_globals["fig"]
            
            # --- Inject Working Play Buttons Programmatically ---
            fig.update_layout(
                updatemenus=[dict(
                    type="buttons",
                    showactive=False,
                    y=1, x=0, xanchor="left", yanchor="top",
                    pad=dict(t=10, r=10),
                    buttons=[
                        dict(label="▶️ Play",
                             method="animate",
                             args=[None, dict(frame=dict(duration=frame_dur, redraw=True), 
                                              fromcurrent=True, 
                                              transition=dict(duration=0))]),
                        dict(label="⏸️ Pause",
                             method="animate",
                             args=[[None], dict(frame=dict(duration=0, redraw=False), 
                                                mode="immediate", 
                                                transition=dict(duration=0))])
                    ]
                )],
                height=800,
                autosize=True,
                margin=dict(l=0, r=0, t=0, b=0),
                scene=dict(aspectmode='cube')
            )
            
            st.plotly_chart(fig, use_container_width=True)
        elif not success:
            st.error(f"⚠️ Runtime Error in generated code:\n{error}")
        else:
            st.error("⚠️ The AI generated code, but did not assign the visualization to a variable named `fig`.")

if __name__ == "__main__":
    if check_password():
        main_app()
