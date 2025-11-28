# streamlit run phys_modeller.py
#  directory setup: cd C:\users\oakhtar\documents\pyprojs_local
import streamlit as st
import openai
import numpy as np
import plotly.graph_objects as go
import re
import time

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="GenAI Physics Modeler")

# --- Constants ---
PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "grok-beta": {"input": 5.00, "output": 15.00}
}

# --- Helper: Safe Secret Retrieval ---
def get_secret(key_name):
    if key_name in st.secrets: return st.secrets[key_name]
    if key_name.upper() in st.secrets: return st.secrets[key_name.upper()]
    if key_name.lower() in st.secrets: return st.secrets[key_name.lower()]
    return None

# --- Authentication Logic ---
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
    else:
        return True

# --- Core Application ---
def main_app():
    st.title("⚛️ Generative Physics Modeler")

    # --- Sidebar: Configuration & Cost ---
    with st.sidebar:
        st.header("Configuration")
        provider = st.radio("Select Provider", ["OpenAI", "xAI (Grok)"])

        api_key = None
        base_url = None
        model_name = ""
        
        if provider == "OpenAI":
            api_key = get_secret("openai_api_key")
            if api_key: model_name = "gpt-4o" 
            else: st.error("Missing 'openai_api_key'")
                
        elif provider == "xAI (Grok)":
            api_key = get_secret("xai_api_key")
            if api_key: 
                base_url = "https://api.x.ai/v1"
                model_name = "grok-beta"
            else: st.error("Missing 'xai_api_key'")

        st.divider()
        
        # Cost Estimator
        with st.expander("💰 Cost Estimate", expanded=False):
            if "last_cost_data" in st.session_state:
                c_in, c_out, t_in, t_out = st.session_state["last_cost_data"]
                st.caption(f"Last Run ({model_name})")
                st.write(f"**Input:** ${c_in:.4f}")
                st.write(f"**Output:** ${c_out:.4f}")
                st.markdown(f"### Total: ${c_in+c_out:.4f}")
            else:
                st.info("Run a simulation to see costs.")

    # --- Helpers ---
    def clean_code(code):
        code = re.sub(r'^```python', '', code)
        code = re.sub(r'^```', '', code)
        code = re.sub(r'```$', '', code)
        return code.strip()

    def estimate_cost(input_text, output_text, model):
        in_tok = len(input_text) / 4
        out_tok = len(output_text) / 4
        rates = PRICING.get(model, {"input": 2.50, "output": 10.00})
        return (in_tok/1e6)*rates["input"], (out_tok/1e6)*rates["output"], in_tok, out_tok

    def get_system_prompt():
        return """
        You are a Python Code Generator for a 3D Physics Visualization.
        
        GOAL: Convert description to a Python script using `numpy` and `plotly.graph_objects`.
        
        STRICT CONSTRAINTS:
        1. Libraries: `numpy` (as np), `plotly.graph_objects` (as go), `streamlit` (as st).
        2. NO infinite loops in Python. Pre-calculate 60 frames of data.
        3. **CRITICAL:** Define a figure variable named `fig`. 
        4. **CRITICAL:** Do NOT call `st.plotly_chart` or `fig.show()`. The host app will render `fig`.
        5. In `fig.layout.updatemenus`, set type='buttons' (Play/Pause).
        6. Try to create a seamless loop (make the data in the last frame approach the data in the first frame).
        7. Adhere to physics principles using NumPy vector math.
        8. Output RAW CODE only (no markdown blocks).
        9. Initialize all arrays as floats.
        """

    def call_llm(messages, key, url, model):
        client = openai.OpenAI(api_key=key, base_url=url)
        response = client.chat.completions.create(model=model, messages=messages, temperature=0.5)
        return response.choices[0].message.content

    def generate_with_retry(prompt, key, url, model, max_retries=2):
        system_prompt = get_system_prompt()
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
        
        raw_code = call_llm(messages, key, url, model)
        code = clean_code(raw_code)
        
        t_in = system_prompt + prompt
        t_out = raw_code
        
        for attempt in range(max_retries + 1):
            try:
                compile(code, '<string>', 'exec')
                return code, None, t_in, t_out
            except Exception as e:
                if attempt < max_retries:
                    err_prompt = f"Error: {str(e)}. Fix code and return ONLY valid Python."
                    messages.append({"role": "assistant", "content": code})
                    messages.append({"role": "user", "content": err_prompt})
                    raw_code = call_llm(messages, key, url, model)
                    code = clean_code(raw_code)
                    t_in += err_prompt; t_out += raw_code
                else:
                    return None, str(e), t_in, t_out

    # --- Main Layout ---
    col1, col2 = st.columns([1, 2])

    with col1:
        user_instruction = st.text_area("Physics Description", height=150, 
            value="A wireframe sphere rotating on the Z-axis with 20 gas molecules bouncing around it inside a cubic container.")
        
        if api_key:
            generate_btn = st.button("Generate Simulation", type="primary")
        else:
            st.warning("API Key missing.")
            generate_btn = False
        
        st.divider()
        
        # --- Improved Animation Controls ---
        st.subheader("🎮 Controls")
        
        # Speed Factor: 1 (Slow) to 100 (Fast).
        speed_factor = st.slider("⚡ Animation Speed", min_value=1, max_value=100, value=50, help="Slide right to make the animation faster.")
        
        # Math: Speed 100 -> 10ms delay. Speed 1 -> 1000ms delay.
        frame_duration = int(1000 / speed_factor)
        
        st.caption(f"Current Setting: {frame_duration}ms per frame")

    with col2:
        if "generated_code" not in st.session_state:
            st.session_state["generated_code"] = None

        if generate_btn:
            with st.spinner(f"Simulating ({model_name})..."):
                final_code, error, in_txt, out_txt = generate_with_retry(user_instruction, api_key, base_url, model_name)
                
                if final_code:
                    st.session_state["generated_code"] = final_code
                    c_in, c_out, t_in, t_out = estimate_cost(in_txt, out_txt, model_name)
                    st.session_state["last_cost_data"] = (c_in, c_out, t_in, t_out)
                    st.rerun()
                else:
                    st.error(f"Generation failed: {error}")

        if st.session_state["generated_code"]:
            with st.expander("View Code", expanded=False):
                st.code(st.session_state["generated_code"], language='python')
                st.download_button("Download .py", st.session_state["generated_code"], "sim.py", "text/x-python")

            # --- Execute and Render ---
            try:
                # 1. Setup execution context
                exec_globals = {"st": st, "np": np, "go": go, "__name__": "__main__"}
                
                # 2. Execute code (Defines 'fig', does not render)
                exec(st.session_state["generated_code"], exec_globals)
                
                # 3. Extract Figure & Apply Speed
                if "fig" in exec_globals:
                    fig = exec_globals["fig"]
                    
                    # FIXED: Safely apply speed to the Play Button only.
                    # We do NOT iterate over fig.frames to set duration anymore.
                    try:
                        if fig.layout.updatemenus:
                            for menu in fig.layout.updatemenus:
                                if menu.type == 'buttons':
                                    for button in menu.buttons:
                                        if button.label == 'Play':
                                            if len(button.args) > 1 and isinstance(button.args[1], dict):
                                                # Update frame duration (speed)
                                                button.args[1]['frame']['duration'] = frame_duration
                                                # Update transition duration (smoothness)
                                                button.args[1]['transition']['duration'] = 0
                    except Exception as e:
                        # If button structure varies, we skip optimization to prevent crash
                        print(f"Could not update animation speed: {e}")

                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Code executed but `fig` variable was not defined.")
            except Exception as e:
                st.error(f"Runtime Error: {e}")

if __name__ == "__main__":
    if check_password():
        main_app()
