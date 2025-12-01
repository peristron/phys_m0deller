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

# --- Pre-Defined Scenarios (Bridge between Hard-coded & Generative) ---
SCENARIOS = {
    "Custom": "",
    "Rotating Sphere with Gas (Basic)": "A wireframe sphere rotating on the Z-axis with 20 gas molecules bouncing around it inside a cubic container.",
    "Rarefied Gas Spin-Up (Advanced)": "Simulate the transfer of conserved momentum from a rotating solid disk to gas molecules immediately adjacent to it in a vacuum. The gas should gradually spin up due to wall collisions. Use free molecular flow physics.",
    "Solar System with Comet": "A solar system simulation with a static yellow sun, 3 orbiting planets at different distances/speeds, and a comet passing through on a hyperbolic trajectory.",
    "Damped Pendulum Phase Space": "A 3D visualization of a simple pendulum with damping. Show the pendulum bob swinging in 3D space and trace its path color-coded by velocity.",
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
    
    # --- SIDEBAR: All Controls ---
    with st.sidebar:
        st.title("⚙️ Settings")
        
        # 1. Provider Config
        st.subheader("AI Provider")
        provider = st.radio("Select Model", ["OpenAI", "xAI (Grok)"])
        
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

        # 2. Animation Controls
        st.subheader("🎮 Animation Speed")
        speed_factor = st.slider("Speed", min_value=1, max_value=100, value=50, label_visibility="collapsed")
        frame_duration = int(1000 / speed_factor)
        st.caption(f"Frame Delay: {frame_duration}ms")

        st.divider()
        
        # 3. Cost Estimate
        with st.expander("💰 Cost Estimate", expanded=False):
            if "last_cost_data" in st.session_state:
                c_in, c_out, t_in, t_out = st.session_state["last_cost_data"]
                st.caption(f"Last Run ({model_name})")
                st.write(f"**Input:** ${c_in:.4f}")
                st.write(f"**Output:** ${c_out:.4f}")
                st.markdown(f"### Total: ${c_in+c_out:.4f}")
            else:
                st.info("Run a simulation to see costs.")

    # --- MAIN PAGE ---
    st.title("⚛️ Generative Physics Modeler")

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
        2. NO infinite loops. Pre-calculate 90 frames of data.
        3. **CRITICAL:** Define a figure variable named `fig`. 
        4. **CRITICAL:** Do NOT call `st.plotly_chart` or `fig.show()`. The host app will render `fig`.
        5. In `fig.layout.updatemenus`, set type='buttons' with 'Play' and 'Pause' buttons.
        6. Ensure the simulation uses linear, continuous motion (e.g. full rotation).
        7. **IMPORTANT:** When defining frames, ensure you update the specific data of the traces.
        8. **PHYSICS:** If simulating gas in vacuum, consider free molecular flow and momentum transfer via collisions.
        9. Output RAW CODE only (no markdown blocks).
        10. Initialize all arrays as floats.
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
                test_globals = {"st": st, "np": np, "go": go, "__name__": "__main__"}
                exec(code, test_globals)
                if "fig" not in test_globals: raise ValueError("No 'fig' defined.")
                return code, None, t_in, t_out
            except Exception as e:
                if attempt < max_retries:
                    err_prompt = f"Runtime Error: {str(e)}. Fix code/matrix shapes and return ONLY valid Python."
                    messages.append({"role": "assistant", "content": code})
                    messages.append({"role": "user", "content": err_prompt})
                    raw_code = call_llm(messages, key, url, model)
                    code = clean_code(raw_code)
                    t_in += err_prompt; t_out += raw_code
                else:
                    return None, str(e), t_in, t_out

    # --- Input Section ---
    with st.container():
        # Scenario Selector
        col_sel, col_dummy = st.columns([3, 2])
        with col_sel:
            selected_scenario = st.selectbox("📚 Load Example Scenario", list(SCENARIOS.keys()))
        
        # Text Input & Button
        col_input, col_btn = st.columns([4, 1])
        with col_input:
            # Determine default text
            default_text = SCENARIOS[selected_scenario] if selected_scenario != "Custom" else ""
            
            # We use session state to handle text area updates from selectbox
            if "prompt_text" not in st.session_state: st.session_state.prompt_text = SCENARIOS["Rotating Sphere with Gas (Basic)"]
            if selected_scenario != "Custom": st.session_state.prompt_text = default_text
            
            user_instruction = st.text_area(
                "Physics Description", 
                height=100, 
                value=st.session_state.prompt_text,
                placeholder="Describe the simulation..."
            )
            
        with col_btn:
            st.write("") # Spacing
            st.write("") # Spacing
            if api_key:
                generate_btn = st.button("🚀 Generate", type="primary", use_container_width=True)
            else:
                st.warning("Key Missing")
                generate_btn = False

    if "generated_code" not in st.session_state:
        st.session_state["generated_code"] = None

    if generate_btn:
        st.session_state["generated_code"] = None 
        
        # --- STATUS INDICATOR ---
        with st.status(f"⚛️ Simulating with {model_name}...", expanded=True) as status:
            st.write("Calculating Physics Vectors...")
            final_code, error, in_txt, out_txt = generate_with_retry(user_instruction, api_key, base_url, model_name)
            
            if final_code:
                st.write("Compiling Python...")
                st.session_state["generated_code"] = final_code
                c_in, c_out, t_in, t_out = estimate_cost(in_txt, out_txt, model_name)
                st.session_state["last_cost_data"] = (c_in, c_out, t_in, t_out)
                status.update(label="Simulation Ready!", state="complete", expanded=False)
                st.rerun()
            else:
                status.update(label="Generation Failed", state="error")
                st.error(f"Error: {error}")

    if st.session_state["generated_code"]:

        # --- Download / Code View ---
        with st.expander("View Source Code & Download"):
            c1, c2 = st.columns([1, 5])
            with c1:
                st.download_button("📥 Download .py", st.session_state["generated_code"], "simulation.py", "text/x-python")
            with c2:
                st.code(st.session_state["generated_code"], language='python')

        # --- Render Simulation ---
        try:
            exec_globals = {"st": st, "np": np, "go": go, "__name__": "__main__"}
            exec(st.session_state["generated_code"], exec_globals)
            
            if "fig" in exec_globals:
                fig = exec_globals["fig"]
                
                # --- LAYOUT POLISH (Inspired by notes) ---
                fig.update_layout(
                    height=850,
                    margin=dict(l=0, r=0, t=0, b=0),
                    uirevision="Don't Reset",
                    scene=dict(uirevision="Don't Reset"),
                    autosize=True # Responsive
                )

                # --- ROBUST UI FIXES ---
                if fig.layout.updatemenus:
                    # TOP LEFT BUTTONS
                    fig.update_layout(
                        updatemenus=[
                            dict(
                                type="buttons",
                                direction="left",
                                x=0.0, y=1.0, 
                                xanchor="left", yanchor="top",
                                showactive=True,
                                bgcolor="white",
                                bordercolor="#333",
                                borderwidth=1,
                                font=dict(color="black", size=12),
                                pad={"r": 10, "t": 10},
                                buttons=fig.layout.updatemenus[0].buttons
                            )
                        ]
                    )

                    # SPEED CONTROL
                    for button in fig.layout.updatemenus[0].buttons:
                        if button.label == 'Play':
                            if hasattr(button, 'args') and len(button.args) > 1:
                                arg_dict = button.args[1]
                                if 'frame' not in arg_dict: arg_dict['frame'] = {}
                                if 'transition' not in arg_dict: arg_dict['transition'] = {}
                                
                                arg_dict['frame']['duration'] = frame_duration
                                arg_dict['transition']['duration'] = 0

                # --- RENDER WITH CLEAN CONFIG ---
                st.plotly_chart(
                    fig, 
                    use_container_width=True, 
                    height=850, 
                    key=f"sim_chart_{speed_factor}",
                    config={
                        'displayModeBar': True, 
                        'scrollZoom': True,
                        'displaylogo': False, # Cleaner look
                        'modeBarButtonsToRemove': ['lasso2d', 'select2d'], # Cleanup
                        'modeBarButtonsToAdd': ['zoomIn3d', 'zoomOut3d']
                    }
                )
                
            else:
                st.error("Code executed but `fig` variable was not defined.")
        except Exception as e:
            st.error(f"Runtime Error: {e}")

if __name__ == "__main__":
    if check_password():
        main_app()
