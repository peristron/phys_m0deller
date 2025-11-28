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

# --- Constants for Cost Estimation ---
# Estimated pricing per 1M tokens (Input / Output)
PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "grok-beta": {"input": 5.00, "output": 15.00} # Conservative estimate
}

# --- Helper: Safe Secret Retrieval ---
def get_secret(key_name):
    """
    Tries to fetch a secret from st.secrets in a case-insensitive way.
    Returns None if not found.
    """
    # 1. Try exact match
    if key_name in st.secrets:
        return st.secrets[key_name]
    # 2. Try all uppercase
    if key_name.upper() in st.secrets:
        return st.secrets[key_name.upper()]
    # 3. Try all lowercase
    if key_name.lower() in st.secrets:
        return st.secrets[key_name.lower()]
    return None

# --- Authentication Logic ---
def check_password():
    """Returns `True` if the user had the correct password."""
    
    # Retrieve the password from secrets safely
    stored_password = get_secret("app_password")
    
    if not stored_password:
        st.error("❌ Configuration Error: 'app_password' not found in Streamlit Secrets.")
        st.stop()

    def password_entered():
        if st.session_state["password"] == stored_password:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Please enter the App Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Please enter the App Password", type="password", on_change=password_entered, key="password")
        st.error("😕 Password incorrect")
        return False
    else:
        return True

# --- Core Application Logic ---
def main_app():
    st.title("⚛️ Generative Physics Modeler")
    st.markdown("Enter a natural language description, and the AI will code a dynamic 3D simulation for you.")

    # --- Sidebar: API Setup ---
    st.sidebar.header("Configuration")
    provider = st.sidebar.radio("Select Provider", ["OpenAI", "xAI (Grok)"])

    api_key = None
    base_url = None
    model_name = ""
    
    # Safe retrieval of API keys
    if provider == "OpenAI":
        api_key = get_secret("openai_api_key")
        if api_key:
            model_name = "gpt-4o" 
        else:
            st.sidebar.error("Missing 'openai_api_key' in secrets.")
            
    elif provider == "xAI (Grok)":
        api_key = get_secret("xai_api_key")
        if api_key:
            base_url = "https://api.x.ai/v1"
            model_name = "grok-beta"
        else:
            st.sidebar.error("Missing 'xai_api_key' in secrets.")

    # --- Helper Functions ---
    def clean_code(code):
        code = re.sub(r'^```python', '', code)
        code = re.sub(r'^```', '', code)
        code = re.sub(r'```$', '', code)
        return code.strip()

    def estimate_cost(input_text, output_text, model):
        # Rough estimation: 1 token ~= 4 characters
        in_tok = len(input_text) / 4
        out_tok = len(output_text) / 4
        rates = PRICING.get(model, {"input": 2.50, "output": 10.00})
        
        cost_in = (in_tok / 1_000_000) * rates["input"]
        cost_out = (out_tok / 1_000_000) * rates["output"]
        
        return cost_in, cost_out, in_tok, out_tok

    def get_system_prompt():
        return """
        You are a Python Code Generator for a 3D Physics Visualization app.
        
        YOUR GOAL:
        Convert the user's physics description into a Python script that generates a 3D Plotly Animation.
        
        STRICT CONSTRAINTS:
        1. Use ONLY these libraries: `numpy` (as np), `plotly.graph_objects` (as go), `streamlit` (as st).
        2. Do NOT create a 'while True' loop. Streamlit cannot handle infinite loops.
        3. Pre-calculate 40-60 frames of data for the animation.
        4. The code MUST define a figure `fig`. In `fig.layout.updatemenus`, set the 'buttons' to trigger the animation automatically if possible, or make the button prominent.
        5. The code MUST end by calling `st.plotly_chart(fig, use_container_width=True)`.
        6. Adhere to physics principles (gravity, momentum, gas laws) using NumPy for vector math.
        7. Do NOT use markdown blocks (```python). Output RAW CODE only.
        8. Visual style: Use Wireframes (mode='lines') or Molecules (mode='markers').
        9. Ensure all arrays used in calculations are initialized as floats (e.g., np.zeros(..., dtype=float)).
        """

    def call_llm(messages, api_key, base_url, model):
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.5
            )
            return response.choices[0].message.content
        except Exception as e:
            raise e

    # --- Self-Healing Generation Logic ---
    def generate_with_retry(user_prompt, api_key, base_url, model, max_retries=2):
        """
        Generates code, attempts to exec it. 
        If exec fails, feeds error back to LLM to fix.
        """
        
        # 1. Initial Generation
        system_prompt = get_system_prompt()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Create a 3D simulation for: {user_prompt}"}
        ]
        
        raw_code = call_llm(messages, api_key, base_url, model)
        code = clean_code(raw_code)
        
        # Track total text for cost estimation
        total_input_text = system_prompt + user_prompt
        total_output_text = raw_code
        
        # 2. Validation Loop
        for attempt in range(max_retries + 1):
            try:
                # Syntax check via compilation
                compile(code, '<string>', 'exec')
                return code, None, total_input_text, total_output_text
                
            except Exception as e:
                if attempt < max_retries:
                    st.warning(f"⚠️ Attempt {attempt+1} produced invalid code. Auto-correcting...")
                    
                    error_prompt = f"""
                    The code you provided threw this error:
                    {str(e)}
                    
                    Fix the code and return ONLY the valid Python code.
                    """
                    
                    messages.append({"role": "assistant", "content": code})
                    messages.append({"role": "user", "content": error_prompt})
                    
                    raw_code = call_llm(messages, api_key, base_url, model)
                    code = clean_code(raw_code)
                    
                    total_input_text += error_prompt
                    total_output_text += raw_code
                else:
                    return None, str(e), total_input_text, total_output_text

    # --- Interface ---
    col1, col2 = st.columns([1, 2])

    with col1:
        user_instruction = st.text_area(
            "Describe the Physics Model", 
            height=150,
            value="A wireframe sphere rotating on the Z-axis with 20 gas molecules bouncing around it inside a cubic container."
        )
        
        if api_key:
            generate_btn = st.button("Generate Simulation", type="primary")
        else:
            st.warning("API Key missing. Check sidebar configuration.")
            generate_btn = False

        # --- Cost Estimator UI ---
        st.divider()
        if "last_cost_data" in st.session_state:
            c_in, c_out, t_in, t_out = st.session_state["last_cost_data"]
            total_cost = c_in + c_out
            
            st.subheader("💰 Estimated Cost")
            st.caption(f"Model: {model_name}")
            
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.metric("Input", f"${c_in:.4f}", help=f"~{int(t_in)} Tokens")
            with m_col2:
                st.metric("Output", f"${c_out:.4f}", help=f"~{int(t_out)} Tokens")
            
            st.success(f"**Total: ${total_cost:.4f}**")

    with col2:
        # Initialize session state for code persistence
        if "generated_code" not in st.session_state:
            st.session_state["generated_code"] = None

        # Handle Generation
        if generate_btn:
            with st.spinner(f"Calculating Physics & Generating Code ({model_name})..."):
                # Run the self-healing generator
                final_code, error, in_txt, out_txt = generate_with_retry(user_instruction, api_key, base_url, model_name)
                
                if final_code:
                    st.session_state["generated_code"] = final_code
                    
                    # Calculate Costs
                    c_in, c_out, t_in, t_out = estimate_cost(in_txt, out_txt, model_name)
                    st.session_state["last_cost_data"] = (c_in, c_out, t_in, t_out)
                    
                    st.rerun() # Rerun to update the UI with code and cost
                else:
                    st.error(f"Failed to generate valid code after retries.\nLast Error: {error}")

        # Display Results (if code exists in state)
        if st.session_state["generated_code"]:
            
            # 1. Display Code Expander
            with st.expander("View Generated Python Code", expanded=False):
                st.code(st.session_state["generated_code"], language='python')
            
            # 2. Download Button
            st.download_button(
                label="📥 Download Script",
                data=st.session_state["generated_code"],
                file_name="simulation.py",
                mime="text/x-python"
            )

            # 3. Execute the Code
            try:
                exec_globals = {
                    "st": st,
                    "np": np,
                    "go": go,
                    "__name__": "__main__"
                }
                exec(st.session_state["generated_code"], exec_globals)
            except Exception as e:
                st.error(f"Runtime Error during rendering: {e}")

# --- Entry Point ---
if __name__ == "__main__":
    if check_password():
        main_app()

