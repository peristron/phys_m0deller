# streamlit run phys_modeller.py
#  directory setup: cd C:\users\oakhtar\documents\pyprojs_local


import streamlit as st
import openai
import numpy as np
import plotly.graph_objects as go
import re

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="GenAI Physics Modeler")

# --- Authentication Logic ---
def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["app_password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Please enter the App Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Please enter the App Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
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
    
    # Load keys from secrets based on selection
    try:
        if provider == "OpenAI":
            if "openai_api_key" in st.secrets:
                api_key = st.secrets["openai_api_key"]
                model_name = "gpt-4o" 
            else:
                st.sidebar.error("Missing 'openai_api_key' in secrets.")
                
        elif provider == "xAI (Grok)":
            if "xai_api_key" in st.secrets:
                api_key = st.secrets["xai_api_key"]
                base_url = "https://api.x.ai/v1"
                model_name = "grok-beta"
            else:
                st.sidebar.error("Missing 'xai_api_key' in secrets.")
    except FileNotFoundError:
        st.error("Secrets file not found. Please set up your .streamlit/secrets.toml")

    # --- LLM Logic ---
    def get_system_prompt():
        return """
        You are a Python Code Generator for a 3D Physics Visualization app.
        
        YOUR GOAL:
        Convert the user's physics description into a Python script that generates a 3D Plotly Animation.
        
        STRICT CONSTRAINTS:
        1. Use ONLY these libraries: `numpy` (as np), `plotly.graph_objects` (as go), `streamlit` (as st).
        2. Do NOT create a 'while True' loop. Streamlit cannot handle infinite loops.
        3. Pre-calculate 40-60 frames of data for the animation.
        4. The code MUST define a figure `fig` using `go.Frame` for animation.
        5. The code MUST end by calling `st.plotly_chart(fig, use_container_width=True)`.
        6. Adhere to physics principles (gravity, momentum, gas laws) using NumPy for vector math.
        7. Do NOT use markdown blocks (```python). Output RAW CODE only.
        8. Visual style: Use Wireframes (mode='lines') or Molecules (mode='markers').
        9. Ensure all arrays used in calculations are initialized as floats (e.g., np.zeros(..., dtype=float)).
        """

    def generate_simulation_code(user_prompt, api_key, base_url, model):
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        
        messages = [
            {"role": "system", "content": get_system_prompt()},
            {"role": "user", "content": f"Create a 3D simulation for: {user_prompt}"}
        ]
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.5
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"

    def clean_code(code):
        code = re.sub(r'^```python', '', code)
        code = re.sub(r'^```', '', code)
        code = re.sub(r'```$', '', code)
        return code.strip()

    # --- Interface ---
    col1, col2 = st.columns([1, 2])

    with col1:
        user_instruction = st.text_area(
            "Describe the Physics Model", 
            height=150,
            value="A wireframe sphere rotating on the Z-axis with 20 gas molecules bouncing around it inside a cubic container."
        )
        
        # Only enable button if we successfully loaded an API key
        if api_key:
            generate_btn = st.button("Generate Simulation", type="primary")
        else:
            st.warning("API Key missing. Check sidebar configuration.")
            generate_btn = False

    with col2:
        if generate_btn:
            with st.spinner(f"Calculating Physics using {model_name}..."):
                raw_code = generate_simulation_code(user_instruction, api_key, base_url, model_name)
                executable_code = clean_code(raw_code)
                
                with st.expander("View Generated Python Code"):
                    st.code(executable_code, language='python')
                
                try:
                    # Create a restricted globals dictionary to safeguard exec slightly
                    # We pass explicit libraries to avoid 'module not found' issues
                    exec_globals = {
                        "st": st,
                        "np": np,
                        "go": go,
                        "__name__": "__main__"
                    }
                    exec(executable_code, exec_globals)
                except Exception as e:
                    st.error(f"Code Execution Error: {e}")
                    st.error("The LLM generated invalid code. Please try modifying your prompt.")

# --- Entry Point ---
if __name__ == "__main__":
    if check_password():
        main_app()