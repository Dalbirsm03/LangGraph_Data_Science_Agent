import streamlit as st
from Data_Science_Agent.UserInterface.config import Config

class SidebarUI:
    def __init__(self):
        self.user_controls = {}
        self.config = Config()

    def Load_UI(self):
        use_case = st.sidebar.selectbox("Select UseCase", self.config.get_usecase_options())
        self.user_controls["usecase"] = use_case
        # LLM Selection
        llm = st.sidebar.selectbox("Select LLM", self.config.get_llms())
        self.user_controls["llm"] = llm

        # Dynamic model options
        if llm == "Google Gemini":
            model = st.sidebar.selectbox("Select Gemini Model", self.config.get_gemini_llm())
        elif llm == "Groq":
            model = st.sidebar.selectbox("Select Groq Model", self.config.get_groq_model_options())
        else:
            model = None

        self.user_controls["model"] = model

        # API key input
        api_key = st.sidebar.text_input(f"Enter {llm} API Key", type="password")
        self.user_controls["api_key"] = api_key


        # Mode selection
        mode = st.sidebar.radio("Choose Data Source", ["Upload File", "Connect SQL"])
        self.user_controls["mode"] = mode

        if mode == "Upload File":
            uploaded_file = st.sidebar.file_uploader("Upload your dataset", type=["csv", "xlsx", "json"], accept_multiple_files = True)
            self.user_controls["file"] = uploaded_file

        elif mode == "Connect SQL":
            st.sidebar.markdown("### 🔐 SQL Credentials")
            self.user_controls["sql_config"] = {
                "user": st.sidebar.text_input("Username"),
                "password": st.sidebar.text_input("Password", type="password"),
                "host": st.sidebar.text_input("Host"),
                "port": st.sidebar.text_input("Port", value="3306"),
                "database": st.sidebar.text_input("Database"),
            }


        return self.user_controls