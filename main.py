import streamlit as st
from Data_Science_Agent.UserInterface.Sidebar import SidebarUI  

def main():
    sidebar_ui = SidebarUI()
    user_input = sidebar_ui.Load_UI()

    st.write("Current User Controls:")
    st.json(user_input)

if __name__ == "__main__":
    main()
