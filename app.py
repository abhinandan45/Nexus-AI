import streamlit as st
from backend import get_qa_chain
import time
import os

# 1. Page Configuration
st.set_page_config(page_title="NEXUS AI", page_icon="🧠", layout="wide")

# 2. Premium Production Styling
st.markdown("""
    <style>
    @import url('[https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap](https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap)');
    
    html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #ffffff; }
    
    /* Custom Sidebar */
    [data-testid="stSidebar"] { 
        background-color: #f8fafc !important; 
        border-right: 1px solid #e2e8f0; 
    }

    /* Chat Bubbles Layout */
    .stChatMessage {
        max-width: 850px;
        margin: 0 auto 1.5rem auto !important;
        padding: 1.2rem !important;
        border-radius: 12px !important;
    }
    
    [data-testid="stChatMessageUser"] {
        background-color: #f1f5f9 !important;
        border: 1px solid #cbd5e1 !important;
    }
    
    [data-testid="stChatMessageAssistant"] {
        background-color: #ffffff !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        border-left: 6px solid #10b981 !important; /* Tech Green accent */
    }

    /* Code Block Dark Mode */
    pre {
        background-color: #0f172a !important;
        color: #f8fafc !important;
        padding: 1rem !important;
        border-radius: 8px !important;
    }
    
    h1 { font-weight: 700; color: #0f172a; text-align: center; }
    h3 { color: #10b981 !important; }
    .stChatInput { max-width: 850px; margin: 0 auto; }
    </style>
    """, unsafe_allow_html=True)

# 3. Sidebar
with st.sidebar:
    st.markdown("<h2 style='color: #0f172a;'>⚡ Nexus Core</h2>", unsafe_allow_html=True)
    st.caption("Status: Online | Memory: Enabled")
    st.divider()
    if st.button("🗑️ Reset Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.info("Ask about your PDF content or request custom code snippets.")

# 4. Main UI
st.markdown("<div style='padding: 10px;'><h1>NEXUS <span style='color: #10b981;'>AI</span></h1></div>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #64748b;'>Elite Technical RAG Engine for Developers</p>", unsafe_allow_html=True)
st.divider()

# 5. Initialize Engine
@st.cache_resource
def load_engine(): return get_qa_chain()
bot = load_engine()

if "messages" not in st.session_state:
    st.session_state.messages = []

# 6. Chat Display
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        avatar = "👤" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

# 7. Interaction & Response
if prompt := st.chat_input("Enter your query..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with chat_container:
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

    with chat_container:
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Generating Response..."):
                try:
                    # Pass prompt and history to backend
                    response_data = bot.ask(prompt, st.session_state.messages)
                    answer = response_data["answer"]
                    
                    # High-Speed Streaming logic
                    placeholder = st.empty()
                    full_res = ""
                    words = answer.split(" ")
                    for i in range(0, len(words), 5):
                        full_res += " ".join(words[i:i+5]) + " "
                        placeholder.markdown(full_res + "▌")
                        time.sleep(0.01)
                    placeholder.markdown(full_res)
                    
                    st.session_state.messages.append({"role": "assistant", "content": full_res})
                except Exception as e:
                    st.error(f"System Error: {e}")