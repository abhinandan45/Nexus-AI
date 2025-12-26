import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

load_dotenv()

DB_PATH = "full_nexus_ai_index"
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")

class NexusBot:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        if not os.path.exists(DB_PATH):
            raise Exception("Index not found!")
        self.db = FAISS.load_local(DB_PATH, self.embeddings, allow_dangerous_deserialization=True)
        
        # Temperature 0.0 for absolute precision
        self.llm = ChatOpenAI(
            model="deepseek/deepseek-chat",
            openai_api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            temperature=0.0, 
            max_tokens=1500
        )

    def ask(self, query, chat_history):
        docs = self.db.similarity_search(query, k=3)
        context_text = "\n\n".join([doc.page_content for doc in docs])
        
        formatted_history = ""
        for msg in chat_history[-5:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted_history += f"{role}: {msg['content']}\n"

        # --- THE NATURAL EXPERT PROMPT (No "Based on context" talk) ---
        full_prompt = f"""
        System: You are 'NEXUS AI', a world-class Technical Expert. 
        Your goal is to provide fluid, natural, and highly intelligent responses.

        <<< BEHAVIORAL PROTOCOL >>>
        - NEVER mention phrases like "Based on the provided context", "According to the documents", or "In the text". 
        - Act as if this knowledge is YOUR OWN. Answer naturally like ChatGPT or Gemini.
        - If the user's query is technical or coding-related, use the provided [KNOWLEDGE BASE] and your internal logic to give a pro-level answer.
        - If the query is completely non-technical (celebs, sports, etc.), politely decline by saying you are a specialized technical engine.
        - Stay focused, sharp, and structured.

        <<< RESPONSE STYLE >>>
        - Use ### for clear headings.
        - Use **bold** for key concepts.
        - Use clean code blocks (```python) for any code requests.
        - Always end with a "Key Takeaway".

        [CHAT HISTORY]
        {formatted_history}

        [KNOWLEDGE BASE]
        {context_text}

        [USER QUERY]
        {query}

        [NEXUS AI RESPONSE]:
        """
        
        response = self.llm.invoke(full_prompt)
        return {"answer": response.content, "sources": docs}

def get_qa_chain():
    return NexusBot()