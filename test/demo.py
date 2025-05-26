import streamlit as st
import requests
from typing import Optional
import json

# --- RAG Chatbot Client Class (from your script, minimal edit for import compatibility) ---
class RagChatbotClient:
    def __init__(self, base_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})

    def health_check(self) -> dict:
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def create_session(self) -> str:
        response = self.session.post(f"{self.base_url}/sessions")
        response.raise_for_status()
        return response.json()["session_id"]

    def chat(self, message: str, session_id: Optional[str] = None, mode: str = "original") -> dict:
        payload = {
            "message": message,
            "mode": mode
        }
        if session_id:
            payload["session_id"] = session_id
        response = self.session.post(f"{self.base_url}/chat", json=payload)
        response.raise_for_status()
        return response.json()

    def get_conversation_history(self, session_id: str) -> dict:
        response = self.session.get(f"{self.base_url}/sessions/{session_id}/history")
        response.raise_for_status()
        return response.json()

    def delete_session(self, session_id: str) -> dict:
        response = self.session.delete(f"{self.base_url}/sessions/{session_id}")
        response.raise_for_status()
        return response.json()

    def get_session_stats(self) -> dict:
        response = self.session.get(f"{self.base_url}/sessions/stats")
        response.raise_for_status()
        return response.json()

# --- Streamlit App ---

st.set_page_config(page_title="RAG Chatbot Demo", page_icon="🤖", layout="centered")

st.title("🤖 RAG Chatbot API Demo")
st.markdown(
    """
    Interact with your RAG Chatbot API in real time.<br>
    _Fill API settings, click "Connect", then chat below!_
    """, unsafe_allow_html=True)

# Sidebar: API setup
with st.sidebar:
    st.header("API Connection")
    api_url = st.text_input("API Base URL", value="http://localhost:8000")
    api_key = st.text_input("API Key (optional)", type="password")
    connect_btn = st.button("Connect")

# State for client and session
if 'client' not in st.session_state or st.session_state.get('reset', False):
    st.session_state['client'] = None
    st.session_state['session_id'] = None
    st.session_state['chat_history'] = []
    st.session_state['reset'] = False

# On Connect
if connect_btn:
    try:
        client = RagChatbotClient(base_url=api_url, api_key=api_key if api_key else None)
        health = client.health_check()
        st.success(f"API Health: {health['status']}")
        session_id = client.create_session()
        st.session_state['client'] = client
        st.session_state['session_id'] = session_id
        st.session_state['chat_history'] = []
        st.info(f"New session started: {session_id}")
    except Exception as e:
        st.error(f"API Connection failed: {e}")

# Chat UI only if connected
if st.session_state.get('client') and st.session_state.get('session_id'):
    with st.expander("Show Conversation History", expanded=False):
        if st.session_state['chat_history']:
            for entry in st.session_state['chat_history']:
                st.markdown(f"**You:** {entry['user']}")
                st.markdown(f"**Bot:** {entry['bot']}")
                if entry.get('metadata'):
                    with st.expander("Metadata", expanded=False):
                        st.json(entry['metadata'])
        else:
            st.caption("No conversation yet.")

    st.markdown("**Type your message:**")
    user_input = st.text_input("Message", key="user_input", placeholder="Ask your question here...", label_visibility='collapsed')

    col1, col2 = st.columns([1, 4])
    with col1:
        send_btn = st.button("Send", use_container_width=True)
    with col2:
        reset_btn = st.button("Start New Session", use_container_width=True)

    # On send
    if send_btn and user_input.strip():
        client = st.session_state['client']
        session_id = st.session_state['session_id']
        try:
            response = client.chat(message=user_input, session_id=session_id)
            st.session_state['chat_history'].append({
                "user": user_input,
                "bot": response.get('response', ''),
                "metadata": response.get('metadata', {})
            })
            st.success("Bot responded!")
        except Exception as e:
            st.error(f"Failed to get response: {e}")

    # On reset
    if reset_btn:
        try:
            client = st.session_state['client']
            if st.session_state.get('session_id'):
                client.delete_session(st.session_state['session_id'])
        except Exception:
            pass  # ignore errors
        st.session_state['reset'] = True
        st.experimental_rerun()

    # Show latest exchange (streamlit-style chat bubbles)
    if st.session_state['chat_history']:
        st.markdown("---")
        for i, chat in enumerate(st.session_state['chat_history'][-5:]):  # Last 5 messages
            st.markdown(f"**You:** {chat['user']}")
            st.markdown(f"**🤖 Bot:** {chat['bot']}")
            if chat.get('metadata'):
                with st.expander("Metadata", expanded=False):
                    st.json(chat['metadata'])

# Footer
st.markdown("""<hr><small>
Made for RAG Chatbot API testing. <br>
Streamlit quick demo by ChatGPT.
</small>""", unsafe_allow_html=True)
