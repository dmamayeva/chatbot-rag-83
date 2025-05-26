"""
Example client to demonstrate how to use the RAG Chatbot API
"""

import requests
import json
from typing import Optional

class RagChatbotClient:
    """Client for interacting with the RAG Chatbot API"""
    
    def __init__(self, base_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})
    
    def health_check(self) -> dict:
        """Check if the API is healthy"""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def create_session(self) -> str:
        """Create a new chat session"""
        response = self.session.post(f"{self.base_url}/sessions")
        response.raise_for_status()
        return response.json()["session_id"]
    
    def chat(self, message: str, session_id: Optional[str] = None, mode: str = "original") -> dict:
        """Send a chat message"""
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
        """Get conversation history for a session"""
        response = self.session.get(f"{self.base_url}/sessions/{session_id}/history")
        response.raise_for_status()
        return response.json()
    
    def delete_session(self, session_id: str) -> dict:
        """Delete a session"""
        response = self.session.delete(f"{self.base_url}/sessions/{session_id}")
        response.raise_for_status()
        return response.json()
    
    def get_session_stats(self) -> dict:
        """Get session statistics"""
        response = self.session.get(f"{self.base_url}/sessions/stats")
        response.raise_for_status()
        return response.json()

def main():
    """Example usage of the RAG Chatbot API"""
    
    # Initialize client
    client = RagChatbotClient(
        base_url="http://localhost:8000",
        api_key="your-secret-api-key"  # Set to None if no API key required
    )
    
    try:
        # Check health
        health = client.health_check()
        print(f"✅ API Health: {health['status']}")
        
        # Create a new session
        session_id = client.create_session()
        print(f"📝 Created session: {session_id}")
        
        # Example conversation
        questions = ['Какое решение принимает аттестационная комиссия, если отсутствует один из критериев на заявляемую категорию?',
 'Можно ли сохранить действующую квалификационную категорию руководителю и заместителю руководителя за 4 года до выхода на пенсию?',
 'Обязан ли педагог проходить ОЗП, если у него стаж более 30 лет?',
 'Если в дипломе педагога два предмета, по какому из них он должен проходить аттестацию?',
 'Сохраняется ли квалификационная категория педагога, если он переходит из дошкольной организации в организацию  среднего образования?',
 'Какой пороговый уровень ОЗП?',
 'Сколько баллов нужно педагогу-модератору для прохождения квалификационного теста?']
        
        print("\n🤖 Starting conversation...")
        print("=" * 50)
        
        for i, question in enumerate(questions, 1):
            print(f"\n👤 User: {question}")
            
            # Send message
            response = client.chat(message=question, session_id=session_id)
            
            print(f"🤖 Bot: {response['response']}")
            print(f"📊 Message #{response['message_count']} | Session: {response['session_id'][:8]}...")
            
            if response.get('metadata'):
                print(f"📈 Metadata: {json.dumps(response['metadata'], indent=2)}")
        
        # Get conversation history
        print(f"\n📜 Getting conversation history...")
        history = client.get_conversation_history(session_id)
        print(f"Total messages: {history['message_count']}")
        
        # Get session stats
        stats = client.get_session_stats()
        print(f"\n📊 Active sessions: {stats['active_sessions']}")
        
        # Clean up
        client.delete_session(session_id)
        print(f"🗑️ Deleted session: {session_id}")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ API Error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()