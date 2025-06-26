import streamlit as st
import requests
import json
from typing import Optional, Union
import base64
import io
from datetime import datetime

# Configuration
CHATBOT_CONFIG = {
    "api_base_url": "http://localhost:8000", 
    "api_key": "",  
    "title": "AI-ассистент \"Ұстаз\"",
    "welcome_message": "Здравствуйте! Я AI-ассистент разработанный АО \"Өрлеу\". Как я могу помочь вам сегодня?"
}

def get_headers() -> dict:
    """Get headers for API requests"""
    headers = {
        'Content-Type': 'application/json'
    }
    # Only add Authorization header if API key is provided
    if CHATBOT_CONFIG["api_key"] and CHATBOT_CONFIG["api_key"].strip():
        headers['Authorization'] = f'Bearer {CHATBOT_CONFIG["api_key"]}'
    return headers

def create_session() -> Optional[str]:
    """Create a new chat session"""
    try:
        response = requests.post(
            f"{CHATBOT_CONFIG['api_base_url']}/api/v1/sessions",
            headers=get_headers(),
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            return data.get("session_id")
        else:
            st.error(f"Failed to create session: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {str(e)}")
        return None

def send_message(message: str, session_id: str) -> Optional[Union[dict, bytes]]:
    """Send message to chatbot API and handle both JSON and file responses"""
    try:
        response = requests.post(
            f"{CHATBOT_CONFIG['api_base_url']}/api/v1/chat/",
            headers=get_headers(),
            json={
                "message": message,
                "session_id": session_id,
                "mode": "generated"
            },
            timeout=30
        )
        
        if response.status_code == 200:
            # Check content type to determine response type
            content_type = response.headers.get('content-type', '')
            
            if 'application/pdf' in content_type:
                # Handle PDF file response
                pdf_data = response.content
                
                # Extract metadata from headers with safe decoding
                def safe_decode_header(header_value: str, is_b64_encoded: bool = False) -> str:
                    """Safely decode header value"""
                    if not header_value or header_value == 'Unknown':
                        return 'Unknown Document'
                    
                    if is_b64_encoded:
                        try:
                            import base64
                            decoded_bytes = base64.b64decode(header_value.encode('ascii'))
                            return decoded_bytes.decode('utf-8')
                        except:
                            return header_value
                    return header_value
                
                # Check if document name is base64 encoded
                is_original = response.headers.get('X-Document-Name-Original', 'true') == 'true'
                document_name_header = response.headers.get('X-Document-Name-B64', 'Unknown Document')
                document_name = safe_decode_header(document_name_header, not is_original)
                
                match_score = response.headers.get('X-Match-Score', 'N/A')
                match_type_header = response.headers.get('X-Match-Type', 'N/A')
                match_type = safe_decode_header(match_type_header, not match_type_header.isascii() if match_type_header else False)
                
                # Get filename from Content-Disposition header or use default
                filename = 'document.pdf'
                content_disposition = response.headers.get('content-disposition', '')
                if 'filename=' in content_disposition:
                    filename = content_disposition.split('filename=')[1].strip('"')
                
                return {
                    'type': 'pdf',
                    'data': pdf_data,
                    'filename': filename,
                    'document_name': document_name,
                    'match_score': match_score,
                    'match_type': match_type
                }
            else:
                # Handle JSON response
                return response.json()
        else:
            error_data = {}
            try:
                error_data = response.json()
            except:
                pass
            
            error_message = error_data.get('detail', f'HTTP error {response.status_code}')
            st.error(f"API Error: {error_message}")
            return None
            
    except requests.exceptions.Timeout:
        st.error("Request timed out. Please try again.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {str(e)}")
        return None

def display_pdf_in_streamlit(pdf_data: bytes, filename: str) -> None:
    """Display PDF in Streamlit using an iframe"""
    # Convert PDF to base64 for embedding
    b64_pdf = base64.b64encode(pdf_data).decode()
    
    # Create HTML for PDF viewer
    pdf_display = f"""
    <iframe src="data:application/pdf;base64,{b64_pdf}" 
            width="100%" 
            height="600" 
            type="application/pdf">
        <p>Your browser does not support PDFs. 
           <a href="data:application/pdf;base64,{b64_pdf}" download="{filename}">
           Download the PDF</a>
        </p>
    </iframe>
    """
    
    st.markdown(pdf_display, unsafe_allow_html=True)

def create_download_link(pdf_data: bytes, filename: str) -> str:
    """Create a download link for the PDF"""
    b64_pdf = base64.b64encode(pdf_data).decode()
    return f'<a href="data:application/pdf;base64,{b64_pdf}" download="{filename}">📥 Скачать {filename}</a>'

def main():
    # Page configuration
    st.set_page_config(
        page_title=CHATBOT_CONFIG["title"],
        page_icon="🤖",
        layout="wide"  # Changed to wide for better PDF viewing
    )
    
    # Title
    st.title(CHATBOT_CONFIG["title"])
    st.markdown("---")
    
    # Initialize session state
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": CHATBOT_CONFIG["welcome_message"], "type": "text"}
        ]
    
    if "session_created" not in st.session_state:
        st.session_state.session_created = False
    
    # Create session if not already created
    if not st.session_state.session_created:
        with st.spinner("Создание сессии..."):
            session_id = create_session()
            if session_id:
                st.session_state.session_id = session_id
                st.session_state.session_created = True
                st.success("Сессия создана успешно!")
            else:
                st.error("Не удалось создать сессию. Попробуйте обновить страницу.")
                st.stop()
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message.get("type") == "pdf":
                # Display PDF message
                st.markdown(f"📄 **Документ найден:** {message['document_name']}")
                st.markdown(f"🎯 **Соответствие:** {message['match_type']} (оценка: {message['match_score']})")
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown("**Предварительный просмотр:**")
                    display_pdf_in_streamlit(message["pdf_data"], message["filename"])
                
                with col2:
                    st.markdown("**Действия:**")
                    download_link = create_download_link(message["pdf_data"], message["filename"])
                    st.markdown(download_link, unsafe_allow_html=True)
                    
                    # Display file info
                    st.markdown("**Информация о файле:**")
                    st.text(f"Имя: {message['filename']}")
                    st.text(f"Размер: {len(message['pdf_data']) / 1024:.1f} KB")
                    st.text(f"Время: {message.get('timestamp', 'N/A')}")
            else:
                st.markdown(message["content"])
    
    if prompt := st.chat_input("Введите ваше сообщение..."):
    
        if not st.session_state.session_id:
            st.error("Сессия не создана. Обновите страницу.")
            st.stop()
        
        st.session_state.messages.append({
            "role": "user", 
            "content": prompt, 
            "type": "text"
        })
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Ищу ответ... Пожалуйста, подождите."):
                response_data = send_message(prompt, st.session_state.session_id)
                
                if response_data:
                    
                    if isinstance(response_data, dict) and response_data.get("session_id"):
                        st.session_state.session_id = response_data["session_id"]
                    
                    if isinstance(response_data, dict) and response_data.get("type") == "pdf":
                    
                        st.markdown(f"📄 **Документ найден:** {response_data['document_name']}")
                        st.markdown(f"🎯 **Соответствие:** {response_data['match_type']} (оценка: {response_data['match_score']})")
                        
                    
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.markdown("**Предварительный просмотр:**")
                            display_pdf_in_streamlit(response_data["data"], response_data["filename"])
                        
                        with col2:
                            st.markdown("**Действия:**")
                            download_link = create_download_link(response_data["data"], response_data["filename"])
                            st.markdown(download_link, unsafe_allow_html=True)
                            
                            # Display file info
                            st.markdown("**Информация о файле:**")
                            st.text(f"Имя: {response_data['filename']}")
                            st.text(f"Размер: {len(response_data['data']) / 1024:.1f} KB")
                        
                    
                        st.session_state.messages.append({
                            "role": "assistant",
                            "type": "pdf",
                            "pdf_data": response_data["data"],
                            "filename": response_data["filename"],
                            "document_name": response_data["document_name"],
                            "match_score": response_data["match_score"],
                            "match_type": response_data["match_type"],
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
                    
                    else:
                    
                        bot_response = response_data.get("response", "Извините, не удалось получить ответ.")
                        st.markdown(bot_response)
                        
                    
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": bot_response, 
                            "type": "text"
                        })
                else:
                    error_message = "Извините, произошла ошибка. Попробуйте еще раз."
                    st.markdown(error_message)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_message, 
                        "type": "text"
                    })
    

    with st.sidebar:
        st.subheader("Информация о сессии")
        if st.session_state.session_id:
            st.success(f"Session ID: {st.session_state.session_id[:8]}...")
        else:
            st.error("Сессия не создана")
        
        st.subheader("Настройки API")
        st.code(f"API URL: {CHATBOT_CONFIG['api_base_url']}")
        

        st.subheader("Статистика чата")
        total_messages = len(st.session_state.messages) - 1 
        pdf_messages = len([msg for msg in st.session_state.messages if msg.get("type") == "pdf"])
        text_messages = total_messages - pdf_messages
        
        st.metric("Всего сообщений", total_messages)
        st.metric("Текстовых ответов", text_messages)
        st.metric("PDF документов", pdf_messages)
        
        
        if st.button("🗑️ Очистить чат", type="secondary"):
            st.session_state.messages = [
                {"role": "assistant", "content": CHATBOT_CONFIG["welcome_message"], "type": "text"}
            ]
            st.session_state.session_id = None
            st.session_state.session_created = False
            st.rerun()
        
        
        if st.button("🔄 Новая сессия", type="secondary"):
            st.session_state.session_id = None
            st.session_state.session_created = False
            st.rerun()
        
        
        st.subheader("Помощь")
        st.markdown("""
        **Как использовать:**
        - Задавайте вопросы в текстовом поле
        - Если найден документ, он будет показан с возможностью скачивания
        - Используйте кнопки для управления сессией
        
        **Поддерживаемые форматы:**
        - 📄 PDF документы
        - 💬 Текстовые ответы
        """)

if __name__ == "__main__":
    main()