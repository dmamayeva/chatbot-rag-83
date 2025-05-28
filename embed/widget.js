/**
 * RAG Chatbot Widget - Standalone Embedding Script
 * Add this script to any website to embed the chatbot
 */

(function(window, document) {
    'use strict';

    // Default configuration - can be overridden
    const DEFAULT_CONFIG = {
        apiBaseUrl: 'http://localhost:8000',
        apiKey: null, // Set if API key authentication is enabled
        theme: {
            primaryColor: '#007bff',
            backgroundColor: '#f8f9fa',
            textColor: '#333',
            botMessageColor: '#e9ecef',
            userMessageColor: '#007bff'
        },
        position: 'bottom-right', // 'bottom-right' or 'bottom-left'
        triggerText: '💬',
        title: 'RAG Assistant',
        subtitle: 'Ask me anything!',
        placeholder: 'Type your message...',
        welcomeMessage: 'Hello! I\'m your AI assistant. How can I help you today?',
        maxWidth: '350px',
        maxHeight: '500px'
    };

    // Global chatbot state
    let chatbotState = {
        sessionId: null,
        isOpen: false,
        messageHistory: [],
        config: DEFAULT_CONFIG
    };

    // Merge user config with defaults
    function mergeConfig(userConfig) {
        if (!userConfig) return DEFAULT_CONFIG;
        
        return {
            ...DEFAULT_CONFIG,
            ...userConfig,
            theme: {
                ...DEFAULT_CONFIG.theme,
                ...(userConfig.theme || {})
            }
        };
    }

    // Create chatbot styles
    function createStyles() {
        const style = document.createElement('style');
        style.id = 'rag-chatbot-styles';
        style.textContent = `
            @keyframes rag-chatbot-spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            .rag-chatbot-trigger:hover {
                transform: scale(1.1);
            }
            
            .rag-chatbot-close:hover {
                background: rgba(255,255,255,0.2) !important;
            }
            
            .rag-chatbot-messages::-webkit-scrollbar {
                width: 6px;
            }
            
            .rag-chatbot-messages::-webkit-scrollbar-track {
                background: #f1f1f1;
            }
            
            .rag-chatbot-messages::-webkit-scrollbar-thumb {
                background: #c1c1c1;
                border-radius: 3px;
            }

            .rag-chatbot-input:focus {
                border-color: ${chatbotState.config.theme.primaryColor} !important;
                box-shadow: 0 0 0 2px ${chatbotState.config.theme.primaryColor}33 !important;
            }

            .rag-chatbot-send:hover {
                opacity: 0.9;
            }

            .rag-chatbot-send:active {
                transform: scale(0.95);
            }

            @media (max-width: 768px) {
                .rag-chatbot-widget {
                    width: calc(100vw - 40px) !important;
                    height: calc(100vh - 140px) !important;
                    max-width: none !important;
                    max-height: none !important;
                }
            }
        `;
        document.head.appendChild(style);
    }

    // Create chatbot HTML structure
    function createChatbotHTML() {
        const config = chatbotState.config;
        const position = config.position === 'bottom-left' ? 'left: 20px;' : 'right: 20px;';
        
        const container = document.createElement('div');
        container.id = 'rag-chatbot-container';
        container.innerHTML = `
            <!-- Chatbot Trigger Button -->
            <div class="rag-chatbot-trigger" style="
                position: fixed;
                bottom: 20px;
                ${position}
                width: 60px;
                height: 60px;
                background: ${config.theme.primaryColor};
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                cursor: pointer;
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                z-index: 2147483647;
                transition: all 0.3s ease;
                font-size: 24px;
                color: white;
                border: none;
                user-select: none;
            ">
                ${config.triggerText}
            </div>

            <!-- Chatbot Widget -->
            <div class="rag-chatbot-widget" style="
                position: fixed;
                bottom: 90px;
                ${position}
                width: ${config.maxWidth};
                height: ${config.maxHeight};
                max-width: calc(100vw - 40px);
                max-height: calc(100vh - 140px);
                background: white;
                border-radius: 10px;
                box-shadow: 0 8px 25px rgba(0,0,0,0.15);
                z-index: 2147483647;
                display: none;
                flex-direction: column;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                border: 1px solid #e0e0e0;
            ">
                <!-- Header -->
                <div style="
                    background: ${config.theme.primaryColor};
                    color: white;
                    padding: 15px;
                    border-radius: 10px 10px 0 0;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                ">
                    <div>
                        <div style="font-weight: 600; font-size: 16px;">${config.title}</div>
                        <div style="font-size: 12px; opacity: 0.9;">${config.subtitle}</div>
                    </div>
                    <div class="rag-chatbot-close" style="
                        cursor: pointer;
                        font-size: 20px;
                        padding: 5px;
                        border-radius: 3px;
                        transition: background 0.2s;
                        user-select: none;
                        width: 30px;
                        height: 30px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                    ">×</div>
                </div>

                <!-- Messages Container -->
                <div class="rag-chatbot-messages" style="
                    flex: 1;
                    overflow-y: auto;
                    padding: 15px;
                    background: ${config.theme.backgroundColor};
                ">
                    <div class="rag-chatbot-message-bot" style="
                        background: ${config.theme.botMessageColor};
                        color: ${config.theme.textColor};
                        padding: 10px 12px;
                        border-radius: 18px 18px 18px 4px;
                        margin-bottom: 10px;
                        max-width: 80%;
                        font-size: 14px;
                        line-height: 1.4;
                        word-wrap: break-word;
                    ">
                        ${config.welcomeMessage}
                    </div>
                </div>

                <!-- Input Container -->
                <div style="
                    padding: 15px;
                    border-top: 1px solid #eee;
                    background: white;
                    border-radius: 0 0 10px 10px;
                ">
                    <div style="display: flex; gap: 10px; align-items: flex-end;">
                        <textarea 
                            class="rag-chatbot-input" 
                            placeholder="${config.placeholder}"
                            rows="1"
                            style="
                                flex: 1;
                                padding: 10px;
                                border: 1px solid #ddd;
                                border-radius: 6px;
                                font-size: 14px;
                                outline: none;
                                resize: none;
                                min-height: 40px;
                                max-height: 120px;
                                font-family: inherit;
                                line-height: 1.4;
                            "
                        ></textarea>
                        <button class="rag-chatbot-send" style="
                            background: ${config.theme.primaryColor};
                            color: white;
                            border: none;
                            padding: 10px 15px;
                            border-radius: 6px;
                            cursor: pointer;
                            font-size: 14px;
                            transition: all 0.2s;
                            height: 40px;
                            min-width: 60px;
                        ">Send</button>
                    </div>
                </div>

                <!-- Loading indicator -->
                <div class="rag-chatbot-loading" style="
                    display: none; 
                    padding: 15px; 
                    text-align: center; 
                    color: #666;
                    background: ${config.theme.backgroundColor};
                    border-top: 1px solid #eee;
                ">
                    <div style="display: inline-block; animation: rag-chatbot-spin 1s linear infinite; margin-right: 8px;">⟳</div>
                    Thinking...
                </div>
            </div>
        `;

        return container;
    }

    // Auto-resize textarea
    function autoResizeTextarea(textarea) {
        textarea.style.height = 'auto';
        const newHeight = Math.min(textarea.scrollHeight, 120);
        textarea.style.height = newHeight + 'px';
    }

    // API functions
    function getHeaders() {
        const headers = {
            'Content-Type': 'application/json'
        };
        if (chatbotState.config.apiKey) {
            headers['Authorization'] = `Bearer ${chatbotState.config.apiKey}`;
        }
        return headers;
    }

    async function createSession() {
        try {
            const response = await fetch(`${chatbotState.config.apiBaseUrl}/sessions`, {
                method: 'POST',
                headers: getHeaders()
            });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const data = await response.json();
            chatbotState.sessionId = data.session_id;
            return data.session_id;
        } catch (error) {
            console.error('RAG Chatbot: Failed to create session:', error);
            return null;
        }
    }

    async function sendMessage(message) {
        try {
            const response = await fetch(`${chatbotState.config.apiBaseUrl}/chat`, {
                method: 'POST',
                headers: getHeaders(),
                body: JSON.stringify({
                    message: message,
                    session_id: chatbotState.sessionId,
                    mode: 'original'
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            
            // Update session ID if new
            if (data.session_id) {
                chatbotState.sessionId = data.session_id;
            }

            return data;
        } catch (error) {
            console.error('RAG Chatbot: Chat error:', error);
            throw error;
        }
    }

    // UI functions
    function addMessage(text, sender) {
        const messagesContainer = document.querySelector('.rag-chatbot-messages');
        const isBot = sender === 'bot';
        const config = chatbotState.config;
        
        const messageDiv = document.createElement('div');
        messageDiv.className = `rag-chatbot-message-${sender}`;
        
        messageDiv.style.cssText = `
            background: ${isBot ? config.theme.botMessageColor : config.theme.userMessageColor};
            color: ${isBot ? config.theme.textColor : 'white'};
            padding: 10px 12px;
            border-radius: ${isBot ? '18px 18px 18px 4px' : '18px 18px 4px 18px'};
            margin-bottom: 10px;
            margin-left: ${isBot ? '0' : 'auto'};
            margin-right: ${isBot ? 'auto' : '0'};
            max-width: 80%;
            font-size: 14px;
            line-height: 1.4;
            word-wrap: break-word;
            white-space: pre-wrap;
        `;
        
        messageDiv.textContent = text;
        messagesContainer.appendChild(messageDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
        
        chatbotState.messageHistory.push({ 
            sender, 
            text, 
            timestamp: new Date().toISOString() 
        });
    }

    function showLoading(show) {
        const loading = document.querySelector('.rag-chatbot-loading');
        const inputContainer = loading.previousElementSibling;
        
        if (show) {
            loading.style.display = 'block';
            inputContainer.style.display = 'none';
        } else {
            loading.style.display = 'none';
            inputContainer.style.display = 'block';
        }
    }

    function toggleWidget() {
        const widget = document.querySelector('.rag-chatbot-widget');
        const input = document.querySelector('.rag-chatbot-input');
        
        chatbotState.isOpen = !chatbotState.isOpen;
        widget.style.display = chatbotState.isOpen ? 'flex' : 'none';
        
        if (chatbotState.isOpen) {
            input.focus();
            if (!chatbotState.sessionId) {
                createSession();
            }
        }
    }

    async function handleSendMessage() {
        const input = document.querySelector('.rag-chatbot-input');
        const message = input.value.trim();
        
        if (!message) return;

        // Add user message
        addMessage(message, 'user');
        input.value = '';
        input.style.height = 'auto';
        
        // Show loading
        showLoading(true);

        try {
            const response = await sendMessage(message);
            addMessage(response.response, 'bot');
        } catch (error) {
            addMessage('Sorry, I encountered an error. Please try again later.', 'bot');
        } finally {
            showLoading(false);
            input.focus();
        }
    }

    // Event handlers
    function setupEventListeners() {
        const trigger = document.querySelector('.rag-chatbot-trigger');
        const closeBtn = document.querySelector('.rag-chatbot-close');
        const input = document.querySelector('.rag-chatbot-input');
        const sendBtn = document.querySelector('.rag-chatbot-send');

        trigger.addEventListener('click', toggleWidget);
        closeBtn.addEventListener('click', toggleWidget);
        sendBtn.addEventListener('click', handleSendMessage);
        
        input.addEventListener('input', function() {
            autoResizeTextarea(this);
        });

        input.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSendMessage();
            }
        });

        // Handle clicks outside widget to close
        document.addEventListener('click', function(e) {
            const widget = document.querySelector('.rag-chatbot-widget');
            const trigger = document.querySelector('.rag-chatbot-trigger');
            
            if (chatbotState.isOpen && 
                !widget.contains(e.target) && 
                !trigger.contains(e.target)) {
                // Optionally close on outside click
                // toggleWidget();
            }
        });
    }

    // Initialization
    function initChatbot(userConfig) {
        // Prevent multiple initializations
        if (document.getElementById('rag-chatbot-container')) {
            console.warn('RAG Chatbot: Already initialized');
            return;
        }

        // Merge configuration
        chatbotState.config = mergeConfig(userConfig);

        // Create and inject styles
        createStyles();

        // Create and inject HTML
        const chatbotContainer = createChatbotHTML();
        document.body.appendChild(chatbotContainer);

        // Setup event listeners
        setupEventListeners();

        console.log('RAG Chatbot: Initialized successfully');
    }

    // Public API
    window.RAGChatbot = {
        init: initChatbot,
        
        // Additional methods for external control
        open: function() {
            if (!chatbotState.isOpen) {
                toggleWidget();
            }
        },
        
        close: function() {
            if (chatbotState.isOpen) {
                toggleWidget();
            }
        },
        
        sendMessage: function(message) {
            if (!chatbotState.isOpen) {
                this.open();
            }
            const input = document.querySelector('.rag-chatbot-input');
            if (input) {
                input.value = message;
                handleSendMessage();
            }
        },
        
        clearHistory: function() {
            const messagesContainer = document.querySelector('.rag-chatbot-messages');
            if (messagesContainer) {
                // Clear all messages except welcome message
                const welcomeMessage = messagesContainer.querySelector('.rag-chatbot-message-bot');
                messagesContainer.innerHTML = '';
                if (welcomeMessage) {
                    messagesContainer.appendChild(welcomeMessage);
                }
                chatbotState.messageHistory = [];
            }
        },
        
        getSessionId: function() {
            return chatbotState.sessionId;
        },
        
        getHistory: function() {
            return chatbotState.messageHistory;
        },
        
        updateConfig: function(newConfig) {
            chatbotState.config = mergeConfig(newConfig);
            // Note: This requires re-initialization to take full effect
            console.log('RAG Chatbot: Config updated. Re-initialize for visual changes.');
        },
        
        destroy: function() {
            const container = document.getElementById('rag-chatbot-container');
            const styles = document.getElementById('rag-chatbot-styles');
            
            if (container) {
                container.remove();
            }
            if (styles) {
                styles.remove();
            }
            
            // Reset state
            chatbotState = {
                sessionId: null,
                isOpen: false,
                messageHistory: [],
                config: DEFAULT_CONFIG
            };
            
            console.log('RAG Chatbot: Destroyed successfully');
        }
    };

})(window, document);