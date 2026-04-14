(function () {
  'use strict';

  const CONFIG = {
    BACKEND_URL: 'http://127.0.0.1:8000',
    STORAGE_KEYS: {
      SESSION: 'anie-session-id',
      CHAT_HISTORY: 'anie-chat-history'
    }
  };

  function generateId() {
    return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }

  const storage = {
    get(key, defaultValue = null) {
      try {
        const item = localStorage.getItem(key);
        return item ? JSON.parse(item) : defaultValue;
      } catch (e) {
        return defaultValue;
      }
    },
    set(key, value) {
      try {
        localStorage.setItem(key, JSON.stringify(value));
      } catch (e) { }
    }
  };

  class ChatHistoryAPI {
    constructor(baseURL) {
      this.baseURL = baseURL;
    }

    async listConversations(limit = 50, offset = 0) {
      try {
        const response = await fetch(`${this.baseURL}/api/knowledge-base/list?limit=${limit}&offset=${offset}`);
        if (!response.ok) throw new Error('Failed to fetch conversations');
        return await response.json();
      } catch (error) {
        console.error('Error listing conversations:', error);
        return { conversations: [], total: 0 };
      }
    }

    async getConversation(sessionId) {
      try {
        const response = await fetch(`${this.baseURL}/api/knowledge-base/${sessionId}`);
        if (!response.ok) throw new Error('Failed to fetch conversation');
        return await response.json();
      } catch (error) {
        console.error('Error getting conversation:', error);
        return null;
      }
    }

    async saveConversation(sessionId, title = null) {
      try {
        const response = await fetch(`${this.baseURL}/api/knowledge-base/save`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ session_id: sessionId, title: title })
        });
        if (!response.ok) throw new Error('Failed to save conversation');
        return await response.json();
      } catch (error) {
        console.error('Error saving conversation:', error);
        return { success: false, error: error.message };
      }
    }

    async deleteConversation(sessionId) {
      try {
        const response = await fetch(`${this.baseURL}/api/knowledge-base/${sessionId}`, {
          method: 'DELETE'
        });
        if (!response.ok) throw new Error('Failed to delete conversation');
        return await response.json();
      } catch (error) {
        console.error('Error deleting conversation:', error);
        return { success: false };
      }
    }
  }

  class ChatAPI {
    constructor(baseURL) {
      this.baseURL = baseURL;
      this.abortController = null;
    }

    async sendMessage(message, sessionId) {
      if (this.abortController) {
        this.abortController.abort();
      }

      this.abortController = new AbortController();

      try {
        const response = await fetch(`${this.baseURL}/api/chat`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            message: message,
            session_id: sessionId,
            stream: true
          }),
          signal: this.abortController.signal
        });

        if (!response.ok) {
          const errorText = await response.text();
          console.error('Backend error:', response.status, errorText);
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        // Check if response is streaming (SSE) or JSON
        const contentType = response.headers.get('content-type');
        if (contentType && contentType.includes('text/event-stream')) {
          return response.body;
        } else {
          // Handle JSON response
          const data = await response.json();
          // Convert JSON response to stream-like format for compatibility
          return this._convertJsonToStream(data);
        }
      } catch (error) {
        if (error.name === 'AbortError') {
          return null;
        }
        throw error;
      }
    }

    _convertJsonToStream(data) {
      // Convert JSON response to a readable stream format
      const encoder = new TextEncoder();
      let content = '';

      // Extract content from response
      if (data.reply) {
        content = data.reply;
      } else if (data.message && typeof data.message === 'object') {
        content = data.message.content || data.message.message || '';
      } else if (typeof data.message === 'string') {
        content = data.message;
      } else if (data.response) {
        content = data.response;
      } else if (data.content) {
        content = data.content;
      }

      // Create a stream-like object
      const chunks = content.split('').map(char => encoder.encode(`data: ${JSON.stringify({ content: char })}\n\n`));
      let index = 0;

      return {
        getReader: () => ({
          read: async () => {
            if (index >= chunks.length) {
              return { done: true, value: undefined };
            }
            const chunk = chunks[index++];
            return { done: false, value: chunk };
          },
          releaseLock: () => { }
        })
      };
    }

    cancel() {
      if (this.abortController) {
        this.abortController.abort();
        this.abortController = null;
      }
    }
  }

  class FullscreenUIManager {
    constructor(elements, chatAPI, historyAPI) {
      this.elements = elements;
      this.chatAPI = chatAPI;
      this.historyAPI = historyAPI;
      // Generate a fresh session ID on each page load to start clean
      // Old conversations can still be accessed via chat history sidebar
      this.sessionId = generateId();
      this.chatHistory = [];

      storage.set(CONFIG.STORAGE_KEYS.SESSION, this.sessionId);


      console.log('✓ Initialized');
    }

    init() {
      this.setupEventListeners();
      // Don't auto-restore chat history on page load - start fresh
      // Chat history will be loaded only when user clicks on a history item
      this.showWelcomeScreen();
    }

    setupEventListeners() {
      this.elements.messageInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
          e.preventDefault();
          this.handleSendMessage();
        }
      });

      this.elements.sendBtn.addEventListener('click', () => {
        this.handleSendMessage();
      });

      // New Chat button - simply refresh the page
      const newChatBtn = document.getElementById('new-chat-btn');
      if (newChatBtn) {
        newChatBtn.addEventListener('click', () => {
          window.location.reload();
        });
      }
    }

    async handleSendMessage() {
      const message = this.elements.messageInput.value.trim();
      if (!message) return;

      this.elements.messageInput.value = '';
      this.showMessagesContainer();
      this.addMessage(message, 'user');
      this.chatHistory.push({ role: 'user', content: message });
      this.saveChatHistory();

      const typingId = this.showTypingIndicator();

      try {
        const stream = await this.chatAPI.sendMessage(message, this.sessionId);

        if (!stream) {
          this.removeTypingIndicator(typingId);
          return;
        }

        // Keep typing indicator visible until first content arrives
        await this.processStream(stream, typingId);

      } catch (error) {
        console.error('Error sending message:', error);
        this.removeTypingIndicator(typingId);

        // Provide more helpful error messages
        let errorMessage = 'Sorry, I encountered an error.';

        if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
          errorMessage = 'Cannot connect to the backend server. Please make sure:\n\n1. The backend is running (cd Agent && python main.py)\n2. Ollama is running on http://localhost:11434\n3. Check the console for more details';
        } else if (error.message.includes('404')) {
          errorMessage = 'Backend endpoint not found. The server might be running but the API endpoint is incorrect.';
        } else if (error.message.includes('500')) {
          errorMessage = 'Backend server error. Check if Ollama is running and the model is available.';
        }

        this.addMessage(errorMessage, 'assistant', true);
      }
    }

    async processStream(stream, typingId = null) {
      const reader = stream.getReader();
      const decoder = new TextDecoder();
      let assistantMessage = '';
      let messageElement = null;
      let typingIndicatorRemoved = false;
      let buffer = ''; // Buffer for incomplete chunks

      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) {
            // Stream complete - any remaining buffer data is typically just whitespace
            break;
          }

          const chunk = decoder.decode(value, { stream: true });
          buffer += chunk;

          // Split by newlines but keep the last incomplete line in buffer
          const lines = buffer.split('\n');
          buffer = lines.pop() || ''; // Keep the last incomplete line

          for (const line of lines) {
            if (!line.trim()) continue;

            if (line.startsWith('data: ')) {
              const data = line.slice(6).trim();

              if (data === '[DONE]') {
                continue;
              }

              if (!data) continue;

              try {
                const parsed = JSON.parse(data);

                // Handle session_id message (ignore it)
                if (parsed.session_id) {
                  this.sessionId = parsed.session_id;
                  continue;
                }

                // Handle status messages - update typing indicator
                if (parsed.status) {
                  console.log('Status:', parsed.status);
                  if (typingId && !typingIndicatorRemoved) {
                    this.updateTypingIndicator(typingId, parsed.status);
                  }
                  continue;
                }

                // Handle content messages
                if (parsed.content !== undefined) {
                  // Filter out TOOL calls that shouldn't be shown
                  let content = parsed.content;
                  if (content.includes('[TOOL:')) {
                    content = content.replace(/\[TOOL:.*?\]/gs, '');
                    if (!content.trim()) continue; // Skip if only tool call
                  }

                  // Remove typing indicator when first content arrives
                  if (typingId && !typingIndicatorRemoved) {
                    this.removeTypingIndicator(typingId);
                    typingIndicatorRemoved = true;
                  }

                  assistantMessage += content;

                  if (!messageElement) {
                    messageElement = this.addMessage('', 'assistant');
                    const contentDiv = messageElement.querySelector('.message-content');
                    if (contentDiv) {
                      contentDiv.classList.add('markdown-content');
                    }
                  }

                  // Render markdown for streaming content
                  const contentDiv = messageElement.querySelector('.message-content');
                  if (contentDiv) {
                    if (typeof marked !== 'undefined') {
                      contentDiv.innerHTML = marked.parse(assistantMessage);
                    } else {
                      contentDiv.textContent = assistantMessage;
                    }
                  }
                  this.scrollToBottom();
                }
              } catch (e) {
                console.warn('Failed to parse SSE data:', e, 'Data:', data);
              }
            }
          }
        }
      } catch (error) {
        console.error('Stream processing error:', error);
      } finally {
        // Make sure typing indicator is removed even if stream ends early
        if (typingId && !typingIndicatorRemoved) {
          this.removeTypingIndicator(typingId);
        }

        // Save assistant message ONCE at the end
        if (assistantMessage && assistantMessage.trim().length > 0) {
          // Check if this exact message was already saved
          const lastMessage = this.chatHistory[this.chatHistory.length - 1];
          const wasSaved = lastMessage &&
            lastMessage.role === 'assistant' &&
            lastMessage.content === assistantMessage;

          if (!wasSaved) {
            this.chatHistory.push({ role: 'assistant', content: assistantMessage });
            this.saveChatHistory();
            console.log('Saved assistant message:', assistantMessage.substring(0, 50) + '...');
          }
        }

        reader.releaseLock();
      }
    }

    showMessagesContainer() {
      this.elements.welcomeScreen.style.display = 'none';
      this.elements.messagesContainer.style.display = 'block';
    }

    showWelcomeScreen() {
      this.elements.welcomeScreen.style.display = 'flex';
      this.elements.messagesContainer.style.display = 'none';
    }

    addMessage(content, role, isError = false) {
      const messageDiv = document.createElement('div');
      messageDiv.className = `message ${role}`;

      const contentDiv = document.createElement('div');
      contentDiv.className = 'message-content';

      // Handle error messages (plain text)
      if (isError) {
        contentDiv.textContent = content;
        contentDiv.style.color = '#ef4444';
      } else {
        // Render markdown for assistant messages, plain text for user messages
        if (role === 'assistant' && typeof marked !== 'undefined') {
          contentDiv.className += ' markdown-content';
          contentDiv.innerHTML = marked.parse(content);
        } else {
          contentDiv.textContent = content;
        }
      }

      messageDiv.appendChild(contentDiv);
      this.elements.messagesContainer.appendChild(messageDiv);
      this.scrollToBottom();

      return messageDiv;
    }

    showTypingIndicator(statusText = 'Thinking...') {
      const typingId = generateId();
      const typingDiv = document.createElement('div');
      typingDiv.className = 'message assistant';
      typingDiv.id = typingId;

      const typingContent = document.createElement('div');
      typingContent.className = 'message-content';

      const typingIndicator = document.createElement('div');
      typingIndicator.className = 'typing-indicator';
      typingIndicator.setAttribute('aria-label', statusText);

      const indicatorContent = document.createElement('div');
      indicatorContent.className = 'typing-indicator-content';

      const dots = document.createElement('div');
      dots.className = 'typing-dots';
      dots.innerHTML = '<span></span><span></span><span></span>';

      const statusTextEl = document.createElement('div');
      statusTextEl.className = 'typing-status-text';
      statusTextEl.textContent = statusText;

      indicatorContent.appendChild(statusTextEl);
      indicatorContent.appendChild(dots);
      typingIndicator.appendChild(indicatorContent);

      typingContent.appendChild(typingIndicator);
      typingDiv.appendChild(typingContent);
      this.elements.messagesContainer.appendChild(typingDiv);
      this.scrollToBottom();

      // Store reference to update status text
      typingDiv._statusTextEl = statusTextEl;
      typingDiv._indicatorContent = indicatorContent;

      return typingId;
    }

    updateTypingIndicator(typingId, statusText) {
      const typingElement = document.getElementById(typingId);
      if (typingElement && typingElement._statusTextEl) {
        typingElement._statusTextEl.textContent = statusText;
      }
    }

    removeTypingIndicator(typingId) {
      const typingElement = document.getElementById(typingId);
      if (typingElement) {
        typingElement.remove();
      }
    }

    scrollToBottom() {
      this.elements.chatArea.scrollTop = this.elements.chatArea.scrollHeight;
    }



    // Sidebar functionality removed

    // New chat handled by page refresh

    // Chat history functionality removed

    escapeHtml(text) {
      const div = document.createElement('div');
      div.textContent = text;
      return div.innerHTML;
    }

    formatTimestamp(timestamp) {
      const date = new Date(timestamp);
      const now = new Date();
      const diffMs = now - date;
      const diffMins = Math.floor(diffMs / 60000);
      const diffHours = Math.floor(diffMs / 3600000);
      const diffDays = Math.floor(diffMs / 86400000);

      if (diffMins < 1) return 'Just now';
      if (diffMins < 60) return `${diffMins}m ago`;
      if (diffHours < 24) return `${diffHours}h ago`;
      if (diffDays < 7) return `${diffDays}d ago`;

      return date.toLocaleDateString();
    }

    async saveChatHistory() {
      // Save to localStorage as backup
      storage.set(CONFIG.STORAGE_KEYS.CHAT_HISTORY, this.chatHistory);

      // Auto-save is now handled by backend after each message
      // This method kept for compatibility
    }

    async restoreChatHistory(sessionId = null) {
      // Only restore if explicitly requested (when loading a specific conversation)
      const targetSessionId = sessionId || this.sessionId;

      // Try to load from database first
      try {
        const conversation = await this.historyAPI.getConversation(targetSessionId);
        if (conversation && conversation.messages && conversation.messages.length > 0) {
          this.chatHistory = conversation.messages.map(msg => ({
            role: msg.role,
            content: msg.content
          }));
          this.showMessagesContainer();
          this.chatHistory.forEach(msg => {
            this.addMessage(msg.content, msg.role);
          });
          return true;
        }
      } catch (error) {
        console.error('Error loading from database:', error);
      }

      // Don't fallback to localStorage on page load - start fresh
      // Only use localStorage as backup when explicitly loading a conversation
      if (sessionId) {
        const localHistory = storage.get(CONFIG.STORAGE_KEYS.CHAT_HISTORY, []);
        if (localHistory.length > 0) {
          this.chatHistory = localHistory;
          this.showMessagesContainer();
          this.chatHistory.forEach(msg => {
            this.addMessage(msg.content, msg.role);
          });
          return true;
        }
      }

      return false;
    }
  }

  function init() {
    const elements = {
      messageInput: document.getElementById('message-input'),
      sendBtn: document.getElementById('send-btn'),
      chatArea: document.getElementById('chat-area'),
      welcomeScreen: document.getElementById('welcome-screen'),
      messagesContainer: document.getElementById('messages-container'),
      inputArea: document.getElementById('input-area')
    };

    const chatAPI = new ChatAPI(CONFIG.BACKEND_URL);
    const historyAPI = new ChatHistoryAPI(CONFIG.BACKEND_URL);
    const uiManager = new FullscreenUIManager(elements, chatAPI, historyAPI);

    uiManager.init();

    window.uiManager = uiManager;
    console.log('ANIE Fullscreen initialized successfully');
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();


