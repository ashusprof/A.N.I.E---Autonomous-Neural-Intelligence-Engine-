// Voice Mode Methods for FullscreenUIManager

// Add these methods to the FullscreenUIManager class

toggleVoiceMode() {
    this.voiceModeActive = !this.voiceModeActive;

    if (this.voiceModeActive) {
        // Activate voice mode
        this.elements.voiceModeToggle.classList.add('active');
        this.elements.voiceModeInterface.style.display = 'flex';
        this.elements.inputArea.style.display = 'none';
        this.elements.chatArea.style.display = 'none';
        this.updateVoiceStatus('Click the microphone to start');
    } else {
        // Deactivate voice mode
        this.elements.voiceModeToggle.classList.remove('active');
        this.elements.voiceModeInterface.style.display = 'none';
        this.elements.inputArea.style.display = 'block';
        this.elements.chatArea.style.display = 'block';
        this.stopListening();
        this.stopSpeaking();
    }
}

handleVoiceRecord() {
    if (!this.recognition) {
        alert('Speech recognition is not supported in your browser. Please use Chrome, Edge, or Safari.');
        return;
    }

    if (this.isListening) {
        this.stopListening();
    } else {
        this.startListening();
    }
}

startListening() {
    if (!this.recognition) return;

    this.isListening = true;
    this.elements.voiceRecordBtn.classList.add('recording');
    this.elements.voiceWaveform.classList.add('listening');
    this.updateVoiceStatus('Listening...');

    this.recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        this.updateVoiceStatus(`You said: "${transcript}"`);
        this.sendVoiceMessage(transcript);
    };

    this.recognition.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
        this.updateVoiceStatus('Error: ' + event.error);
        this.stopListening();
    };

    this.recognition.onend = () => {
        this.stopListening();
    };

    try {
        this.recognition.start();
    } catch (e) {
        console.error('Failed to start recognition:', e);
        this.stopListening();
    }
}

stopListening() {
    if (!this.isListening) return;

    this.isListening = false;
    this.elements.voiceRecordBtn.classList.remove('recording');
    this.elements.voiceWaveform.classList.remove('listening');

    if (this.recognition) {
        try {
            this.recognition.stop();
        } catch (e) {
            // Already stopped
        }
    }
}

async sendVoiceMessage(message) {
    if (!message) return;

    this.updateVoiceStatus('Processing...');

    try {
        const response = await fetch(`${CONFIG.BACKEND_URL}/api/voice/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                session_id: this.sessionId,
                stream: true
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        // Process streaming response
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullResponse = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value, { stream: true });
            const lines = chunk.split('\\n');

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = line.slice(6).trim();
                    if (data === '[DONE]') continue;

                    try {
                        const parsed = JSON.parse(data);
                        if (parsed.content) {
                            fullResponse += parsed.content;
                        }
                    } catch (e) {
                        // Ignore parse errors
                    }
                }
            }
        }

        if (fullResponse) {
            this.speakResponse(fullResponse);
        } else {
            this.updateVoiceStatus('No response received');
        }

    } catch (error) {
        console.error('Error sending voice message:', error);
        this.updateVoiceStatus('Error: Could not connect to backend');
        this.speak('Sorry, I encountered an error. Please try again.');
    }
}

speakResponse(text) {
    if (!this.synthesis) {
        this.updateVoiceStatus(text);
        return;
    }

    this.stopSpeaking();

    this.currentUtterance = new SpeechSynthesisUtterance(text);
    this.currentUtterance.lang = 'en-US';
    this.currentUtterance.rate = 1.0;
    this.currentUtterance.pitch = 1.0;

    this.currentUtterance.onstart = () => {
        this.isSpeaking = true;
        this.elements.voiceWaveform.classList.add('speaking');
        this.elements.voiceStopBtn.style.display = 'flex';
        this.updateVoiceStatus('Speaking...');
    };

    this.currentUtterance.onend = () => {
        this.isSpeaking = false;
        this.elements.voiceWaveform.classList.remove('speaking');
        this.elements.voiceStopBtn.style.display = 'none';
        this.updateVoiceStatus('Click the microphone to speak again');
    };

    this.currentUtterance.onerror = (event) => {
        console.error('Speech synthesis error:', event);
        this.stopSpeaking();
        this.updateVoiceStatus('Error speaking response');
    };

    this.synthesis.speak(this.currentUtterance);
}

stopSpeaking() {
    if (this.synthesis && this.isSpeaking) {
        this.synthesis.cancel();
        this.isSpeaking = false;
        this.elements.voiceWaveform.classList.remove('speaking');
        this.elements.voiceStopBtn.style.display = 'none';
        this.updateVoiceStatus('Stopped');
    }
}

updateVoiceStatus(status) {
    if (this.elements.voiceStatus) {
        this.elements.voiceStatus.textContent = status;
    }
}
