from flask import Flask, render_template, request, jsonify
import threading
import json
import os

# Import functions from your inference script
# Note: We're creating a modified version to avoid circular imports
from inference_wrapper import (
    get_user_input_from_mic,
    process_text_input,
    reset_conversation_context
)

app = Flask(__name__)

# Ensure the templates directory exists
os.makedirs('templates', exist_ok=True)

# Create the HTML template file
with open('templates/index.html', 'w') as f:
    with open('static/index.html', 'r') as src:
        f.write(src.read())


@app.route('/')
def index():
    """Serve the main UI page."""
    return render_template('index.html')


@app.route('/process_text', methods=['POST'])
def process_text():
    """Handle text input from the UI."""
    data = request.json
    user_text = data.get('text', '')

    if not user_text:
        return jsonify({'error': 'No text provided'}), 400

    # Process the text input using your inference script
    response = process_text_input(user_text)

    return jsonify({
        'response': response
    })


@app.route('/start_listening', methods=['GET'])
def start_listening():
    """Handle voice input request."""
    # Use your existing speech recognition and processing code
    transcription = get_user_input_from_mic()

    if not transcription:
        return jsonify({'error': 'Failed to transcribe speech'}), 400

    # Process the transcribed text
    response = process_text_input(transcription)

    return jsonify({
        'transcription': transcription,
        'response': response
    })


@app.route('/reset_context', methods=['GET'])
def reset_context():
    """Reset the conversation context."""
    reset_conversation_context()
    return jsonify({'success': True})


if __name__ == '__main__':
    # Ensure static directory exists
    os.makedirs('static', exist_ok=True)

    # Write the HTML to a static file that will be copied to templates
    with open('static/index.html', 'w') as f:
        # Read the HTML content from a file or paste it here
        html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Atlanta Bot</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            display: flex;
            max-width: 1200px;
            margin: 0 auto;
            gap: 20px;
        }
        .chat-panel {
            flex: 2;
            display: flex;
            flex-direction: column;
            height: 80vh;
        }
        .control-panel {
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .chat-box {
            flex-grow: 1;
            background-color: white;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 10px;
            overflow-y: auto;
        }
        .speak-button {
            background-color: #4CAF50;
            color: white;
            padding: 15px 20px;
            border: none;
            border-radius: 5px;
            font-size: 18px;
            cursor: pointer;
            width: 100%;
            margin-top: 10px;
        }
        .input-area {
            display: flex;
            margin-top: 10px;
        }
        .input-area input {
            flex-grow: 1;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            margin-right: 10px;
        }
        .input-area button {
            background-color: #2196F3;
            color: white;
            padding: 10px 15px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        .microphone {
            width: 120px;
            height: 120px;
            background-color: #f0f0f0;
            border-radius: 50%;
            display: flex;
            justify-content: center;
            align-items: center;
            position: relative;
            margin-bottom: 30px;
        }
        .mic-icon {
            width: 50px;
            height: 60px;
            background-color: #333;
            border-radius: 15px;
            position: relative;
        }
        .mic-icon:after {
            content: '';
            position: absolute;
            width: 20px;
            height: 10px;
            background-color: #333;
            bottom: -5px;
            left: 15px;
            border-radius: 10px;
        }
        .sound-wave {
            position: absolute;
            border: 3px solid #333;
            border-radius: 50%;
            opacity: 0;
        }
        .wave1 {
            width: 70px;
            height: 70px;
        }
        .wave2 {
            width: 90px;
            height: 90px;
        }
        .wave3 {
            width: 110px;
            height: 110px;
        }
        .language-toggle {
            margin: the 30px 0;
        }
        .toggle-container {
            display: flex;
            background-color: #e0e0e0;
            border-radius: 20px;
            width: 160px;
            position: relative;
            margin-top: 10px;
        }
        .toggle-option {
            flex: 1;
            text-align: center;
            padding: 8px 0;
            z-index: 1;
            cursor: pointer;
        }
        .toggle-slider {
            position: absolute;
            width: 80px;
            height: 100%;
            background-color: #2196F3;
            border-radius: 20px;
            transition: 0.3s;
            left: 0;
        }
        .reset-context {
            background-color: #f44336;
            color: white;
            padding: 10px 15px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            margin-top: 20px;
            width: 160px;
        }
        .message {
            margin-bottom: 10px;
            padding: 10px;
            border-radius: 5px;
        }
        .user-message {
            background-color: #e3f2fd;
            text-align: right;
        }
        .assistant-message {
            background-color: #f1f1f1;
        }
        @keyframes ripple {
            0% {
                opacity: 1;
                transform: scale(0.8);
            }
            100% {
                opacity: 0;
                transform: scale(1);
            }
        }
        .animate-waves .wave1 {
            animation: ripple 1s infinite ease-out;
        }
        .animate-waves .wave2 {
            animation: ripple 1s infinite ease-out 0.3s;
        }
        .animate-waves .wave3 {
            animation: ripple 1s infinite ease-out 0.6s;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="chat-panel">
            <div class="chat-box" id="chatBox"></div>
            <div class="input-area">
                <input type="text" id="userInput" placeholder="Type your message here...">
                <button id="sendBtn">Send</button>
            </div>
            <button class="speak-button" id="speakBtn">Speak</button>
        </div>
        <div class="control-panel">
            <div class="microphone" id="microphone">
                <div class="mic-icon"></div>
                <div class="sound-wave wave1"></div>
                <div class="sound-wave wave2"></div>
                <div class="sound-wave wave3"></div>
            </div>
            <div class="language-toggle">
                <div>Eng | SPA</div>
                <div class="toggle-container">
                    <div class="toggle-option" id="engOption">ENG</div>
                    <div class="toggle-option" id="spaOption">SPA</div>
                    <div class="toggle-slider" id="toggleSlider"></div>
                </div>
            </div>
            <button class="reset-context" id="resetBtn">Reset Context</button>
        </div>
    </div>

    <script>
        const chatBox = document.getElementById('chatBox');
        const userInput = document.getElementById('userInput');
        const sendBtn = document.getElementById('sendBtn');
        const speakBtn = document.getElementById('speakBtn');
        const microphone = document.getElementById('microphone');
        const resetBtn = document.getElementById('resetBtn');
        const engOption = document.getElementById('engOption');
        const spaOption = document.getElementById('spaOption');
        const toggleSlider = document.getElementById('toggleSlider');

        let isListening = false;
        let isSpanish = false;

        // Add message to chat
        function addMessage(text, isUser) {
            const messageDiv = document.createElement('div');
            messageDiv.classList.add('message');
            messageDiv.classList.add(isUser ? 'user-message' : 'assistant-message');
            messageDiv.textContent = text;
            chatBox.appendChild(messageDiv);
            chatBox.scrollTop = chatBox.scrollHeight;
        }

        // Toggle animation for microphone waves
        function toggleMicAnimation(show) {
            if (show) {
                microphone.classList.add('animate-waves');
            } else {
                microphone.classList.remove('animate-waves');
            }
        }

        // Handle text input
        sendBtn.addEventListener('click', () => {
            if (userInput.value.trim() === '') return;

            const userText = userInput.value;
            addMessage(userText, true);
            userInput.value = '';

            // Send to backend
            fetch('/process_text', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: userText }),
            })
            .then(response => response.json())
            .then(data => {
                // Show the assistant's response
                addMessage(data.response, false);
                // Animate mic during playback
                toggleMicAnimation(true);

                // Stop animation after estimated playback time (rough estimation)
                const words = data.response.split(' ').length;
                const estimatedTimeMs = words * 250; // ~250ms per word
                setTimeout(() => {
                    toggleMicAnimation(false);
                }, estimatedTimeMs);
            })
            .catch(error => {
                console.error('Error:', error);
                addMessage('Error processing your request', false);
            });
        });

        // Handle Enter key in input
        userInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                sendBtn.click();
            }
        });

        // Handle voice input
        speakBtn.addEventListener('click', () => {
            if (isListening) return;

            isListening = true;
            toggleMicAnimation(true);
            speakBtn.textContent = 'Listening...';

            // Request voice input from backend
            fetch('/start_listening')
                .then(response => response.json())
                .then(data => {
                    isListening = false;
                    speakBtn.textContent = 'Speak';
                    toggleMicAnimation(false);

                    // Add user's transcribed message
                    if (data.transcription) {
                        addMessage(data.transcription, true);
                    }

                    // Add assistant's response
                    if (data.response) {
                        addMessage(data.response, false);
                        // Animate mic during playback
                        toggleMicAnimation(true);

                        // Stop animation after estimated playback time
                        const words = data.response.split(' ').length;
                        const estimatedTimeMs = words * 250; // ~250ms per word
                        setTimeout(() => {
                            toggleMicAnimation(false);
                        }, estimatedTimeMs);
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    isListening = false;
                    speakBtn.textContent = 'Speak';
                    toggleMicAnimation(false);
                    addMessage('Error processing your speech', false);
                });
        });

        // Language toggle
        engOption.addEventListener('click', () => {
            if (isSpanish) {
                toggleSlider.style.left = '0';
                isSpanish = false;
                // Language switching is not implemented yet
            }
        });

        spaOption.addEventListener('click', () => {
            if (!isSpanish) {
                toggleSlider.style.left = '80px';
                isSpanish = true;
                // Language switching is not implemented yet
            }
        });

        // Reset context
        resetBtn.addEventListener('click', () => {
            fetch('/reset_context')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        chatBox.innerHTML = '';
                        addMessage('Context has been reset.', false);
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                });
        });
    </script>
</body>
</html>"""
        f.write(html_content)

    app.run(debug=True, port=5000)