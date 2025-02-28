# This MUST be at the top before ANY other imports
import eventlet

eventlet.monkey_patch()

from melo.api import TTS
import whisper
from ollama import Client
import pygame
import re
import os
import time
import threading
from queue import Queue
import pyaudio
import wave
import numpy as np
import json
import requests
from scipy.spatial.distance import cosine

from flask import Flask, request, render_template, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
import base64


# Add this at the top of your script to test connection directly
import requests
try:
    response = requests.get("http://localhost:11434/api/version")
    print(f"Ollama API check: {response.status_code}, {response.text}")
except Exception as e:
    print(f"Ollama API not accessible: {e}")


# Set mic_mode to True to use microphone input, False to use text input
mic_mode = False

# Toggle between English and Spanish modes
spanish_mode = True

# Language database for prompts
language_db = {
    "english": {
        "system_prompt": "You are Atlanta-bot, a helpful assistant that knows all about the city of Atlanta, Georgia. You only answer questions relevant to the city of Atlanta, so if the query is irrelevant, ensure you stay on-topic. You will be given information related to the users query if it is within your knowledge-base. If you are given information, ensure you only reference relevant parts related to the users query. Provide brief and helpful responses.",
        "loading_whisper": "Loading Whisper model...",
        "loading_tts": "Loading TTS model...",
        "init_ollama": "Initializing Ollama client...",
        "loading_embeddings": "Loading embeddings database...",
        "loaded_embeddings": "Loaded embeddings for {} categories",
        "all_models_loaded": "All models loaded and ready. Starting conversation loop...",
        "recording_start": "Recording... (speak now)",
        "recording_with_silence": "Recording... (speak now, will stop after 3 seconds of silence)",
        "recording_finished": "Recording finished.",
        "transcribing": "Transcribing...",
        "user_transcribed": "User (transcribed): {}",
        "user_prompt": "\nUser: ",
        "searching_kb": "Searching knowledge base for relevant information...",
        "no_embeddings": "No similar embeddings found above threshold (0.5)",
        "no_info_context": "I don't have specific information about that in my knowledge base. I'll try to help with what I know generally if the query is on-topic.",
        "found_info": "Found relevant information:",
        "press_record": "\nPress Enter to start recording...",
        "assistant_prefix": "Assistant: "
    },
    "spanish": {
        "system_prompt": "Eres Spanish Atlanta-bot, un asistente español útil que conoce todo sobre la ciudad de Atlanta, Georgia. Solo respondes preguntas relevantes a la ciudad de Atlanta, así que si la consulta no es relevante, asegúrate de mantenerte en el tema. Se te proporcionará información relacionada con la consulta del usuario si está dentro de tu base de conocimiento. Si se te proporciona información, asegúrate de hacer referencia solo a partes relevantes relacionadas con la consulta del usuario. Proporciona respuestas breves y útiles. SOLO respondes en español, incluso si el usuario responde en inglés.",
        "loading_whisper": "Cargando modelo Whisper...",
        "loading_tts": "Cargando modelo TTS...",
        "init_ollama": "Inicializando cliente Ollama...",
        "loading_embeddings": "Cargando base de datos de embeddings...",
        "loaded_embeddings": "Embeddings cargados para {} categorías",
        "all_models_loaded": "Todos los modelos cargados y listos. Iniciando bucle de conversación...",
        "recording_start": "Grabando... (habla ahora)",
        "recording_with_silence": "Grabando... (habla ahora, se detendrá después de 3 segundos de silencio)",
        "recording_finished": "Grabación finalizada.",
        "transcribing": "Transcribiendo...",
        "user_transcribed": "Usuario (transcrito): {}",
        "user_prompt": "\nUsuario: ",
        "searching_kb": "Buscando información relevante en la base de conocimiento...",
        "no_embeddings": "No se encontraron embeddings similares por encima del umbral (0.5)",
        "no_info_context": "No tengo información específica sobre eso en mi base de conocimiento. Intentaré ayudar con lo que sé en general si la consulta está dentro del tema.",
        "found_info": "Información relevante encontrada:",
        "press_record": "\nPresiona Enter para comenzar a grabar...",
        "assistant_prefix": "Asistente: "
    }
}


# Function to get the correct language text
def get_text(key, *args):
    lang = "spanish" if spanish_mode else "english"
    text = language_db[lang][key]
    if args:
        return text.format(*args)
    return text


# Path to embeddings JSON file
EMBEDDINGS_FILE = "embeddings.json"

# Load models once at the beginning and keep them in memory
print(get_text("loading_whisper"))
whisper_model = whisper.load_model("tiny")

# After loading the TTS model, add this:
print("Loading TTS model...")
device = 'cuda'  # cpu or cuda
tts_language = "ES" if spanish_mode else "EN-BR"
melo_model = TTS(language=tts_language, device=device)
speaker_ids = melo_model.hps.data.spk2id

# Debug available speakers
print(f"Available TTS speakers: {list(speaker_ids.keys())}")
print(f"Speaker IDs dictionary: {speaker_ids}")

print(get_text("init_ollama"))
client = Client(host='http://127.0.0.1:11434')
messages = [{'role': 'system', 'content': get_text("system_prompt")}]

# Initialize pygame mixer once
pygame.mixer.init()

# Pre-load embeddings to keep them in memory
print(get_text("loading_embeddings"))
with open(EMBEDDINGS_FILE, 'r') as f:
    embeddings_database = json.load(f)
print(get_text("loaded_embeddings", len(embeddings_database)))


# RAG Functions
# Replace your current get_embedding function with this:
def get_embedding(text, model="bge-m3:latest"):
    """Get embeddings from Ollama for the user query"""
    try:
        # Use the Ollama client instance you already have
        response = client.embeddings(model=model, prompt=text)
        return response["embedding"]
    except Exception as e:
        print(f"Embedding error: {e}")
        # Fallback to direct API call if client method fails
        try:
            response = requests.post(
                "http://127.0.0.1:11434/api/embeddings",
                json={
                    "model": model,
                    "prompt": text
                },
                timeout=10
            )
            if response.status_code == 200:
                return response.json()["embedding"]
            else:
                print(f"API error: {response.status_code}, {response.text}")
                return None
        except Exception as e2:
            print(f"API fallback error: {e2}")
            return None


def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    return 1 - cosine(vec1, vec2)


def find_closest_embeddings(user_text, top_n=2, similarity_threshold=0.5):
    """Find the top N closest embeddings to the user text"""
    # Get embedding for user text
    user_embedding = get_embedding(user_text)
    if not user_embedding:
        return []

    # Calculate similarities across all categories
    all_similarities = []

    for category, items in embeddings_database.items():
        for item in items:
            similarity = cosine_similarity(user_embedding, item["embedding"])
            if similarity >= similarity_threshold:
                all_similarities.append({
                    "category": category,
                    "text": item["text"],
                    "similarity": similarity,
                    "row_index": item["row_index"]
                })

    # Sort by similarity (highest first)
    all_similarities.sort(key=lambda x: x["similarity"], reverse=True)

    # Return top N results
    return all_similarities[:top_n]


def play_wav(file_path):
    sound = pygame.mixer.Sound(file_path)
    sound.play()
    pygame.time.wait(int(sound.get_length() * 1000))
    # Delete the wav file after playing it
    try:
        sound.stop()  # Make sure it's not being accessed
        pygame.time.wait(100)  # Small delay to ensure file is released
        os.remove(file_path)
    except Exception as e:
        print(f"Error removing file {file_path}: {e}")


def text2speech(text, speed, speaker_id, output_path):
    # Make sure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Use the original pattern - look up the speaker_id in the dictionary
    melo_model.tts_to_file(text, speaker_ids[speaker_id], output_path, speed=speed)
    return output_path


def split_into_sentences(text):
    # Simple sentence splitting - can be improved
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s for s in sentences if s.strip()]


def tts_worker(sentence_queue, speed, speaker_id, i):
    wav_files = []
    while True:
        idx, sentence = sentence_queue.get()
        if sentence is None:  # Sentinel value to stop the thread
            sentence_queue.task_done()
            break

        output_path = f"tts_output\\Mello_TTS_Output_{i}_{idx}.wav"
        text2speech(sentence, speed, speaker_id, output_path)
        play_wav(output_path)
        sentence_queue.task_done()


def record_audio(seconds=5, filename="input.wav"):
    # Parameters for recording
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 1024

    audio = pyaudio.PyAudio()

    # Start recording
    print(get_text("recording_start"))
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    frames = []

    # Record for the specified duration
    for _ in range(0, int(RATE / CHUNK * seconds)):
        data = stream.read(CHUNK)
        frames.append(data)

    # Stop recording
    print(get_text("recording_finished"))
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save the recorded audio to a WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    return filename


def get_user_input():
    if mic_mode:
        print(get_text("press_record"))
        input()

        # Parameters for recording
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        CHUNK = 1024
        THRESHOLD = 1000  # Adjust this threshold based on your microphone sensitivity
        SILENCE_LIMIT = 3  # 3 seconds of silence to stop recording

        audio = pyaudio.PyAudio()

        # Start recording
        print(get_text("recording_with_silence"))
        stream = audio.open(format=FORMAT, channels=CHANNELS,
                            rate=RATE, input=True,
                            frames_per_buffer=CHUNK)

        frames = []
        silent_chunks = 0
        silent_threshold = int(SILENCE_LIMIT * RATE / CHUNK)  # Number of chunks for silence threshold

        # Record until silence is detected for SILENCE_LIMIT seconds
        while True:
            data = stream.read(CHUNK)
            frames.append(data)

            # Convert audio data to numpy array and calculate volume
            audio_data = np.frombuffer(data, dtype=np.int16)
            volume = np.abs(audio_data).mean()

            # Check if volume is below threshold
            if volume < THRESHOLD:
                silent_chunks += 1
                if silent_chunks >= silent_threshold:
                    break
            else:
                silent_chunks = 0

        # Stop recording
        print(get_text("recording_finished"))
        stream.stop_stream()
        stream.close()
        audio.terminate()

        # Save the recorded audio to a WAV file
        filename = "input.wav"
        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

        # Transcribe the audio using Whisper
        print(get_text("transcribing"))
        result = whisper_model.transcribe(filename)
        transcribed_text = result["text"].strip()

        print(get_text("user_transcribed", transcribed_text))
        return transcribed_text
    else:
        return input(get_text("user_prompt")).strip()


def get_first_n_words(text, n=5):
    """Get the first n words from a text"""
    words = text.split()
    return " ".join(words[:n])


print(get_text("all_models_loaded"))


# Function to ensure models are not unloaded
def keep_model_loaded():
    # This is a small dummy operation to keep the model in VRAM
    # Just access the model without trying to use device attribute
    if hasattr(melo_model, 'model'):
        # Simply reference the model to keep it in memory
        _ = melo_model.model
    # For the whisper model
    if whisper_model is not None:
        _ = whisper_model


# --------------------------------------------------------------
# Add debug logging for Socket.IO
# Initialize Flask
app = Flask(__name__, static_folder='static')
app.config['SECRET_KEY'] = 'atlanta-bot-secret'
socketio = SocketIO(app, cors_allowed_origins="*")  # Add Socket.IO instance

# Create necessary directories
os.makedirs("static/tts_output", exist_ok=True)
os.makedirs("templates", exist_ok=True)


# Flask routes
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/static/tts_output/<path:filename>')
def serve_audio(filename):
    return send_from_directory('static/tts_output', filename)


@app.route('/toggle_language', methods=['POST'])
def toggle_language():
    global spanish_mode, melo_model, speaker_ids
    data = request.get_json()
    new_spanish_mode = data.get('spanish', False)

    # Only reload if language actually changed
    if new_spanish_mode != spanish_mode:
        spanish_mode = new_spanish_mode

        # Update system prompt with new language
        system_prompt = get_text("system_prompt")
        messages[0] = {'role': 'system', 'content': system_prompt}

        # Reload TTS model with new language setting
        tts_language = "ES" if spanish_mode else "EN"
        try:
            print(f"Reloading TTS model with language: {tts_language}")
            melo_model = TTS(language=tts_language, device='cuda')
            speaker_ids = melo_model.hps.data.spk2id
            print(f"Available TTS speakers after reload: {list(speaker_ids.keys())}")
        except Exception as e:
            print(f"Error reloading TTS model: {e}")
    else:
        # Language unchanged, no need to reload
        spanish_mode = new_spanish_mode

    return jsonify({'success': True, 'spanish': spanish_mode})


@app.route('/clear_context', methods=['POST'])
def clear_context():
    global messages
    # Reset to just the system message
    system_prompt = get_text("system_prompt")
    messages = [{'role': 'system', 'content': system_prompt}]
    return jsonify({'success': True})


# New function to prepare RAG context with socketio status updates
def prepare_rag_context_with_updates(user_input):
    """Prepare RAG context from user input, with status updates via socketio"""
    search_status = get_text("searching_kb")
    print(search_status)

    # Emit searching status to the client
    socketio.emit('search_status', {'status': search_status})

    # Get embedding for user query
    relevant_results = find_closest_embeddings(user_input, top_n=3, similarity_threshold=0.5)

    if not relevant_results:
        no_results_msg = get_text("no_embeddings")
        print(no_results_msg)

        # Emit no results found status
        socketio.emit('search_status', {'status': no_results_msg})

        context = get_text("no_info_context")
        augmented_prompt = f"User query: {user_input}\n\nNote: {context}"
    else:
        results_found_msg = get_text("found_info")
        print(results_found_msg)

        # Prepare result summary for the status update
        result_summary = []
        for i, result in enumerate(relevant_results, 1):
            preview = get_first_n_words(result['text'])
            result_info = f"{i}. [{result['category']}] \"{preview}...\" (Similarity: {result['similarity']:.4f})"
            print(f"  {result_info}")
            result_summary.append(result_info)

        # Emit found results status with summary
        socketio.emit('search_status', {
            'status': results_found_msg,
            'results': result_summary
        })

        # Prepare the context for the LLM
        context = "Based on my knowledge base, I found the following relevant information:\n\n"
        for i, result in enumerate(relevant_results, 1):
            context += f"[Source {i} from category '{result['category']}']:\n{result['text']}\n\n"

        augmented_prompt = f"User query: {user_input}\n\n{context}\nPlease use the above information to provide an accurate and helpful response to the user's query."

    return user_input, augmented_prompt


@app.route('/send_message', methods=['POST'])
def send_message():
    global messages
    data = request.get_json()
    user_input = data.get('message', '')

    # Add the original user input to the conversation history
    messages.append({'role': 'user', 'content': user_input})

    # Process user input with status updates
    original_input, augmented_prompt = prepare_rag_context_with_updates(user_input)

    # Create a temporary messages list with the augmented prompt
    rag_messages = messages.copy()
    rag_messages[-1] = {'role': 'user', 'content': augmented_prompt}

    # Ensure models stay loaded
    keep_model_loaded()

    # Get LLM response
    socketio.emit('search_status', {'status': 'Generating response...'})
    response = client.chat(model='llama3.2:3b', messages=rag_messages)
    assistant_response = response['message']['content']

    # Process TTS
    sentences = split_into_sentences(assistant_response)
    audio_files = []

    # Get available speakers
    available_keys = list(speaker_ids.keys())
    print(f"Available speaker keys: {available_keys}")

    # Choose appropriate speaker - use EN-US if available, otherwise use the first available English speaker
    if spanish_mode:
        speaker_key = "ES"
    elif "EN-BR" in available_keys:
        speaker_key = "EN-BR"
    elif len(available_keys) > 0:
        # Just use the first available speaker (likely English)
        speaker_key = available_keys[0]
    else:
        # Handle case with no speakers
        speaker_key = None

    print(f"Using speaker key: {speaker_key}")

    for idx, sentence in enumerate(sentences):
        if sentence.strip() and speaker_key:
            output_path = f"static/tts_output/tts_{int(time.time())}_{idx}.wav"
            try:
                # Use positional arguments as in your original code
                melo_model.tts_to_file(sentence, speaker_ids[speaker_key], output_path, speed=1.0)
                audio_files.append(output_path)
            except Exception as e:
                print(f"TTS error: {str(e)}")

    # Add the LLM's response to conversation history
    messages.append({'role': 'assistant', 'content': assistant_response})

    return jsonify({
        'response': assistant_response,
        'audio_files': audio_files
    })


@app.route('/process_audio', methods=['POST'])
def process_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file'})

    audio_file = request.files['audio']
    filename = f"input_{int(time.time())}.wav"
    audio_file.save(filename)

    try:
        # Transcribe audio
        result = whisper_model.transcribe(filename)
        transcribed_text = result["text"].strip()

        # Clean up temp file
        os.remove(filename)
        return jsonify({'text': transcribed_text})
    except Exception as e:
        return jsonify({'error': str(e)})


# Socket.IO event handlers
@socketio.on('connect')
def handle_connect():
    print(f"Client connected: {request.sid}")
    socketio.emit('language_setting', {'spanish': spanish_mode})


@socketio.on('disconnect')
def handle_disconnect():
    print(f"Client disconnected: {request.sid}")


@socketio.on('user_input')
def handle_user_input(data):
    user_message = data.get('message', '')
    # Emit user message to all clients
    socketio.emit('chat_update', {'role': 'user', 'content': user_message})

    # Process user input with status updates
    original_input, augmented_prompt = prepare_rag_context_with_updates(user_message)

    # Create a temporary messages list with the augmented prompt
    rag_messages = messages.copy()
    rag_messages.append({'role': 'user', 'content': augmented_prompt})

    # Ensure models stay loaded
    keep_model_loaded()

    socketio.emit('search_status', {'status': 'Generating response...'})

    # Get LLM response
    response = client.chat(model='llama3.2:3b', messages=rag_messages)
    assistant_response = response['message']['content']

    # Stream the response (simulate streaming for now)
    words = assistant_response.split()
    current_chunk = ""
    for word in words:
        current_chunk += word + " "
        if len(current_chunk.split()) >= 3 or word == words[-1]:  # Send chunks of ~3 words
            socketio.emit('assistant_stream', {'content': current_chunk})
            current_chunk = ""
            socketio.sleep(0.1)  # Small delay for "streaming" effect

    socketio.emit('assistant_done', {})

    # Process TTS and emit audio paths
    sentences = split_into_sentences(assistant_response)
    audio_files = []

    for idx, sentence in enumerate(sentences):
        if sentence.strip():
            speaker_key = "ES" if spanish_mode else "EN-BR" #available_keys[0]
            output_path = f"/static/tts_output/tts_{int(time.time())}_{idx}.wav"
            file_path = f"static/tts_output/tts_{int(time.time())}_{idx}.wav"
            try:
                melo_model.tts_to_file(sentence, speaker_ids[speaker_key], file_path, speed=1.0)
                audio_files.append(output_path)
            except Exception as e:
                print(f"TTS error: {str(e)}")
    emit('audio_sequence', {'files': audio_files})

    # Add to conversation history
    messages.append({'role': 'user', 'content': user_message})
    messages.append({'role': 'assistant', 'content': assistant_response})


@socketio.on('toggle_language')
def handle_toggle_language(data):
    global spanish_mode, melo_model, speaker_ids
    new_spanish_mode = data.get('spanish', False)

    # Only reload if language actually changed
    if new_spanish_mode != spanish_mode:
        spanish_mode = new_spanish_mode

        # Update system prompt with new language
        system_prompt = get_text("system_prompt")
        messages[0] = {'role': 'system', 'content': system_prompt}

        # Reload TTS model with new language setting
        tts_language = "ES" if spanish_mode else "EN"
        try:
            socketio.emit('search_status', {'status': f"Reloading TTS model with language: {tts_language}"})
            melo_model = TTS(language=tts_language, device='cuda')
            speaker_ids = melo_model.hps.data.spk2id
        except Exception as e:
            socketio.emit('search_status', {'status': f"Error reloading TTS model: {str(e)}"})

    socketio.emit('language_setting', {'spanish': spanish_mode})


@socketio.on('clear_context')
def handle_clear_context():
    global messages
    # Reset to just the system message
    system_prompt = get_text("system_prompt")
    messages = [{'role': 'system', 'content': system_prompt}]
    socketio.emit('context_cleared', {})

@app.route('/test_ollama', methods=['GET'])
def test_ollama():
    try:
        # Test simple model list call
        models = client.list()
        return jsonify({"success": True, "models": models})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


# Replace your main loop with this
if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)