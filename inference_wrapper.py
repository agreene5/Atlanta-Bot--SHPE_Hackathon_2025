# inference_wrapper.py
import os
import threading
from queue import Queue
import time
import json
import re
from threading import Lock

# Import necessary components from original script
from melo.api import TTS
import whisper
from ollama import Client
import pygame
import pyaudio
import wave
import numpy as np
import requests
from scipy.spatial.distance import cosine

# Global variables with thread safety
messages = [{'role': 'system', 'content': 'You are Atlanta-bot, a helpful assistant that knows all about Atlanta...'}]
messages_lock = Lock()
EMBEDDINGS_FILE = "embeddings.json"

# Model initialization
print("Loading Whisper model...")
whisper_model = whisper.load_model("tiny")

print("Loading TTS model...")
device = 'cuda'  # or 'cpu'
melo_model = TTS(language="EN", device=device)
speaker_ids = melo_model.hps.data.spk2id

print("Initializing Ollama client...")
client = Client(host='http://localhost:11434')

pygame.mixer.init()

# Load embeddings
print("Loading embeddings database...")
try:
    with open(EMBEDDINGS_FILE, 'r') as f:
        embeddings_database = json.load(f)
    print(f"Loaded {len(embeddings_database)} categories")
except FileNotFoundError:
    print(f"Warning: {EMBEDDINGS_FILE} not found")
    embeddings_database = {}


def get_embedding(text, model="bge-m3:latest"):
    """Get text embedding from Ollama"""
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": model, "prompt": text}
    )
    return response.json()["embedding"] if response.ok else None


def cosine_similarity(vec1, vec2):
    return 1 - cosine(vec1, vec2)


def find_closest_embeddings(user_text, top_n=2, similarity_threshold=0.5):
    user_embedding = get_embedding(user_text)
    if not user_embedding:
        return []

    results = []
    for category, items in embeddings_database.items():
        for item in items:
            similarity = cosine_similarity(user_embedding, item["embedding"])
            if similarity >= similarity_threshold:
                results.append({
                    "category": category,
                    "text": item["text"],
                    "similarity": similarity
                })
    return sorted(results, key=lambda x: x["similarity"], reverse=True)[:top_n]


def prepare_rag_context(user_input):
    results = find_closest_embeddings(user_input)
    if not results:
        context = "No relevant info found. Keep response Atlanta-focused."
    else:
        context = "Relevant information:\n"
        for i, res in enumerate(results, 1):
            context += f"{i}. [{res['category']}] {res['text'][:200]}...\n"
    return user_input, f"{user_input}\n\n{context}"


def text2speech(text, speed=1.0, speaker="EN-BR", output_dir="tts_output"):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"tts_{int(time.time())}.wav")
    melo_model.tts_to_file(text, speaker_ids[speaker], output_path, speed=speed)
    return output_path


def play_audio(file_path):
    pygame.mixer.Sound(file_path).play()
    while pygame.mixer.get_busy():
        pygame.time.wait(100)


def split_into_sentences(text):
    return re.split(r'(?<=[.!?])\s+', text)


def process_tts(text, speaker="EN-BR"):
    sentences = split_into_sentences(text)
    for sentence in sentences:
        if len(sentence.strip()) > 0:
            audio_file = text2speech(sentence, speaker=speaker)
            play_audio(audio_file)
            try:
                os.remove(audio_file)
            except:
                pass


def get_user_input_from_mic():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    SILENCE_LIMIT = 2.5  # seconds
    THRESHOLD = 1500

    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    print("Recording...")
    frames = []
    silent_chunks = 0
    silence_threshold = int(SILENCE_LIMIT * RATE / CHUNK)
    recording = False

    while True:
        data = stream.read(CHUNK)
        audio_data = np.frombuffer(data, dtype=np.int16)
        volume = np.abs(audio_data).mean()

        if volume > THRESHOLD:
            recording = True
            silent_chunks = 0
        elif recording:
            silent_chunks += 1

        if recording:
            frames.append(data)
            if silent_chunks > silence_threshold:
                break

    print("Recording stopped")
    stream.stop_stream()
    stream.close()
    audio.terminate()

    with wave.open("temp.wav", 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    result = whisper_model.transcribe("temp.wav")
    return result["text"].strip()


def process_text_input(user_text):
    global messages
    original_input, augmented_prompt = prepare_rag_context(user_text)

    with messages_lock:
        messages.append({"role": "user", "content": original_input})
        rag_messages = messages.copy()
        rag_messages[-1]["content"] = augmented_prompt

    full_response = ""
    response_buffer = ""

    # Create TTS processing queue
    tts_queue = Queue()
    speaker = "EN-BR"
    session_id = int(time.time())

    # Start TTS worker thread
    tts_thread = threading.Thread(
        target=tts_worker,
        args=(tts_queue, 1.0, speaker, session_id)
    )
    tts_thread.start()

    try:
        # Stream response from LLM
        for chunk in client.chat(
                model='llama3.2:3b',
                messages=rag_messages,
                stream=True
        ):
            content = chunk.get('message', {}).get('content', '')
            full_response += content
            response_buffer += content

            # Process sentences for TTS
            if re.search(r'[.!?]\s*$', response_buffer):
                sentences = split_into_sentences(response_buffer)
                for sentence in sentences[:-1]:
                    tts_queue.put((0, sentence.strip()))
                response_buffer = sentences[-1] if len(sentences) > 1 else ''

        # Process remaining content
        if response_buffer.strip():
            tts_queue.put((0, response_buffer.strip()))

    finally:
        tts_queue.put((None, None))
        tts_thread.join()

    with messages_lock:
        messages.append({"role": "assistant", "content": full_response})

    return full_response


def tts_worker(queue, speed, speaker_id, session_id):
    while True:
        idx, sentence = queue.get()
        if sentence is None:
            break
        try:
            output_file = text2speech(
                sentence,
                speed=speed,
                speaker=speaker_id,
                output_dir=f"tts_{session_id}"
            )
            play_audio(output_file)
            os.remove(output_file)
        except Exception as e:
            print(f"TTS Error: {e}")
        queue.task_done()


def reset_conversation_context():
    global messages
    with messages_lock:
        messages = [messages[0]]  # Keep system message
    return True