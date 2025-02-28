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

# Set mic_mode to True to use microphone input, False to use text input
mic_mode = False

# Path to embeddings JSON file
EMBEDDINGS_FILE = "embeddings.json"

# Load models once at the beginning and keep them in memory
print("Loading Whisper model...")
whisper_model = whisper.load_model("tiny")

print("Loading TTS model...")
device = 'cuda'  # cpu or cuda
melo_model = TTS(language="EN", device=device)  # ES - Spanish, EN - English
speaker_ids = melo_model.hps.data.spk2id

print("Initializing Ollama client...")
client = Client(host='http://localhost:11434')
messages = [{'role': 'system',
             'content': 'You are Spanish Atlanta-bot, a helpful spanish assistant that knows all about the city of Atlanta, Georiga. You only answer questions relavent to the city of Atlanta, so if the query is irrelevent, ensure you stay on-topic. You will be given information related to the users query if it is within your knowledge-base. If you are given information, ensure you only reference relavent parts related to the users query. Provide brief and helpful responses. You ONLY respond in the Spanish language, even if the user responds in English.'}]
# messages = [{'role': 'system', 'content': 'You are Atlanta-bot, a helpful assistant that knows all about the city of Atlanta, Georiga. You only answer questions relavent to the city of Atlanta, so if the query is irrelevent, ensure you stay on-topic. You will be given information related to the users query if it is within your knowledge-base. If you are given information, ensure you only reference relavent parts related to the users query. Provide brief and helpful responses.'}]

# Initialize pygame mixer once
pygame.mixer.init()

# Pre-load embeddings to keep them in memory
print("Loading embeddings database...")
with open(EMBEDDINGS_FILE, 'r') as f:
    embeddings_database = json.load(f)
print(f"Loaded embeddings for {len(embeddings_database)} categories")


# RAG Functions
def get_embedding(text, model="bge-m3:latest"):
    """Get embeddings from Ollama for the user query"""
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={
            "model": model,
            "prompt": text
        }
    )
    if response.status_code == 200:
        return response.json()["embedding"]
    else:
        print(f"Error: {response.status_code}, {response.text}")
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


def text2speech(text, speed, speaker_id, output_path):
    melo_model.tts_to_file(text, speaker_ids[speaker_id], output_path, speed=speed)
    return output_path


def split_into_sentences(text):
    # Simple sentence splitting - can be improved
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s for s in sentences if s.strip()]


def tts_worker(sentence_queue, speed, speaker_id, i):
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
    print("Recording... (speak now)")
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    frames = []

    # Record for the specified duration
    for _ in range(0, int(RATE / CHUNK * seconds)):
        data = stream.read(CHUNK)
        frames.append(data)

    # Stop recording
    print("Recording finished.")
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
        print("\nPress Enter to start recording...")
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
        print("Recording... (speak now, will stop after 3 seconds of silence)")
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
        print("Recording finished.")
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
        print("Transcribing...")
        result = whisper_model.transcribe(filename)
        transcribed_text = result["text"].strip()

        print(f"User (transcribed): {transcribed_text}")
        return transcribed_text
    else:
        return input("\nUser: ").strip()


def get_first_n_words(text, n=5):
    """Get the first n words from a text"""
    words = text.split()
    return " ".join(words[:n])


def prepare_rag_context(user_input):
    """Prepare RAG context from user input"""
    print("\nSearching knowledge base for relevant information...")
    relevant_results = find_closest_embeddings(user_input, top_n=3, similarity_threshold=0.5)

    if not relevant_results:
        print("No similar embeddings found above threshold (0.5)")
        context = "I don't have specific information about that in my knowledge base. I'll try to help with what I know generally if the query is on-topic."
        augmented_prompt = f"User query: {user_input}\n\nNote: {context}"
    else:
        context = "Based on my knowledge base, I found the following relevant information:\n\n"
        print("Found relevant information:")
        for i, result in enumerate(relevant_results, 1):
            preview = get_first_n_words(result['text'])
            print(f"  {i}. [{result['category']}] \"{preview}...\" (Similarity: {result['similarity']:.4f})")
            context += f"[Source {i} from category '{result['category']}']:\n{result['text']}\n\n"

        augmented_prompt = f"User query: {user_input}\n\n{context}\nPlease use the above information to provide an accurate and helpful response to the user's query."

    return user_input, augmented_prompt


print("\nAll models loaded and ready. Starting conversation loop...")

i = 0
while True:
    user_input = get_user_input()

    # Apply RAG to enhance the prompt with relevant context
    original_input, augmented_prompt = prepare_rag_context(user_input)

    # Add the original user input to the conversation history
    messages.append({'role': 'user', 'content': original_input})

    # Create a temporary messages list with the augmented prompt for the LLM
    rag_messages = messages.copy()
    # Replace the last user message with the augmented one
    rag_messages[-1] = {'role': 'user', 'content': augmented_prompt}

    # Stream the LLM response
    full_response = ""
    buffer = ""
    print("Assistant: ", end="", flush=True)

    # Create a queue for sentences to be processed by TTS
    sentence_queue = Queue()

    # Start TTS worker thread
    speed = 1.0
    speaker_id = "EN_BR"  # or "EN_BR" methinks
    tts_thread = threading.Thread(target=tts_worker, args=(sentence_queue, speed, speaker_id, i))
    tts_thread.start()

    # Process the streaming response
    sentence_idx = 0
    for chunk in client.chat(model='llama3.2:3b', messages=rag_messages, stream=True):
        if 'message' in chunk and 'content' in chunk['message']:
            content = chunk['message']['content']
            print(content, end="", flush=True)

            buffer += content
            full_response += content

            # Check if we have complete sentences in the buffer
            if re.search(r'[.!?]\s', buffer) or re.search(r'[.!?]$', buffer):
                sentences = split_into_sentences(buffer)
                if len(sentences) > 1:  # Keep the last potentially incomplete sentence in buffer
                    for sentence in sentences[:-1]:
                        sentence_queue.put((sentence_idx, sentence))
                        sentence_idx += 1
                    buffer = sentences[-1]
                elif len(sentences) == 1 and buffer.rstrip().endswith(('.', '!', '?')):
                    sentence_queue.put((sentence_idx, sentences[0]))
                    sentence_idx += 1
                    buffer = ""

    # If there's still some text in the buffer, process it
    if buffer.strip():
        sentence_queue.put((sentence_idx, buffer))

    # Signal the TTS thread to terminate
    sentence_queue.put((999, None))

    # Add the LLM's response to the conversation history
    messages.append({'role': 'assistant', 'content': full_response})

    # Wait for TTS to complete
    tts_thread.join()

    i += 1