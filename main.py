from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import whisper
import numpy as np
import joblib
import json
import os
import librosa
import soundfile
import tempfile
from transformers import pipeline
from typing import Dict, List, Optional, Any

app = FastAPI(title="Emotion Detection API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (in production, specify your frontend domain)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models paths - update these to your actual paths
WHISPER_MODEL_SIZE = "small"
HINDI_MODEL_PATH = "models/mlp_classifier_hindi.model"
ENGLISH_MODEL_PATH = "models/mlp_classifier_english.model"
Q_TABLE_FILE = "models/q_table.json"

# Global variables
interventions = ["calming_music", "play_game", "meditation"]
negative_emotions = ["angry", "sad", "fearful"]
q_table = {}

# Models loading
try:
    whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)
    hindi_emotion_model = joblib.load(HINDI_MODEL_PATH)
    english_emotion_model = joblib.load(ENGLISH_MODEL_PATH)
    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-hi-en")
    print("All models loaded successfully")
except Exception as e:
    print(f"Error loading models: {e}")
    # We'll initialize placeholders and provide more detailed errors in the endpoints

# Pydantic models for API
class QTableUpdate(BaseModel):
    emotion: str
    action: str
    reward: float

class AudioResult(BaseModel):
    language: str
    transcription: str
    translation: Optional[str] = None
    emotion: str

class ActionRecommendation(BaseModel):
    action: str
    q_value: float

# Load Q-table
def load_q_table():
    global q_table
    try:
        if os.path.exists(Q_TABLE_FILE):
            with open(Q_TABLE_FILE, "r") as f:
                q_table = json.load(f)
        else:
            # Initialize new Q-table
            q_table = {emo: {interv: 0.0 for interv in interventions} for emo in negative_emotions}
            with open(Q_TABLE_FILE, "w") as f:
                json.dump(q_table, f)
                
        # Ensure all emotions and interventions exist
        for emo in negative_emotions:
            if emo not in q_table:
                q_table[emo] = {interv: 0.0 for interv in interventions}
            else:
                for interv in interventions:
                    if interv not in q_table[emo]:
                        q_table[emo][interv] = 0.0
        
        return q_table
    except Exception as e:
        print(f"Error loading Q-table: {e}")
        return {emo: {interv: 0.0 for interv in interventions} for emo in negative_emotions}

# Save Q-table
def save_q_table():
    try:
        with open(Q_TABLE_FILE, "w") as f:
            json.dump(q_table, f)
            return True
    except Exception as e:
        print(f"Error saving Q-table: {e}")
        return False

# Extract audio features
def extract_features(file_path):
    try:
        with soundfile.SoundFile(file_path) as sound_file:
            X = sound_file.read(dtype="float32")
            sample_rate = sound_file.samplerate
            
            # Compute STFT
            stft = np.abs(librosa.stft(X))
            
            # Initialize feature vector
            result = np.array([])
            
            # Extract MFCC
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
            
            # Extract Chroma
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
            
            # Extract Mel Spectrogram
            mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
        
        return np.expand_dims(result, axis=0)
    except Exception as e:
        print(f"Error extracting features: {e}")
        raise HTTPException(status_code=500, detail=f"Error extracting audio features: {str(e)}")

# Choose best action based on Q-table
def choose_action(emotion, epsilon=0.1):
    """Choose action based on epsilon-greedy strategy"""
    if emotion not in q_table:
        raise HTTPException(status_code=400, detail=f"Emotion '{emotion}' not found in Q-table")
        
    if np.random.random() < epsilon:
        # Exploration: choose random action
        return np.random.choice(interventions)
    
    # Exploitation: choose best action
    max_val = max(q_table[emotion].values())
    best_actions = [a for a, val in q_table[emotion].items() if val == max_val]
    return np.random.choice(best_actions)

# Initialize on startup
@app.on_event("startup")
async def startup_event():
    load_q_table()
    app.mount("/static", StaticFiles(directory="static"), name="static")

# API endpoints
@app.get("/")
async def root():
    return {"message": "Emotion Detection API is running"}

@app.post("/api/process-audio", response_model=AudioResult)
async def process_audio(audio: UploadFile = File(...)):
    # Ensure the models are loaded
    if 'whisper_model' not in globals():
        raise HTTPException(status_code=500, detail="Whisper model not loaded")
    
    # Save uploaded file temporarily
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    try:
        contents = await audio.read()
        with open(temp_file.name, 'wb') as f:
            f.write(contents)
        
        # Process with Whisper
        transcript_result = whisper_model.transcribe(temp_file.name)
        transcription = transcript_result["text"]
        
        # Detect language (simplified)
        detected_language = "Hindi" if any('\u0900' <= char <= '\u097F' for char in transcription) else "English"
        
        # Translate if Hindi
        translation = None
        if detected_language == "Hindi":
            translation = translator(transcription)[0]['translation_text']
        
        # Extract features and detect emotion
        features = extract_features(temp_file.name)
        model = hindi_emotion_model if detected_language == "Hindi" else english_emotion_model
        emotion = model.predict(features)[0]
        
        return {
            "language": detected_language,
            "transcription": transcription,
            "translation": translation,
            "emotion": emotion
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up the temp file
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

@app.get("/api/get-intervention/{emotion}", response_model=ActionRecommendation)
async def get_intervention(emotion: str):
    emotion = emotion.lower()
    if emotion not in negative_emotions:
        raise HTTPException(status_code=400, detail=f"Emotion '{emotion}' not supported")
    
    action = choose_action(emotion)
    q_value = q_table[emotion][action]
    
    return {"action": action, "q_value": q_value}

@app.post("/api/update-q-table", response_model=Dict[str, Any])
async def update_q_table(update: QTableUpdate):
    emotion = update.emotion.lower()
    if emotion not in q_table:
        raise HTTPException(status_code=400, detail=f"Emotion '{emotion}' not found in Q-table")
    
    if update.action not in interventions:
        raise HTTPException(status_code=400, detail=f"Action '{update.action}' not supported")
    
    # Update Q-value using simple Q-learning
    alpha = 0.2  # Learning rate
    gamma = 0.8  # Discount factor
    current_q = q_table[emotion][update.action]
    max_future_q = max(q_table[emotion].values())
    
    # Q-learning update formula
    new_q = current_q + alpha * (update.reward + gamma * max_future_q - current_q)
    q_table[emotion][update.action] = new_q
    
    # Save updated Q-table
    save_q_table()
    
    return {
        "emotion": emotion,
        "action": update.action,
        "previous_q": current_q,
        "new_q": new_q,
        "success": True
    }

@app.get("/api/get-q-table")
async def get_q_table():
    return q_table

# Run the application
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
