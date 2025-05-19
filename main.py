from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
import whisper
import numpy as np
import joblib
import os
import json
import tempfile
import librosa
import soundfile
from typing import Dict, List, Optional, Any
import random
from transformers import pipeline
import traceback
from twilio.rest import Client
from pathlib import Path
import pickle
import sys

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="Emotion Detection API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define paths
WHISPER_MODEL_SIZE = "small"
HINDI_MODEL_PATH = "models/mlp_classifier_hindi.model"
ENGLISH_MODEL_PATH = "models/mlp_classifier_english.model"
Q_TABLE_FILE = "models/q_table.json"

# Get Twilio configuration from .env
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")

# Verify Twilio configuration is available
if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER]):
    print("WARNING: Twilio credentials are not properly configured in .env file.")
    print("SMS notifications will not work until these are set.")
    # Set dummy values for development
    if not TWILIO_ACCOUNT_SID:
        TWILIO_ACCOUNT_SID = ""
    if not TWILIO_AUTH_TOKEN:
        TWILIO_AUTH_TOKEN = ""
    if not TWILIO_PHONE_NUMBER:
        TWILIO_PHONE_NUMBER = ""
    print("Using default Twilio credentials for development")

# Debug mode from .env
DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")

# SMS simulation mode
SMS_SIMULATION = os.getenv("SMS_SIMULATION", "False").lower() in ("true", "1", "t")

# Global variables - MODIFIED: Only track specific negative emotions
interventions = ["calming_music", "play_game", "meditation"]
# CHANGED: Only these three negative emotions are detected
target_negative_emotions = ["angry", "fearful", "sad"]
# Keep original lists for model compatibility
negative_emotions = ["angry", "sad", "fearful", "disgusted", "surprised"]
positive_emotions = ["happy", "neutral", "calm"]
all_emotions = negative_emotions + positive_emotions
q_table = {}

# Global model variables
whisper_model = None
hindi_emotion_model = None
english_emotion_model = None
translator = None

# Special function to load scikit-learn models with version compatibility
def load_sklearn_model(model_path):
    """Load scikit-learn model with numpy._core compatibility handling"""
    try:
        # First try normal loading
        return joblib.load(model_path)
    except ModuleNotFoundError as e:
        if "numpy._core" in str(e):
            print("Fixing numpy._core reference...")
            
            # Add a temporary patch for numpy._core
            import numpy
            
            # Create a fake numpy._core module
            class FakeNumPyCore:
                pass
            
            # Add it to sys.modules
            sys.modules['numpy._core'] = FakeNumPyCore()
            
            try:
                # Try loading again with patch
                return joblib.load(model_path)
            except Exception as sub_e:
                print(f"Patched loading also failed: {sub_e}")
                raise
            finally:
                # Clean up the patch
                if 'numpy._core' in sys.modules:
                    del sys.modules['numpy._core']
        else:
            # If it's a different module error, re-raise
            raise
    except Exception as other_e:
        print(f"Standard loading failed: {other_e}")
        raise

# Create a simple emotion model if needed
def create_empty_emotion_model():
    """Create a compatible emotion model from scratch"""
    from sklearn.neural_network import MLPClassifier
    
    # Create a model with the same structure as your trained one
    model = MLPClassifier(
        hidden_layer_sizes=(100,),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        max_iter=200,
        random_state=42
    )
    
    # Initialize with some minimal data to make it usable
    X_dummy = np.random.rand(10, 193)
    y_dummy = np.array(random.choices(negative_emotions + positive_emotions, k=10))
    model.fit(X_dummy, y_dummy)
    
    return model

# Load models
def load_models():
    global whisper_model, hindi_emotion_model, english_emotion_model, translator
    
    try:
        # Load Whisper model
        print("Loading Whisper model...")
        whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)
        print("Whisper model loaded successfully")
        
        # Load Hindi to English translator
        print("Loading translator model...")
        try:
            translator = pipeline("translation", model="Helsinki-NLP/opus-mt-hi-en")
            print("Translator model loaded successfully")
        except Exception as e:
            print(f"Error loading translator: {e}")
            print("Using fallback translator")
            translator = None
        
        # Load emotion models
        print("Loading emotion detection models...")
        
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(HINDI_MODEL_PATH), exist_ok=True)
        
        # Load Hindi emotion model
        try:
            if os.path.exists(HINDI_MODEL_PATH) and os.path.getsize(HINDI_MODEL_PATH) > 0:
                hindi_emotion_model = load_sklearn_model(HINDI_MODEL_PATH)
                print("Hindi emotion model loaded successfully")
            else:
                print(f"Hindi emotion model not found at {HINDI_MODEL_PATH}")
                hindi_emotion_model = create_empty_emotion_model()
                print("Created new Hindi emotion model")
                joblib.dump(hindi_emotion_model, HINDI_MODEL_PATH)
        except Exception as e:
            print(f"Error loading Hindi emotion model: {e}")
            hindi_emotion_model = create_empty_emotion_model()
            print("Created new Hindi emotion model due to loading error")
            joblib.dump(hindi_emotion_model, HINDI_MODEL_PATH)
            
        # Load English emotion model    
        try:
            if os.path.exists(ENGLISH_MODEL_PATH) and os.path.getsize(ENGLISH_MODEL_PATH) > 0:
                english_emotion_model = load_sklearn_model(ENGLISH_MODEL_PATH)
                print("English emotion model loaded successfully")
            else:
                print(f"English emotion model not found at {ENGLISH_MODEL_PATH}")
                english_emotion_model = create_empty_emotion_model()
                print("Created new English emotion model")
                joblib.dump(english_emotion_model, ENGLISH_MODEL_PATH)
        except Exception as e:
            print(f"Error loading English emotion model: {e}")
            english_emotion_model = create_empty_emotion_model()
            print("Created new English emotion model due to loading error")
            joblib.dump(english_emotion_model, ENGLISH_MODEL_PATH)
            
    except Exception as e:
        print(f"Error loading models: {e}")
        print(traceback.format_exc())

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

class SMSNotification(BaseModel):
    patientName: str
    phoneNumber: str
    emotion: str
    timestamp: str
    message: str

# Load Q-table - MODIFIED: Only initialize for target negative emotions
def load_q_table():
    global q_table
    try:
        if os.path.exists(Q_TABLE_FILE):
            with open(Q_TABLE_FILE, "r") as f:
                q_table = json.load(f)
        else:
            # Initialize new Q-table only for target negative emotions
            q_table = {emo: {interv: 0.0 for interv in interventions} for emo in target_negative_emotions}
            os.makedirs(os.path.dirname(Q_TABLE_FILE), exist_ok=True)
            with open(Q_TABLE_FILE, "w") as f:
                json.dump(q_table, f)
                
        # Ensure all target emotions and interventions exist
        for emo in target_negative_emotions:
            if emo not in q_table:
                q_table[emo] = {interv: 0.0 for interv in interventions}
            else:
                for interv in interventions:
                    if interv not in q_table[emo]:
                        q_table[emo][interv] = 0.0
        
        return q_table
    except Exception as e:
        print(f"Error loading Q-table: {e}")
        return {emo: {interv: 0.0 for interv in interventions} for emo in target_negative_emotions}

# Save Q-table
def save_q_table():
    try:
        os.makedirs(os.path.dirname(Q_TABLE_FILE), exist_ok=True)
        with open(Q_TABLE_FILE, "w") as f:
            json.dump(q_table, f)
            return True
    except Exception as e:
        print(f"Error saving Q-table: {e}")
        return False

# Translate Hindi to English
def translate_hindi_to_english(text):
    """Translate Hindi text to English using the transformer model or fallback"""
    try:
        if translator:
            translation = translator(text)[0]['translation_text']
            return translation
        else:
            # Fallback translation - this is just an indicator, not real translation
            return f"[Translation unavailable - please install sentencepiece]: {text}"
    except Exception as e:
        print(f"Error translating: {e}")
        return f"[Translation failed]: {text}"

# Extract audio features for emotion detection
def extract_features(file_path):
    """Extract audio features used for emotion detection"""
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
        print(traceback.format_exc())
        raise e

# MODIFIED: Detect emotion from audio - only return target negative emotions
def detect_emotion(audio_path, language):
    """Detect emotion from audio using the appropriate model"""
    try:
        # Extract features
        features = extract_features(audio_path)
        
        # Use appropriate model based on language
        if language == "Hindi" and hindi_emotion_model is not None:
            model = hindi_emotion_model
        elif language == "English" and english_emotion_model is not None:
            model = english_emotion_model
        else:
            # Error if no model available
            raise ValueError(f"No emotion model available for {language}")
        
        # Predict emotion
        raw_emotion = model.predict(features)[0]
        print(f"Raw detected emotion: {raw_emotion}")
        
        # CHANGED: Only return specific negative emotions, otherwise return "no negative emotions detected"
        if raw_emotion.lower() in target_negative_emotions:
            emotion = raw_emotion.lower()
            print(f"Target negative emotion detected: {emotion}")
            return emotion
        else:
            print(f"Non-target emotion detected ({raw_emotion}), returning 'No negative emotions detected'")
            return "No negative emotions detected"
    
    except Exception as e:
        print(f"Error detecting emotion: {e}")
        print(traceback.format_exc())
        # For robustness in development, return a message indicating no negative emotions
        if DEBUG:
            print("Using 'no negative emotions detected' for development due to error")
            return "No negative emotions detected"
        else:
            # In production, raise the error
            raise

# MODIFIED: Choose best action based on Q-table - handle non-target emotions
def choose_action(emotion, epsilon=0.1):
    """Choose action based on epsilon-greedy strategy"""
    # If emotion is not a target negative emotion, no action needed
    if emotion == "no negative emotions detected" or emotion not in target_negative_emotions:
        return "no_intervention"
    
    if emotion not in q_table:
        # Initialize q_table for this emotion if it doesn't exist
        q_table[emotion] = {interv: 0.0 for interv in interventions}
        
    if np.random.random() < epsilon:
        # Exploration: choose random action
        return random.choice(interventions)
    
    # Exploitation: choose best action
    max_val = max(q_table[emotion].values())
    best_actions = [a for a, val in q_table[emotion].items() if val == max_val]
    return random.choice(best_actions)

# Send SMS function using Twilio
def send_sms(to_number, message):
    """Send SMS using Twilio API"""
    # Check if in simulation mode
    if SMS_SIMULATION:
        print(f"[SIMULATED SMS] To: {to_number}")
        print(f"[SIMULATED SMS] Message: {message}")
        return True, "SIMULATED_" + str(random.randint(10000, 99999))
    
    # Check if Twilio is configured
    if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER]):
        print("ERROR: Twilio is not properly configured. SMS will not be sent.")
        return False, "Twilio not configured. Check .env file and make sure credentials are set."
    
    # Check if to_number is the same as from_number
    if to_number == TWILIO_PHONE_NUMBER:
        print(f"ERROR: Cannot send SMS to the same number as the Twilio number ({to_number})")
        return False, "The caregiver phone number cannot be the same as your Twilio number. Please use a different phone number."
        
    try:
        # Initialize Twilio client
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        
        # Send message
        message = client.messages.create(
            body=message,
            from_=TWILIO_PHONE_NUMBER,
            to=to_number
        )
        
        print(f"SMS sent successfully, SID: {message.sid}")
        return True, message.sid
    except Exception as e:
        print(f"Error sending SMS: {e}")
        print(traceback.format_exc())
        
        # Check for specific Twilio errors
        if "To' and 'From' number cannot be the same" in str(e):
            return False, "The caregiver phone number cannot be the same as the Twilio number. Please use a different phone number."
        
        return False, str(e)

# Initialize on startup
@app.on_event("startup")
async def startup_event():
    # Load models
    load_models()
    
    # Load Q-table
    load_q_table()
    
    # Mount static files
    try:
        app.mount("/static", StaticFiles(directory="static"), name="static")
    except Exception as e:
        print(f"Error mounting static files: {e}")
        # Create static directory if it doesn't exist
        os.makedirs("static", exist_ok=True)
        app.mount("/static", StaticFiles(directory="static"), name="static")

# API endpoints
@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "ok",
        "version": "1.0.0",
        "whisper_model": whisper_model is not None,
        "hindi_emotion_model": hindi_emotion_model is not None,
        "english_emotion_model": english_emotion_model is not None,
        "translator": translator is not None,
        "twilio_configured": all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER]),
        "target_emotions": target_negative_emotions  # Added for clarity
    }

@app.post("/api/process-audio", response_model=AudioResult)
async def process_audio(audio: UploadFile = File(...)):
    # Validate file type
    if not audio.filename.lower().endswith(('.wav', '.mp3', '.m4a', '.mp4')):
        raise HTTPException(status_code=400, detail="Only WAV, MP3, M4A, and MP4 files are supported")
    
    # Save uploaded file temporarily
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio.filename)[1])
    temp_filename = temp_file.name
    
    try:
        # Read and save the uploaded audio
        contents = await audio.read()
        with open(temp_filename, 'wb') as f:
            f.write(contents)
        
        print(f"Saved audio file to {temp_filename}, size: {os.path.getsize(temp_filename)} bytes")
        
        # Process with Whisper
        try:
            # Ensure the whisper model is loaded
            if whisper_model is None:
                raise HTTPException(status_code=500, detail="Whisper model not loaded")
                
            transcript_result = whisper_model.transcribe(temp_filename)
            transcription = transcript_result["text"]
        except Exception as e:
            print(f"Error transcribing audio: {e}")
            if DEBUG:
                # In debug mode, provide a simulated transcription
                transcription = "This is a simulated transcription for development testing."
            else:
                raise HTTPException(status_code=500, detail=f"Error transcribing audio: {str(e)}")
        
        # Detect language (simplified)
        detected_language = "Hindi" if any('\u0900' <= char <= '\u097F' for char in transcription) else "English"
        
        # Translate if Hindi
        translation = None
        if detected_language == "Hindi":
            translation = translate_hindi_to_english(transcription)
        
        # Detect emotion
        try:
            # Check if emotion model is available
            if detected_language == "Hindi" and hindi_emotion_model is None:
                raise HTTPException(status_code=500, detail="Hindi emotion model not available")
            elif detected_language == "English" and english_emotion_model is None:
                raise HTTPException(status_code=500, detail="English emotion model not available")
                
            emotion = detect_emotion(temp_filename, detected_language)
        except Exception as e:
            print(f"Error in emotion detection: {e}")
            if DEBUG:
                # In debug mode, return no negative emotions detected
                emotion = "no negative emotions detected"
                print(f"Using 'no negative emotions detected' for development")
            else:
                raise HTTPException(status_code=500, detail=f"Error detecting emotion: {str(e)}")
        
        # Prepare response
        result = {
            "language": detected_language,
            "transcription": transcription,
            "translation": translation,
            "emotion": emotion
        }
        
        print(f"Processing result: {result}")
        return result
    
    except Exception as e:
        print(f"Error processing audio: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up the temp file - use try/except to handle permission errors
        try:
            if os.path.exists(temp_filename):
                # Close the file if it's still open
                try:
                    temp_file.close()
                except Exception:
                    pass
                
                # Try to delete the file
                try:
                    os.unlink(temp_filename)
                except Exception as e:
                    print(f"Warning: Could not delete temp file immediately: {e}")
        except Exception as e:
            print(f"Warning: Error in file cleanup: {e}")

# MODIFIED: Handle intervention requests for non-target emotions
@app.get("/api/get-intervention/{emotion}", response_model=ActionRecommendation)
async def get_intervention(emotion: str):
    emotion = emotion.lower()
    
    # Handle case where no negative emotions were detected
    if emotion == "no negative emotions detected":
        return {"action": "no_intervention", "q_value": 1.0}
    
    # Only provide interventions for target negative emotions
    if emotion not in target_negative_emotions:
        return {"action": "no_intervention", "q_value": 1.0}
    
    if emotion not in q_table:
        # Initialize q_table for this emotion if it doesn't exist
        q_table[emotion] = {interv: 0.0 for interv in interventions}
    
    action = choose_action(emotion)
    q_value = q_table[emotion].get(action, 0.0) if action != "no_intervention" else 1.0
    
    return {"action": action, "q_value": q_value}

# MODIFIED: Update Q-table only for target emotions
@app.post("/api/update-q-table", response_model=Dict[str, Any])
async def update_q_table(update: QTableUpdate):
    emotion = update.emotion.lower()
    
    # Only allow updates for target negative emotions
    if emotion not in target_negative_emotions:
        raise HTTPException(status_code=400, detail=f"Q-table updates only supported for emotions: {target_negative_emotions}")
    
    # Initialize if emotion not in q_table
    if emotion not in q_table:
        q_table[emotion] = {interv: 0.0 for interv in interventions}
    
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

@app.post("/api/contact-caregiver-sms")
async def contact_caregiver_sms(notification: SMSNotification):
    try:
        # Log the request
        print(f"SMS request received for {notification.patientName}, emotion: {notification.emotion}")
        
        # Only send SMS for target negative emotions
        if notification.emotion.lower() not in target_negative_emotions:
            return {
                "success": False,
                "message": f"SMS notifications are only sent for emotions: {target_negative_emotions}"
            }
        
        # Validate phone number format
        if not notification.phoneNumber.startswith('+'):
            return {
                "success": False,
                "message": "Phone number must be in E.164 format (e.g., +1234567890)"
            }
        
        # Format the message
        message_body = f"ALERT from Emotion Assistant: {notification.patientName} is feeling {notification.emotion} and may need assistance. {notification.message}"
        
        # Send the SMS
        success, message_id = send_sms(notification.phoneNumber, message_body)
        
        if success:
            return {
                "success": True,
                "message": "SMS sent successfully",
                "message_id": message_id
            }
        else:
            return {
                "success": False,
                "message": f"Failed to send SMS: {message_id}"
            }
    
    except Exception as e:
        print(f"Error in /api/contact-caregiver-sms: {e}")
        print(traceback.format_exc())
        return {
            "success": False,
            "message": str(e)
        }

@app.get("/api/get-q-table")
async def get_q_table():
    return q_table

# Run the application
if __name__ == "__main__":
    # Get port from environment variable or use default 8000
    port = int(os.getenv("PORT", 8000))
    
    # Print startup message
    print(f"Starting Emotion Detection API on port {port}")
    print(f"Debug mode: {DEBUG}")
    print(f"SMS simulation mode: {SMS_SIMULATION}")
    print(f"Twilio configured: {all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER])}")
    print(f"Target negative emotions: {target_negative_emotions}")
    
    # Run the application
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=DEBUG)
