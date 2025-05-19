# Major Project API

A FastAPI-based application that processes audio files to detect negative emotions (angry, fearful, sad) and provides intervention recommendations using Q-learning. The system can also send SMS alerts to caregivers when negative emotions are detected.

## 📁 Project Structure

```
MAJOR-PROJECT/
├── __pycache__/                 # Python cache files (auto-generated)
├── models/                      # ML models and Q-table storage
│   ├── mlp_classifier_hindi.model     # Hindi emotion detection model
│   ├── mlp_classifier_english.model   # English emotion detection model
│   └── q_table.json                   # Q-learning reinforcement learning table
├── static/                      # Static web files
│   ├── calming_music.mp3       # Audio file for calming intervention
│   ├── index.html              # Web interface
│   ├── meditation.mp3          # Audio file for meditation intervention
│   ├── script.js               # Frontend JavaScript
│   └── styles.css              # Frontend styling
├── venv/                       # Virtual environment (auto-generated)
├── .env                        # Environment variables (create this file)
├── main.py                     # Main FastAPI application
├── README.md                   # This file
└── requirements.txt            # Python dependencies
```

## 🚀 Features

- **Audio Processing**: Converts speech to text using OpenAI Whisper
- **Language Detection**: Automatically detects Hindi or English
- **Translation**: Translates Hindi to English
- **Emotion Detection**: Detects three negative emotions: angry, fearful, sad
- **Q-Learning**: Learns optimal interventions for each emotion
- **SMS Alerts**: Sends notifications to caregivers via Twilio
- **Web Interface**: User-friendly interface for audio recording/upload

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- A Twilio account (for SMS functionality)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yuvimor/major-project.git
cd major-project
```

### Step 2: Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` doesn't exist, install manually:
```bash
pip install fastapi uvicorn whisper numpy joblib librosa soundfile transformers twilio python-dotenv scikit-learn
```

### Step 4: Create Environment File

Create a `.env` file in the project root directory:

```env
# Twilio Configuration (Required for SMS)
TWILIO_ACCOUNT_SID=your_account_sid_here
TWILIO_AUTH_TOKEN=your_auth_token_here
TWILIO_PHONE_NUMBER=+1234567890

# Debug Mode (Optional)
DEBUG=True

# SMS Simulation Mode (Optional - set to True for testing)
SMS_SIMULATION=False
```

To get Twilio credentials:
1. Sign up at [twilio.com](https://twilio.com)
2. Go to Console Dashboard
3. Copy Account SID and Auth Token
4. Purchase a phone number for `TWILIO_PHONE_NUMBER`

### Step 5: Create Required Directories

```bash
mkdir -p models static
```

## 🎯 Usage

### Starting the Server

```bash
python main.py
```

The server will start on `http://localhost:8000`

### Using the Web Interface

1. Open your browser and go to `http://localhost:8000/static/index.html`
2. Record or upload an audio file (.wav, .mp3, .m4a, .mp4)
3. The system will:
   - Convert speech to text
   - Detect if it's Hindi or English
   - Translate Hindi to English (if needed)
   - Analyze emotion
   - Recommend intervention if negative emotion detected
   - Optionally send SMS alert to caregiver

## API Endpoints

### 1. Health Check

```bash
GET /health
```

Returns system status and model availability.

### 2. Process Audio

```bash
POST /api/process-audio
```

Upload audio file and get transcription + emotion detection.

Example using curl:
```bash
curl -X POST "http://localhost:8000/api/process-audio" \
     -H "Content-Type: multipart/form-data" \
     -F "audio=@your_audio_file.wav"
```

### 3. Get Intervention

```bash
GET /api/get-intervention/{emotion}
```

Get recommended action for a specific emotion.

Example:
```bash
curl "http://localhost:8000/api/get-intervention/angry"
```

### 4. Update Q-Table

```bash
POST /api/update-q-table
```

Update the Q-learning table with reward feedback.

Example:
```bash
curl -X POST "http://localhost:8000/api/update-q-table" \
     -H "Content-Type: application/json" \
     -d '{"emotion": "angry", "action": "calming_music", "reward": 1.0}'
```

### 5. Send SMS Alert

```bash
POST /api/contact-caregiver-sms
```

Send SMS notification to caregiver.

Example:
```bash
curl -X POST "http://localhost:8000/api/contact-caregiver-sms" \
     -H "Content-Type: application/json" \
     -d '{
       "patientName": "John Doe",
       "phoneNumber": "+1234567890",
       "emotion": "angry",
       "timestamp": "2024-01-01T12:00:00Z",
       "message": "Patient needs immediate attention"
     }'
```

### 6. View Q-Table

```bash
GET /api/get-q-table
```

View current Q-learning values.

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `TWILIO_ACCOUNT_SID` | Twilio Account SID | Yes | None |
| `TWILIO_AUTH_TOKEN` | Twilio Auth Token | Yes | None |
| `TWILIO_PHONE_NUMBER` | Twilio Phone Number | Yes | None |
| `DEBUG` | Enable debug mode | No | False |
| `SMS_SIMULATION` | Simulate SMS sending | No | False |
| `PORT` | Server port | No | 8000 |

### Supported Emotions

The system only detects these negative emotions:
- `angry`
- `fearful`
- `sad`

Any other emotion (happy, neutral, disgusted, surprised, etc.) returns "no negative emotions detected".

### Available Interventions

- `calming_music`: Play calming music
- `play_game`: Suggest playing a game
- `meditation`: Guide through meditation

## 🤖 How It Works

### 1. Audio Processing Pipeline

```
Audio File → Whisper (Speech-to-Text) → Language Detection → Translation (if Hindi) → Feature Extraction → Emotion Classification → Intervention Recommendation
```

### 2. Emotion Detection

- Uses pre-trained MLP classifiers for Hindi and English
- Extracts audio features: MFCC, Chroma, Mel Spectrogram
- Only returns angry/fearful/sad emotions
- All other emotions become "no negative emotions detected"

### 3. Q-Learning System

- Learns optimal interventions for each negative emotion
- Updates Q-values based on user feedback
- Uses epsilon-greedy strategy for exploration/exploitation
- Stores Q-table in `models/q_table.json`

### 4. SMS Notification

- Sends alerts only for detected negative emotions
- Includes patient name, emotion, and timestamp
- Requires valid Twilio configuration

## 🐛 Troubleshooting

### Common Issues

**1. Models not loading:**
- Check if `models/` directory exists
- The system will create empty models if files are missing
- For production, place trained models in the `models/` folder

**2. Whisper model downloading:**
- First run downloads the Whisper model (may take time)
- Ensure stable internet connection

**3. SMS not working:**
- Verify Twilio credentials in `.env` file
- Check phone number format (+1234567890)
- Ensure Twilio account has sufficient balance

**4. Audio upload errors:**
- Only .wav, .mp3, .m4a, .mp4 files supported
- Check file permissions and size

**5. Port already in use:**
- Change port in `.env`: `PORT=8001`
- Or kill the process using the port

### Debug Mode

Enable debug mode in `.env`:
```env
DEBUG=True
```

This provides:
- Detailed error messages
- Random emotion fallback if detection fails
- Enhanced logging

### SMS Simulation

For testing without sending real SMS:
```env
SMS_SIMULATION=True
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create Pull Request

## 📞 Support

For issues and questions:
- DO NOT DISTURB ME. EVERYTHING IS ALREADY EXPLAINED IN THE README.
