// script.js - Updated for FastAPI backend integration

// Global variables
let currentEmotion = null;
let isRecording = false;
let audioRecorder = null;
let audioChunks = [];

// DOM Elements
const themeToggle = document.getElementById("themeToggle");
const body = document.body;
const emotionState = document.getElementById("emotionState");
const emotionDetectedText = document.getElementById("emotionDetectedText");
const listenBtn = document.getElementById("listenBtn");
const transcriptionBox = document.querySelector(".transcription-box");
const actionButtons = document.querySelectorAll(".action-btn");
const ctx = document.getElementById("emotionGraph").getContext("2d");

// API endpoint base - change this to match your deployed API
const API_BASE = "http://localhost:8000";

// Theme toggle and chart initialization code (keep your existing code)
// ...

// Initialize charts
let emotionChart;
let emotionTimelineChart;

// Emotion data for pie chart
const emotionData = {
  labels: ['Happy', 'Sad', 'Anger', 'Fear'],
  datasets: [{
    data: [0.3, 0.2, 0.15, 0.15],
    backgroundColor: []
  }]
};

const lightColors = ['#ffeb3b', '#ff9800', '#f44336', '#8bc34a'];
const darkColors = ['#ffee58', '#ffb74d', '#ef5350', '#aed581'];

// Update chart colors
function updateChartColors(mode) {
  emotionChart.data.datasets[0].backgroundColor = mode === 'dark' ? darkColors : lightColors;
  emotionChart.options.plugins.legend.labels.color = mode === 'dark' ? 'white' : 'black';
  emotionChart.update();
}

// Initialize Pie Chart
function initChart(mode = 'light') {
  emotionData.datasets[0].backgroundColor = mode === 'dark' ? darkColors : lightColors;
  emotionChart = new Chart(ctx, {
    type: 'pie',
    data: emotionData,
    options: {
      responsive: true,
      plugins: {
        legend: {
          position: 'top',
          labels: {
            color: mode === 'dark' ? 'white' : 'black'
          }
        },
        tooltip: { enabled: true }
      }
    }
  });
}

// Timeline Chart Initialization
function initEmotionTimelineChart(mode = 'light') {
  const timelineCtx = document.getElementById("emotionTimelineChart").getContext("2d");

  if (emotionTimelineChart) {
    emotionTimelineChart.destroy();
  }

  emotionTimelineChart = new Chart(timelineCtx, {
    type: 'line',
    data: {
      labels: ["10 AM", "12 PM", "2 PM", "4 PM", "6 PM"],
      datasets: [
        {
          label: "Happy",
          data: [30, 40, 35, 50, 60],
          borderColor: "#4caf50",
          fill: false
        },
        {
          label: "Sad",
          data: [10, 15, 20, 18, 12],
          borderColor: "#2196f3",
          fill: false
        },
        {
          label: "Anger",
          data: [5, 8, 6, 10, 8],
          borderColor: "#f44336",
          fill: false
        },
        {
          label: "Fear",
          data: [7, 9, 11, 13, 10],
          borderColor: "#9c27b0",
          fill: false
        }
      ]
    },
    options: {
      responsive: true,
      plugins: {
        legend: {
          labels: {
            color: mode === 'dark' ? 'white' : 'black'
          }
        }
      },
      scales: {
        x: {
          ticks: {
            color: mode === 'dark' ? 'white' : 'black'
          }
        },
        y: {
          ticks: {
            color: mode === 'dark' ? 'white' : 'black'
          },
          beginAtZero: true
        }
      }
    }
  });
}

// Update emotion chart based on detected emotion
function updateEmotionChart(emotion) {
  // Reset all values to low
  const baseValues = [0.1, 0.1, 0.1, 0.1];
  
  // Find the index of the detected emotion
  const emotionMap = {
    'Happy': 0,
    'Sad': 1,
    'Anger': 2,
    'Fear': 3,
    // Add mappings for other emotions if needed
  };
  
  const index = emotionMap[emotion] || 0;
  
  // Set the detected emotion to a higher value
  baseValues[index] = 0.7;
  
  // Update chart
  emotionChart.data.datasets[0].data = baseValues;
  emotionChart.update();
  
  // Also update the timeline chart
  updateEmotionTimeline(emotion);
}

// Update emotion timeline
function updateEmotionTimeline(emotion) {
  // Get current time
  const now = new Date();
  const timeStr = now.getHours() + ":" + (now.getMinutes() < 10 ? '0' : '') + now.getMinutes();
  
  // Add new time point
  emotionTimelineChart.data.labels.push(timeStr);
  
  // Ensure we don't have too many data points (keep last 6)
  if (emotionTimelineChart.data.labels.length > 6) {
    emotionTimelineChart.data.labels.shift();
    emotionTimelineChart.data.datasets.forEach(dataset => {
      dataset.data.shift();
    });
  }
  
  // Update each emotion value
  const emotionValues = {
    'Happy': [60, 10, 5, 5],
    'Sad': [10, 60, 15, 10],
    'Anger': [5, 15, 60, 10],
    'Fear': [5, 20, 15, 60],
    // Add mappings for other emotions
  };
  
  const values = emotionValues[emotion] || [25, 25, 25, 25];
  
  // Update datasets
  emotionTimelineChart.data.datasets.forEach((dataset, index) => {
    dataset.data.push(values[index]);
  });
  
  // Update chart
  emotionTimelineChart.update();
}

// Audio recording functions
async function startRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioRecorder = new MediaRecorder(stream);
    audioChunks = [];
    
    audioRecorder.addEventListener('dataavailable', event => {
      audioChunks.push(event.data);
    });
    
    audioRecorder.start();
    return true;
  } catch (error) {
    console.error("Error accessing microphone:", error);
    return false;
  }
}

async function stopRecording() {
  return new Promise((resolve, reject) => {
    if (!audioRecorder) {
      reject("No recording in progress");
      return;
    }
    
    audioRecorder.addEventListener('stop', async () => {
      try {
        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
        await processAudioFile(audioBlob);
        resolve(true);
      } catch (error) {
        reject(error);
      }
    });
    
    audioRecorder.stop();
  });
}

// Process audio file (either recorded or uploaded)
async function processAudioFile(audioBlob) {
  // Create form data to send the audio file
  const formData = new FormData();
  formData.append('audio', audioBlob);
  
  try {
    // Show loading indicator
    transcriptionBox.textContent = "Processing audio...";
    
    // Send to FastAPI backend
    const response = await fetch(`${API_BASE}/api/process-audio`, {
      method: 'POST',
      body: formData
    });
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Server error processing audio');
    }
    
    // Get results
    const result = await response.json();
    
    // Update UI with results
    updateUIWithResults(result);
    
  } catch (error) {
    console.error("Error processing audio:", error);
    transcriptionBox.textContent = "Error processing audio: " + error.message;
  }
}

// Update UI with processing results
function updateUIWithResults(result) {
  // Update transcription
  if (result.translation) {
    transcriptionBox.innerHTML = `
      <p><strong>${result.language}:</strong> ${result.transcription}</p>
      <p><strong>English:</strong> ${result.translation}</p>
    `;
  } else {
    transcriptionBox.innerHTML = `
      <p><strong>${result.language}:</strong> ${result.transcription}</p>
    `;
  }
  
  // Update emotion display
  emotionState.textContent = result.emotion;
  currentEmotion = result.emotion;
  
  // Update emotion chart
  updateEmotionChart(result.emotion);
  
  // If negative emotion, offer intervention
  if (["Sad", "Angry", "Fearful", "Disgusted", "Surprised"].includes(result.emotion)) {
    getRecommendedIntervention(result.emotion);
  } else {
    emotionDetectedText.textContent = "We're here to help if you need us.";
  }
}

// Get recommended intervention from backend
async function getRecommendedIntervention(emotion) {
  try {
    const response = await fetch(`${API_BASE}/api/get-intervention/${emotion.toLowerCase()}`);
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to get intervention');
    }
    
    const data = await response.json();
    const action = data.action;
    
    // Update UI with suggested intervention
    emotionDetectedText.innerHTML = `We detected you might be feeling ${emotion}.<br>Would you like assistance?<br><button id="suggestedAction" class="btn gradient-btn">Try ${action.replace('_', ' ')}</button>`;
    
    // Add click handler for suggested action
    document.getElementById("suggestedAction").addEventListener("click", function() {
      performIntervention(action);
    });
    
  } catch (error) {
    console.error("Error getting intervention:", error);
    emotionDetectedText.textContent = "We're here to help if you need us.";
  }
}

// Perform an intervention
function performIntervention(action) {
  // Display the intervention based on the action
  let interventionContent = '';
  
  switch(action) {
    case 'calming_music':
      interventionContent = `
        <h4>Calming Music</h4>
        <audio controls autoplay>
          <source src="calming_music.mp3" type="audio/mpeg">
          Your browser does not support the audio element.
        </audio>
      `;
      break;
    case 'meditation':
      interventionContent = `
        <h4>Guided Meditation</h4>
        <audio controls autoplay>
          <source src="meditation.mp3" type="audio/mpeg">
          Your browser does not support the audio element.
        </audio>
      `;
      break;
    case 'play_game':
      interventionContent = `
        <h4>Word Shuffle Game</h4>
        <div id="gameArea">
          <p>Unscramble this word: <span id="scrambledWord">amcl</span></p>
          <input type="text" id="wordGuess" placeholder="Enter your guess">
          <button id="checkGuess" class="btn">Check</button>
        </div>
      `;
      break;
  }
  
  // Create a modal for intervention
  const interventionModal = document.createElement('div');
  interventionModal.className = 'intervention-modal';
  interventionModal.innerHTML = `
    <div class="intervention-content">
      ${interventionContent}
      <div class="feedback-section">
        <p>Did this help you feel better?</p>
        <button id="feedbackYes" class="btn">Yes</button>
        <button id="feedbackNo" class="btn">No</button>
      </div>
    </div>
  `;
  
  document.body.appendChild(interventionModal);
  
  // Add event listeners for game if applicable
  if (action === 'play_game') {
    document.getElementById('checkGuess').addEventListener('click', function() {
      const guess = document.getElementById('wordGuess').value.toLowerCase();
      if (guess === 'calm') {
        document.getElementById('gameArea').innerHTML = '<p>✅ Correct! Great job!</p>';
      } else {
        document.getElementById('gameArea').innerHTML = '<p>❌ Not quite. The word was "calm". Try again next time!</p>';
      }
    });
  }
  
  // Add feedback handlers
  document.getElementById('feedbackYes').addEventListener('click', function() {
    updateQTable(currentEmotion.toLowerCase(), action, 2);
    document.body.removeChild(interventionModal);
    emotionDetectedText.textContent = "I'm glad that helped! Is there anything else you need?";
  });
  
  document.getElementById('feedbackNo').addEventListener('click', function() {
    updateQTable(currentEmotion.toLowerCase(), action, -1);
    document.body.removeChild(interventionModal);
    
    // Offer alternatives
    emotionDetectedText.innerHTML = `
      <p>I'm sorry that didn't help. Would you like to:</p>
      <button id="tryAnother" class="btn gradient-btn">Try another approach</button>
      <button id="contactCaregiver" class="btn gradient-btn">Contact caregiver</button>
    `;
    
    document.getElementById('tryAnother').addEventListener('click', function() {
      // Get a different intervention
      getRecommendedIntervention(currentEmotion);
    });
    
    document.getElementById('contactCaregiver').addEventListener('click', function() {
      contactCaregiver();
    });
  });
}

// Update Q-table with feedback
async function updateQTable(emotion, action, reward) {
  try {
    const response = await fetch(`${API_BASE}/api/update-q-table`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        emotion: emotion,
        action: action,
        reward: reward
      })
    });
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to update Q-table');
    }
    
    const data = await response.json();
    console.log('Q-table updated:', data);
    
  } catch (error) {
    console.error('Error updating Q-table:', error);
  }
}

// Contact caregiver
function contactCaregiver() {
  emotionDetectedText.innerHTML = `<p>Contacting caregiver...</p>`;
  
  // Simulate contacting caregiver (in a real app, this would call an API)
  setTimeout(() => {
    emotionDetectedText.innerHTML = `<p>Message sent to caregiver. They will check in with you soon.</p>`;
  }, 2000);
}

// Event listeners
document.addEventListener('DOMContentLoaded', function() {
  // Add theme toggle listeners
  themeToggle.addEventListener("click", () => {
    body.classList.toggle("dark-mode");
    const mode = body.classList.contains("dark-mode") ? 'dark' : 'light';
    themeToggle.textContent = mode === 'dark' ? "🌞 Light Mode" : "🌙 Dark Mode";
    localStorage.setItem("theme", mode);
    updateChartColors(mode);
    initEmotionTimelineChart(mode);
  });
  
  // Add listen button listener
  listenBtn.addEventListener("click", function() {
    if (!isRecording) {
      startListening();
    } else {
      stopListening();
    }
  });
  
  // Add file upload area
  const fileUploadArea = document.createElement('div');
  fileUploadArea.className = 'file-upload-area';
  fileUploadArea.innerHTML = `
    <label for="audioFileUpload" class="file-upload-btn">Upload Audio File</label>
    <input type="file" id="audioFileUpload" accept="audio/*" style="display: none;">
  `;
  
  document.querySelector('.card.section-card:nth-child(3)').appendChild(fileUploadArea);
  
  document.getElementById('audioFileUpload').addEventListener('change', function(e) {
    if (e.target.files.length > 0) {
      processAudioFile(e.target.files[0]);
    }
  });
  
  // Add action button listeners
  actionButtons.forEach((button, index) => {
    button.addEventListener("click", function() {
      const actions = ["calming_music", "meditation", "play_game"];
      if (currentEmotion && ["Sad", "Anger", "Fear", "Disgusted", "Surprised"].includes(currentEmotion)) {
        const action = actions[index];
        performIntervention(action);
      } else {
        alert("No negative emotion detected that requires intervention.");
      }
    });
  });
  
  // Initialize app
  const savedTheme = localStorage.getItem("theme") || 'light';
  if (savedTheme === 'dark') {
    body.classList.add("dark-mode");
    themeToggle.textContent = "🌞 Light Mode";
  }
  
  // Initialize charts
  initChart(savedTheme);
  initEmotionTimelineChart(savedTheme);
});

function startListening() {
  listenBtn.textContent = "Stop Listening";
  listenBtn.classList.add("recording");
  isRecording = true;
  
  startRecording()
    .then(() => {
      transcriptionBox.textContent = "Listening...";
    })
    .catch(error => {
      console.error("Error starting recording:", error);
      alert("Could not access microphone. Please check permissions.");
      stopListening();
    });
}

function stopListening() {
  listenBtn.textContent = "Start Listening";
  listenBtn.classList.remove("recording");
  isRecording = false;
  
  stopRecording()
    .catch(error => {
      console.error("Error processing recording:", error);
    });
}
