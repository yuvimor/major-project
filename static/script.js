// Global variables
let currentEmotion = null;
let isRecording = false;
let audioRecorder = null;
let audioChunks = [];
let patientInfo = {
  name: 'Patient',
  caregiverPhone: ''
};

// Twilio phone number - used for validation
const TWILIO_PHONE_NUMBER = "+12183044061";

// DOM Elements
const themeToggle = document.getElementById("themeToggle");
const body = document.body;
const emotionState = document.getElementById("emotionState");
const emotionDetectedText = document.getElementById("emotionDetectedText");
const listenBtn = document.getElementById("listenBtn");
const transcriptionBox = document.getElementById("transcriptionBox");
const recordingStatus = document.getElementById("recordingStatus");
const interventionCard = document.getElementById("interventionCard");
const interventionContent = document.getElementById("interventionContent");
const feedbackArea = document.getElementById("feedbackArea");
const emotionSection = document.getElementById("emotionSection");
const patientInfoModal = document.getElementById("patientInfoModal");
const patientInfoForm = document.getElementById("patientInfoForm");
const skipButton = document.getElementById("skipButton");

// Theme toggle
themeToggle.addEventListener("click", () => {
  body.classList.toggle("dark-mode");
  const mode = body.classList.contains("dark-mode") ? 'dark' : 'light';
  themeToggle.textContent = mode === 'dark' ? "☀️ Light Mode" : "🌙 Dark Mode";
  localStorage.setItem("theme", mode);
});

// Listen button event handler
listenBtn.addEventListener("click", function() {
  if (!isRecording) {
    startRecording();
  } else {
    stopRecording();
  }
});

// Audio file upload handler
document.getElementById('audioFileUpload').addEventListener('change', function(e) {
  if (e.target.files.length > 0) {
    processAudioFile(e.target.files[0]);
  }
});

// Patient info form handler
patientInfoForm.addEventListener('submit', function(e) {
  e.preventDefault();
  patientInfo.name = document.getElementById('patientName').value;
  patientInfo.caregiverPhone = document.getElementById('caregiverPhone').value;
  
  // Basic validation for phone number
  if (!patientInfo.caregiverPhone.startsWith('+')) {
    alert("Phone number must be in international format (start with +)");
    return;
  }
  
  // Check if it matches the Twilio number
  if (patientInfo.caregiverPhone === TWILIO_PHONE_NUMBER) {
    alert("The caregiver phone number cannot be the same as the Twilio number.");
    return;
  }
  
  // Save to localStorage only if remember checkbox is checked
  if (document.getElementById('rememberInfo').checked) {
    localStorage.setItem('patientInfo', JSON.stringify(patientInfo));
  } else {
    // Clear localStorage if not remembering
    localStorage.removeItem('patientInfo');
  }
  
  // Hide the modal
  patientInfoModal.style.display = 'none';
});

// Skip button handler
skipButton.addEventListener('click', function() {
  // Use default values for testing
  patientInfo.name = document.getElementById('patientName').value || 'Test Patient';
  patientInfo.caregiverPhone = document.getElementById('caregiverPhone').value || '+10000000000';
  
  // Hide the modal
  patientInfoModal.style.display = 'none';
});

// Start recording audio
async function startRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioRecorder = new MediaRecorder(stream);
    audioChunks = [];
    
    audioRecorder.addEventListener('dataavailable', event => {
      audioChunks.push(event.data);
    });
    
    audioRecorder.start();
    isRecording = true;
    listenBtn.textContent = "Stop Recording";
    listenBtn.classList.add("recording");
    recordingStatus.textContent = "Recording in progress...";
    
  } catch (error) {
    console.error("Error accessing microphone:", error);
    alert("Could not access microphone. Please check permissions.");
  }
}

// Stop recording and process audio
async function stopRecording() {
  if (!audioRecorder) return;
  
  return new Promise((resolve) => {
    audioRecorder.addEventListener('stop', async () => {
      try {
        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
        await processAudioFile(audioBlob);
        resolve();
      } catch (error) {
        console.error("Error processing audio:", error);
        recordingStatus.textContent = "Error processing audio.";
        resolve();
      }
    });
    
    audioRecorder.stop();
    isRecording = false;
    listenBtn.textContent = "Start Recording";
    listenBtn.classList.remove("recording");
  });
}

// Process audio file (uploaded or recorded)
async function processAudioFile(audioBlob) {
  // Create form data to send the audio file
  const formData = new FormData();
  formData.append('audio', audioBlob);
  
  try {
    // Show loading indicator
    transcriptionBox.textContent = "Processing audio...";
    recordingStatus.textContent = "Analyzing audio...";
    
    // Send to backend API
    const response = await fetch('/api/process-audio', {
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
    recordingStatus.textContent = "";
  }
}

// Update UI with processing results
function updateUIWithResults(result) {
  // Clear recording status
  recordingStatus.textContent = "";
  
  // Update transcription
  if (result.translation) {
    transcriptionBox.innerHTML = `
      <p><strong>Original (${result.language}):</strong> ${result.transcription}</p>
      <p><strong>English:</strong> ${result.translation}</p>
    `;
  } else {
    transcriptionBox.innerHTML = `
      <p><strong>${result.language}:</strong> ${result.transcription}</p>
    `;
  }
  
  // Update emotion display
  const emotionDisplay = result.emotion.charAt(0).toUpperCase() + result.emotion.slice(1);
  emotionState.textContent = emotionDisplay;
  currentEmotion = result.emotion;
  
  // Show emotion section
  emotionSection.style.display = "block";
  
  // Update emotion text
  emotionDetectedText.textContent = getEmotionMessage(result.emotion);
  
  // If negative emotion, get intervention
  const negativeEmotions = ["sad", "angry", "fearful", "disgusted", "surprised"];
  if (negativeEmotions.includes(result.emotion.toLowerCase())) {
    getRecommendedIntervention(result.emotion);
  } else {
    // Hide intervention card for positive emotions
    interventionCard.style.display = "none";
  }
}

// Get customized message based on detected emotion
function getEmotionMessage(emotion) {
  const messages = {
    'happy': "You seem to be in a good mood!",
    'sad': "You seem to be feeling down. Would you like some assistance?",
    'angry': "You appear to be upset. Let's try to help you feel better.",
    'fearful': "You seem to be anxious. We can help you calm down.",
    'disgusted': "You seem to be uncomfortable. Let's try to improve your mood.",
    'surprised': "You seem startled. Let's help you process this.",
    'neutral': "You seem to be feeling neutral.",
    'calm': "You seem to be in a calm state."
  };
  
  return messages[emotion.toLowerCase()] || "We're here to help.";
}

// Get recommended intervention from backend
async function getRecommendedIntervention(emotion) {
  try {
    const response = await fetch(`/api/get-intervention/${emotion.toLowerCase()}`);
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to get intervention');
    }
    
    const data = await response.json();
    const action = data.action;
    
    // If no intervention needed, return
    if (action === "no_intervention") {
      interventionCard.style.display = "none";
      return;
    }
    
    // Show intervention card
    interventionCard.style.display = "block";
    
    // Perform intervention
    performIntervention(action);
    
  } catch (error) {
    console.error("Error getting intervention:", error);
    emotionDetectedText.textContent = "We're here to help if you need us.";
    interventionCard.style.display = "none";
  }
}

// Perform an intervention
function performIntervention(action) {
  // Display the intervention based on the action
  let interventionHtml = '';
  
  switch(action) {
    case 'calming_music':
      interventionHtml = `
        <h3>Calming Music</h3>
        <p>Let's listen to some calming music to help you relax.</p>
        <audio controls autoplay>
          <source src="/static/calming_music.mp3" type="audio/mpeg">
          Your browser does not support the audio element.
        </audio>
      `;
      break;
    case 'meditation':
      interventionHtml = `
        <h3>Guided Meditation</h3>
        <p>Let's try a brief guided meditation to help you feel better.</p>
        <audio controls autoplay>
          <source src="/static/meditation.mp3" type="audio/mpeg">
          Your browser does not support the audio element.
        </audio>
      `;
      break;
    case 'play_game':
      interventionHtml = `
        <h3>Tic Tac Toe Game</h3>
        <p>Let's play a quick game to shift your focus.</p>
        <div id="ticTacToeGame">
          <div class="game-board">
            <div class="cell" data-cell-index="0"></div>
            <div class="cell" data-cell-index="1"></div>
            <div class="cell" data-cell-index="2"></div>
            <div class="cell" data-cell-index="3"></div>
            <div class="cell" data-cell-index="4"></div>
            <div class="cell" data-cell-index="5"></div>
            <div class="cell" data-cell-index="6"></div>
            <div class="cell" data-cell-index="7"></div>
            <div class="cell" data-cell-index="8"></div>
          </div>
          <div class="game-status">Your turn (X). Click on a cell to play.</div>
          <button id="resetGame" class="btn">Reset Game</button>
        </div>
      `;
      break;
    default:
      interventionHtml = `
        <h3>Assistance</h3>
        <p>We're here to help you feel better. Let's try some deep breathing together.</p>
      `;
  }
  
  // Display intervention
  interventionContent.innerHTML = interventionHtml;
  
  // Add feedback buttons
  feedbackArea.innerHTML = `
    <p>Did this help you feel better?</p>
    <div class="feedback-buttons">
      <button id="feedbackYes" class="btn gradient-btn">Yes, it helped</button>
      <button id="feedbackNo" class="btn gradient-btn">No, not really</button>
    </div>
  `;
  
  // Add Tic Tac Toe game logic if that's the intervention
  if (action === 'play_game') {
    initTicTacToe();
  }
  
  // Add feedback handlers
  document.getElementById('feedbackYes').addEventListener('click', function() {
    handleFeedback(currentEmotion, action, true);
  });
  
  document.getElementById('feedbackNo').addEventListener('click', function() {
    handleFeedback(currentEmotion, action, false);
  });
}

// Handle user feedback on intervention
async function handleFeedback(emotion, action, positive) {
  try {
    // Update Q-table
    const response = await fetch('/api/update-q-table', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        emotion: emotion.toLowerCase(),
        action: action,
        reward: positive ? 2 : -1
      })
    });
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to update Q-table');
    }
    
    if (positive) {
      // Positive feedback
      feedbackArea.innerHTML = `
        <p>I'm glad that helped! Is there anything else you need?</p>
      `;
    } else {
      // Negative feedback
      feedbackArea.innerHTML = `
        <p>I'm sorry that didn't help. Would you like to:</p>
        <div class="feedback-buttons">
          <button id="tryAnother" class="btn gradient-btn">Try another approach</button>
          <button id="contactCaregiver" class="btn gradient-btn">Contact caregiver</button>
        </div>
      `;
      
      document.getElementById('tryAnother').addEventListener('click', function() {
        // Get a different intervention
        getRecommendedIntervention(emotion);
      });
      
      document.getElementById('contactCaregiver').addEventListener('click', function() {
        contactCaregiver();
      });
    }
    
  } catch (error) {
    console.error('Error updating Q-table:', error);
    feedbackArea.innerHTML = `<p>There was an error processing your feedback. Please try again.</p>`;
  }
}

// Contact caregiver function with SMS
async function contactCaregiver() {
  interventionContent.innerHTML = `
    <h3>Contacting Caregiver</h3>
    <p>Sending message to caregiver...</p>
  `;
  
  feedbackArea.innerHTML = '';
  
  try {
    // Check if phone number is the same as Twilio number (client-side validation)
    if (patientInfo.caregiverPhone === TWILIO_PHONE_NUMBER) {
      throw new Error("The caregiver phone number cannot be the same as your Twilio number.");
    }
    
    // Send SMS notification to caregiver
    const response = await fetch('/api/contact-caregiver-sms', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        patientName: patientInfo.name,
        phoneNumber: patientInfo.caregiverPhone,
        emotion: currentEmotion,
        timestamp: new Date().toISOString(),
        message: `${patientInfo.name} is currently feeling ${currentEmotion} and may need assistance.`
      })
    });
    
    const result = await response.json();
    
    if (result.success) {
      interventionContent.innerHTML = `
        <h3>Caregiver Notified</h3>
        <p>Your caregiver has been notified via SMS and will check in with you soon.</p>
        <div class="sms-sent-alert">SMS sent successfully!</div>
        <p>In the meantime, would you like to try some deep breathing exercises?</p>
        <button id="tryBreathing" class="btn gradient-btn">Try breathing exercises</button>
      `;
    } else {
      interventionContent.innerHTML = `
        <h3>Notification Error</h3>
        <p>There was an issue contacting your caregiver.</p>
        <div class="sms-error-alert">Error: ${result.message}</div>
        <p>Would you like to try a different intervention instead?</p>
        <button id="tryDifferent" class="btn gradient-btn">Try different intervention</button>
      `;
      
      document.getElementById('tryDifferent').addEventListener('click', function() {
        getRecommendedIntervention(currentEmotion);
      });
    }
    
    // Add breathing exercise functionality if successful
    if (result.success) {
      document.getElementById('tryBreathing').addEventListener('click', function() {
        startBreathingExercise();
      });
    }
    
  } catch (error) {
    console.error('Error contacting caregiver:', error);
    interventionContent.innerHTML = `
      <h3>Notification Error</h3>
      <p>There was an issue contacting your caregiver.</p>
      <div class="sms-error-alert">Error: ${error.message}</div>
      <p>Would you like to try a different intervention instead?</p>
      <button id="tryDifferent" class="btn gradient-btn">Try different intervention</button>
    `;
    
    document.getElementById('tryDifferent').addEventListener('click', function() {
      getRecommendedIntervention(currentEmotion);
    });
  }
}

// Start breathing exercise
function startBreathingExercise() {
  interventionContent.innerHTML = `
    <h3>Deep Breathing</h3>
    <p>Let's try some deep breathing together:</p>
    <div class="breathing-exercise">
      <p id="breathingInstruction">Breathe in deeply...</p>
      <div class="breathing-circle"></div>
    </div>
  `;
  
  // Simple breathing exercise animation
  let count = 0;
  const breathingInstructions = [
    "Breathe in deeply...",
    "Hold your breath...",
    "Breathe out slowly...",
    "Relax..."
  ];
  
  const breathingInterval = setInterval(() => {
    document.getElementById('breathingInstruction').textContent = breathingInstructions[count % 4];
    count++;
    
    if (count > 16) {  // 4 full cycles
      clearInterval(breathingInterval);
      interventionContent.innerHTML += `
        <p>Great job! Do you feel more relaxed now?</p>
      `;
    }
  }, 3000);
}

// Tic Tac Toe game logic
function initTicTacToe() {
  let currentPlayer = 'X';
  const cells = document.querySelectorAll('.cell');
  const gameStatus = document.querySelector('.game-status');
  const resetButton = document.getElementById('resetGame');
  let gameActive = true;
  let gameState = ['', '', '', '', '', '', '', '', ''];
  
  const winningConditions = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8], // rows
    [0, 3, 6], [1, 4, 7], [2, 5, 8], // columns
    [0, 4, 8], [2, 4, 6]             // diagonals
  ];
  
  // Cell clicked function
  function handleCellClick(e) {
    const clickedCell = e.target;
    const clickedCellIndex = parseInt(clickedCell.getAttribute('data-cell-index'));
    
    // Check if cell already played or game over
    if (gameState[clickedCellIndex] !== '' || !gameActive) {
      return;
    }
    
    // Update the game state
    gameState[clickedCellIndex] = currentPlayer;
    clickedCell.textContent = currentPlayer;
    clickedCell.classList.add(currentPlayer === 'X' ? 'player-x' : 'player-o');
    
    // Check for win or draw
    checkResult();
    
    // Computer's turn (simple AI)
    if (gameActive && currentPlayer === 'X') {
      currentPlayer = 'O';
      gameStatus.textContent = "Computer's turn (O)...";
      
      // Slight delay for computer's move
      setTimeout(() => {
        makeComputerMove();
        checkResult();
        currentPlayer = 'X';
        if (gameActive) {
          gameStatus.textContent = "Your turn (X). Click on a cell to play.";
        }
      }, 700);
    }
  }
  
  // Simple AI for computer moves
  function makeComputerMove() {
    // Try to win if possible
    for (let i = 0; i < gameState.length; i++) {
      if (gameState[i] === '') {
        gameState[i] = 'O';
        if (checkWin('O')) {
          cells[i].textContent = 'O';
          cells[i].classList.add('player-o');
          return;
        }
        gameState[i] = '';
      }
    }
    
    // Block player from winning
    for (let i = 0; i < gameState.length; i++) {
      if (gameState[i] === '') {
        gameState[i] = 'X';
        if (checkWin('X')) {
          gameState[i] = 'O';
          cells[i].textContent = 'O';
          cells[i].classList.add('player-o');
          return;
        }
        gameState[i] = '';
      }
    }
    
    // Try center if available
    if (gameState[4] === '') {
      gameState[4] = 'O';
      cells[4].textContent = 'O';
      cells[4].classList.add('player-o');
      return;
    }
    
    // Just pick a random empty cell
    const emptyCells = [];
    for (let i = 0; i < gameState.length; i++) {
      if (gameState[i] === '') {
        emptyCells.push(i);
      }
    }
    
    if (emptyCells.length > 0) {
      const randomIndex = Math.floor(Math.random() * emptyCells.length);
      const cellIndex = emptyCells[randomIndex];
      gameState[cellIndex] = 'O';
      cells[cellIndex].textContent = 'O';
      cells[cellIndex].classList.add('player-o');
    }
  }
  
  // Check if a player has won
  function checkWin(player) {
    return winningConditions.some(condition => {
      return condition.every(index => {
        return gameState[index] === player;
      });
    });
  }
  
  // Check result after a move
  function checkResult() {
    let roundWon = checkWin(currentPlayer);
    
    if (roundWon) {
      gameStatus.textContent = currentPlayer === 'X' ? "You won!" : "Computer won!";
      gameActive = false;
      return;
    }
    
    // Check for draw
    let roundDraw = !gameState.includes('');
    if (roundDraw) {
      gameStatus.textContent = "Game ended in a draw!";
      gameActive = false;
      return;
    }
  }
  
  // Reset game
  function resetGame() {
    gameActive = true;
    currentPlayer = 'X';
    gameState = ['', '', '', '', '', '', '', '', ''];
    gameStatus.textContent = "Your turn (X). Click on a cell to play.";
    cells.forEach(cell => {
      cell.textContent = '';
      cell.classList.remove('player-x', 'player-o');
    });
  }
  
  // Add event listeners
  cells.forEach(cell => {
    cell.addEventListener('click', handleCellClick);
  });
  
  resetButton.addEventListener('click', resetGame);
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
  // Load saved theme
  const savedTheme = localStorage.getItem("theme") || 'light';
  if (savedTheme === 'dark') {
    body.classList.add("dark-mode");
    themeToggle.textContent = "☀️ Light Mode";
  }
  
  // Load saved patient info
  const savedPatientInfo = localStorage.getItem('patientInfo');
  if (savedPatientInfo) {
    patientInfo = JSON.parse(savedPatientInfo);
    
    // Pre-fill the form with saved data
    document.getElementById('patientName').value = patientInfo.name || '';
    document.getElementById('caregiverPhone').value = patientInfo.caregiverPhone || '';
  }
  
  // Always show patient info modal on page load
  patientInfoModal.style.display = 'flex';
  
  // Hide emotion section initially
  emotionSection.style.display = "none";
  
  // Hide intervention card initially
  interventionCard.style.display = "none";
});
