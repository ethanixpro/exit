
// backend/server.js
require('dotenv').config();
const express = require('express');
const bodyParser = require('body-parser');
const axios = require('axios');
const cors = require('cors');
const bcrypt = require('bcrypt');
const jwt = require('jsonwebtoken');
const rateLimit = require('express-rate-limit');
const helmet = require('helmet');
const morgan = require('morgan');
const { body, validationResult } = require('express-validator');
const sqlite3 = require('sqlite3').verbose();
const https = require('https');
const fs = require('fs');

// Initialize Express app
const app = express();
const PORT = process.env.PORT || 5000;

// Security headers middleware
app.use(helmet());
app.use(cors());
app.use(bodyParser.json());

// Rate limiting configuration
const apiLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // limit each IP to 100 requests per window
  standardHeaders: true,
  legacyHeaders: false,
  message: 'Too many requests, please try again later'
});

// Apply rate limiter to all API routes
app.use('/api/', apiLimiter);

// Request logging
app.use(morgan('combined'));

// Database setup
const db = new sqlite3.Database('./database.sqlite3', (err) => {
  if (err) {
    console.error('Database connection error:', err.message);
  } else {
    console.log('Connected to SQLite database');
    
    // Create tables
    db.run(`CREATE TABLE IF NOT EXISTS users (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      email TEXT UNIQUE NOT NULL,
      password TEXT NOT NULL,
      phone TEXT,
      subscription TEXT DEFAULT 'free',
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )`);
    
    db.run(`CREATE TABLE IF NOT EXISTS subscriptions (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      user_id INTEGER NOT NULL,
      plan TEXT NOT NULL,
      date DATETIME DEFAULT CURRENT_TIMESTAMP,
      payment_method TEXT,
      FOREIGN KEY (user_id) REFERENCES users(id)
    )`);
    
    db.run(`CREATE TABLE IF NOT EXISTS prediction_cache (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      home_odds REAL NOT NULL,
      draw_odds REAL NOT NULL,
      away_odds REAL NOT NULL,
      response_data TEXT NOT NULL,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )`);
  }
});

// ChatGPT 4.0 Configuration
const CHATGPT_API = process.env.CHATGPT40_API || 'https://api.openai.com/v1/chat/completions';
const CHATGPT_API_KEY = process.env.CHATGPT40_API_KEY || 'sk-proj-jZWLQJYOd6MmBWeORfo1a';

// JWT Configuration
const JWT_SECRET = process.env.JWT_SECRET || 'your_secure_jwt_secret';
const JWT_EXPIRES_IN = process.env.JWT_EXPIRES_IN || '1h';

// AI Method Names - 100 unique AI models
const AIMethods = [
  "ChatGPT-4 Analysis", "Tesla AI Processing", "Sophia AI Reasoning", "Neural Network Model", 
  "Market Sentiment", "Deep Learning Analyzer", "Probabilistic Engine", "Reinforcement Learning Model",
  "Bayesian Network", "Time Series Forecaster", "Gradient Boosting Predictor", "Random Forest Classifier",
  "Monte Carlo Simulator", "Markov Chain Model", "Statistical Inference Engine", "Quantitative Analysis System",
  "Ensemble Learning Model", "Evolutionary Algorithm", "Fuzzy Logic System", "Support Vector Machine",
  "K-Nearest Neighbors", "Decision Tree Classifier", "Logistic Regression Model", "Linear Discriminant Analysis",
  "Naive Bayes Classifier", "Perceptron Network", "Multi-layer Perceptron", "Radial Basis Network",
  "Self-Organizing Map", "Adaptive Resonance Theory", "Hopfield Network", "Boltzmann Machine",
  "Restricted Boltzmann Machine", "Deep Belief Network", "Convolutional Network", "Recurrent Network",
  "Long Short-Term Memory", "Gated Recurrent Unit", "Autoencoder System", "Generative Adversarial Network",
  "Transformer Network", "BERT Analysis", "GPT-3 Prediction", "XLNet Processor",
  "RoBERTa Evaluator", "DistilBERT System", "ALBERT Model", "T5 Predictor",
  "XGBoost Algorithm", "LightGBM Model", "CatBoost Classifier", "AdaBoost System",
  "Stochastic Gradient Descent", "Newton Method Optimizer", "Quasi-Newton Method", "Levenberg-Marquardt",
  "Genetic Algorithm", "Particle Swarm Optimization", "Ant Colony Optimization", "Simulated Annealing",
  "Tabu Search", "Harmony Search", "Cuckoo Search", "Firefly Algorithm",
  "Grey Wolf Optimizer", "Whale Optimization", "Bat Algorithm", "Flower Pollination",
  "K-Means Clustering", "Hierarchical Clustering", "DBSCAN System", "Gaussian Mixture Model",
  "Principal Component Analysis", "Linear Discriminant Analysis", "t-SNE Model", "Autoencoder Reduction",
  "Factor Analysis", "Independent Component Analysis", "Non-negative Matrix Factorization", "Spectral Clustering",
  "Affinity Propagation", "Mean Shift", "OPTICS Algorithm", "BIRCH System",
  "Gaussian Process", "Dirichlet Process", "Hidden Markov Model", "Kalman Filter",
  "Particle Filter", "Causal Inference Engine", "Graph Neural Network", "Capsule Network",
  "Attention Mechanism", "Memory Network", "Differentiable Neural Computer", "Neural Turing Machine",
  "Spatiotemporal Model", "3D Convolution Network", "Graph Convolution Network", "Meta-Learning System"
];

// Authentication middleware
const authenticateToken = (req, res, next) => {
  const authHeader = req.headers['authorization'];
  const token = authHeader && authHeader.split(' ')[1];
  
  if (!token) return res.status(401).json({ error: 'Authentication required' });
  
  jwt.verify(token, JWT_SECRET, (err, user) => {
    if (err) return res.status(403).json({ error: 'Invalid or expired token' });
    req.user = user;
    next();
  });
};

// Check prediction cache
const checkPredictionCache = (homeOdds, drawOdds, awayOdds) => {
  return new Promise((resolve, reject) => {
    const tolerance = 0.01; // 1% odds difference tolerance
    
    db.get(
      `SELECT response_data FROM prediction_cache 
       WHERE ABS(home_odds - ?) <= ? 
         AND ABS(draw_odds - ?) <= ? 
         AND ABS(away_odds - ?) <= ? 
       ORDER BY created_at DESC 
       LIMIT 1`,
      [homeOdds, tolerance, drawOdds, tolerance, awayOdds, tolerance],
      (err, row) => {
        if (err) return reject(err);
        resolve(row ? JSON.parse(row.response_data) : null);
      }
    );
  });
};

// Save prediction to cache
const savePredictionCache = (homeOdds, drawOdds, awayOdds, responseData) => {
  db.run(
    `INSERT INTO prediction_cache (home_odds, draw_odds, away_odds, response_data) 
     VALUES (?, ?, ?, ?)`,
    [homeOdds, drawOdds, awayOdds, JSON.stringify(responseData)],
    (err) => {
      if (err) console.error('Cache save error:', err);
    }
  );
};

// Query ChatGPT 4.0 API
async function queryChatGPT(oddsData, retries = 2) {
  try {
    const prompt = `Given these football match odds: 
      Home: ${oddsData.homeOdds}, 
      Draw: ${oddsData.drawOdds}, 
      Away: ${oddsData.awayOdds}
      
      Provide prediction in JSON format: 
      { "prediction": "home|draw|away", "confidence": 0-100, "reason": "brief explanation" }`;
    
    const response = await axios.post(
      CHATGPT_API,
      {
        model: "gpt-4-turbo",
        messages: [{ role: "user", content: prompt }],
        response_format: { type: "json_object" },
        temperature: 0.2,
        max_tokens: 150
      },
      {
        headers: { 
          Authorization: `Bearer ${CHATGPT_API_KEY}`,
          "Content-Type": "application/json"
        },
        timeout: 10000 // 10-second timeout
      }
    );
    
    // Extract JSON from response
    const content = response.data.choices[0].message.content;
    return JSON.parse(content);
  } catch (error) {
    if (retries > 0) {
      console.log(`Retrying ChatGPT (${retries} left)`);
      await new Promise(resolve => setTimeout(resolve, 2000)); // wait 2 seconds
      return queryChatGPT(oddsData, retries - 1);
    }
    
    console.error('ChatGPT API error:', error.response?.data || error.message);
    return {
      prediction: 'error',
      confidence: 0,
      reason: 'API connection failed'
    };
  }
}

// Generate prediction using ChatGPT and simulated models
app.post('/api/predict', [
  body('homeOdds').isFloat({ min: 1.0 }),
  body('drawOdds').isFloat({ min: 1.0 }),
  body('awayOdds').isFloat({ min: 1.0 }),
  body('matchName').optional().trim().escape(),
  authenticateToken
], async (req, res) => {
  // Validate input
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    return res.status(400).json({ errors: errors.array() });
  }
  
  const { homeOdds, drawOdds, awayOdds, matchName } = req.body;

  try {
    // Check cache first
    const cached = await checkPredictionCache(homeOdds, drawOdds, awayOdds);
    if (cached) {
      console.log('Serving from cache');
      return res.json({ ...cached, cached: true });
    }

    // Get prediction from ChatGPT
    const chatGPTResponse = await queryChatGPT({ homeOdds, drawOdds, awayOdds });
    
    // Validate response
    if (!['home', 'draw', 'away', 'error'].includes(chatGPTResponse.prediction)) {
      chatGPTResponse.prediction = 'error';
    }
    if (typeof chatGPTResponse.confidence !== 'number' || chatGPTResponse.confidence < 0 || chatGPTResponse.confidence > 100) {
      chatGPTResponse.confidence = 0;
    }

    // Create modelResults array with ChatGPT as the first model
    const modelResults = [{
      name: "ChatGPT-4 Turbo",
      prediction: chatGPTResponse.prediction,
      confidence: chatGPTResponse.confidence
    }];

    // Simulate 99 other models with 95% agreement with ChatGPT
    for (let i = 1; i < 100; i++) {
      let modelPrediction;
      if (Math.random() < 0.95) {
        modelPrediction = chatGPTResponse.prediction;
      } else {
        // Randomly select if not agreeing
        const options = ['home', 'away', 'draw'];
        modelPrediction = options[Math.floor(Math.random() * 3)];
      }
      
      modelResults.push({
        name: AIMethods[i],
        prediction: modelPrediction,
        confidence: Math.min(100, Math.max(0, 
          chatGPTResponse.confidence + (Math.random() * 20 - 10) // +/- 10 variation
        ))
      });
    }

    // Count predictions
    const predictions = {
      home: modelResults.filter(m => m.prediction === 'home').length,
      away: modelResults.filter(m => m.prediction === 'away').length,
      draw: modelResults.filter(m => m.prediction === 'draw').length
    };

    // Determine final prediction based on consensus
    let finalPrediction = 'draw';
    if (predictions.home > predictions.away && predictions.home > predictions.draw) {
      finalPrediction = 'home';
    } else if (predictions.away > predictions.home && predictions.away > predictions.draw) {
      finalPrediction = 'away';
    }

    const confidence = Math.max(predictions.home, predictions.away, predictions.draw);

    const resultData = {
      success: true,
      finalPrediction,
      confidence,
      modelResults,
      reason: chatGPTResponse.reason,
      predictions, // home, away, draw counts
      cached: false
    };

    // Save to cache
    savePredictionCache(homeOdds, drawOdds, awayOdds, resultData);
    
    res.json(resultData);
  } catch (error) {
    console.error('Prediction error:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to generate prediction',
      error: error.message
    });
  }
});

// User registration with validation
app.post('/api/register', [
  body('email').isEmail().normalizeEmail(),
  body('password').isLength({ min: 8 }),
  body('phone').isMobilePhone()
], async (req, res) => {
  // Validate input
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    return res.status(400).json({ errors: errors.array() });
  }
  
  const { email, password, phone } = req.body;

  try {
    // Check if user exists
    db.get('SELECT * FROM users WHERE email = ?', [email], async (err, row) => {
      if (err) throw err;
      
      if (row) {
        return res.status(400).json({ error: 'User already exists' });
      }
      
      // Hash password
      const saltRounds = 10;
      const hashedPassword = await bcrypt.hash(password, saltRounds);
      
      // Create new user
      db.run(
        `INSERT INTO users (email, password, phone) 
         VALUES (?, ?, ?)`,
        [email, hashedPassword, phone],
        function (err) {
          if (err) throw err;
          
          const newUser = {
            id: this.lastID,
            email,
            phone,
            subscription: 'free'
          };
          
          // Generate JWT token
          const token = jwt.sign({ userId: newUser.id, email }, JWT_SECRET, {
            expiresIn: JWT_EXPIRES_IN
          });
          
          res.json({ success: true, user: newUser, token });
        }
      );
    });
  } catch (error) {
    console.error('Registration error:', error);
    res.status(500).json({ error: 'Registration failed' });
  }
});

// User login
app.post('/api/login', [
  body('email').isEmail().normalizeEmail(),
  body('password').notEmpty()
], async (req, res) => {
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    return res.status(400).json({ errors: errors.array() });
  }
  
  const { email, password } = req.body;
  
  try {
    db.get('SELECT * FROM users WHERE email = ?', [email], async (err, user) => {
      if (err) throw err;
      
      if (!user) {
        return res.status(401).json({ error: 'Invalid credentials' });
      }
      
      // Compare passwords
      const match = await bcrypt.compare(password, user.password);
      if (!match) {
        return res.status(401).json({ error: 'Invalid credentials' });
      }
      
      // Generate JWT token
      const token = jwt.sign(
        { userId: user.id, email: user.email }, 
        JWT_SECRET, 
        { expiresIn: JWT_EXPIRES_IN }
      );
      
      res.json({
        success: true,
        token,
        user: {
          id: user.id,
          email: user.email,
          subscription: user.subscription
        }
      });
    });
  } catch (error) {
    console.error('Login error:', error);
    res.status(500).json({ error: 'Login failed' });
  }
});

// Subscription handling
app.post('/api/subscribe', [
  body('plan').isIn(['basic', 'premium', 'pro']),
  body('paymentMethod').isIn(['mobile', 'credit_card', 'crypto']),
  authenticateToken
], (req, res) => {
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    return res.status(400).json({ errors: errors.array() });
  }
  
  const { plan, paymentMethod } = req.body;
  const userId = req.user.userId;
  
  try {
    // Update user subscription
    db.run(
      'UPDATE users SET subscription = ? WHERE id = ?',
      [plan, userId],
      function (err) {
        if (err) throw err;
        
        if (this.changes === 0) {
          return res.status(404).json({ error: 'User not found' });
        }
        
        // Record subscription
        db.run(
          `INSERT INTO subscriptions (user_id, plan, payment_method) 
           VALUES (?, ?, ?)`,
          [userId, plan, paymentMethod],
          function (err) {
            if (err) throw err;
            
            res.json({ 
              success: true, 
              subscription: { 
                id: this.lastID,
                plan,
                paymentMethod
              } 
            });
          }
        );
      }
    );
  } catch (error) {
    console.error('Subscription error:', error);
    res.status(500).json({ error: 'Subscription failed' });
  }
});

// Get current user info
app.get('/api/me', authenticateToken, (req, res) => {
  db.get(
    `SELECT id, email, phone, subscription 
     FROM users WHERE id = ?`,
    [req.user.userId],
    (err, user) => {
      if (err) {
        console.error('User fetch error:', err);
        return res.status(500).json({ error: 'Failed to fetch user' });
      }
      
      if (!user) {
        return res.status(404).json({ error: 'User not found' });
      }
      
      res.json({ success: true, user });
    }
  );
});

// Start server
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
```

**Frontend Updates (mt.html):**

1. Update the prediction function to call the backend:

```javascript
// Updated prediction function
async function runAIPrediction(homeOdds, drawOdds, awayOdds, matchName) {
  try {
    const token = localStorage.getItem('authToken');
    
    const response = await fetch('/api/predict', {
      method: 'POST',
      headers: { 
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`
      },
      body: JSON.stringify({ homeOdds, drawOdds, awayOdds, matchName })
    });

    if (response.status === 401) {
      redirectToLogin();
      return;
    }
    
    const data = await response.json();
    
    if (data.success) {
      displayPredictionResults(
        data.finalPrediction, 
        data.confidence,
        data.modelResults,
        data.predictions,
        matchName
      );
    } else {
      console.error('Prediction failed:', data.message);
      alert('Prediction failed: ' + data.message);
    }
  } catch (error) {
    console.error('API Error:', error);
    alert('An error occurred. Please try again.');
  }
}

// Update display function parameters
function displayPredictionResults(prediction, confidence, modelResults, predictions, matchName) {
  // ... existing implementation ...
}
```

2. Add login/signup modals to your HTML (after the footer):

```html
<!-- Login Modal -->
<div class="modal" id="login-modal">
  <div class="modal-content">
    <span class="close-modal" onclick="closeModal('login-modal')">&times;</span>
    <h2>Sign In</h2>
    <div class="form-group">
      <label for="login-email">Email</label>
      <input type="email" class="form-control" id="login-email">
    </div>
    <div class="form-group">
      <label for="login-password">Password</label>
      <input type="password" class="form-control" id="login-password">
    </div>
    <button class="btn" onclick="login()">Sign In</button>
    <p style="margin-top: 15px;">
      Don't have an account? <a href="#" onclick="openModal('signup-modal'); closeModal('login-modal')">Sign Up</a>
    </p>
  </div>
</div>

<!-- Signup Modal -->
<div class="modal" id="signup-modal">
  <div class="modal-content">
    <span class="close-modal" onclick="closeModal('signup-modal')">&times;</span>
    <h2>Create Account</h2>
    <div class="form-group">
      <label for="signup-email">Email</label>
      <input type="email" class="form-control" id="signup-email">
    </div>
    <div class="form-group">
      <label for="signup-password">Password (min 8 characters)</label>
      <input type="password" class="form-control" id="signup-password">
    </div>
    <div class="form-group">
      <label for="signup-phone">Phone Number</label>
      <input type="tel" class="form-control" id="signup-phone">
    </div>
    <button class="btn" onclick="register()">Create Account</button>
  </div>
</div>
```

3. Add authentication functions to your JavaScript:

```javascript
// Authentication functions
async function login() {
  const email = document.getElementById('login-email').value;
  const password = document.getElementById('login-password').value;
  
  try {
    const response = await fetch('/api/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password })
    });
    
    const data = await response.json();
    
    if (data.success) {
      localStorage.setItem('authToken', data.token);
      appState.currentUser = data.user;
      closeModal('login-modal');
      alert('Login successful!');
    } else {
      alert('Login failed: ' + (data.error || 'Invalid credentials'));
    }
  } catch (error) {
    console.error('Login error:', error);
    alert('Login failed. Please try again.');
  }
}

async function register() {
  const email = document.getElementById('signup-email').value;
  const password = document.getElementById('signup-password').value;
  const phone = document.getElementById('signup-phone').value;
  
  try {
    const response = await fetch('/api/register', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password, phone })
    });
    
    const data = await response.json();
    
    if (data.success) {
      localStorage.setItem('authToken', data.token);
      appState.currentUser = data.user;
      closeModal('signup-modal');
      alert('Registration successful! You are now logged in.');
    } else {
      alert('Registration failed: ' + (data.error || 'Please try again'));
    }
  } catch (error) {
    console.error('Registration error:', error);
    alert('Registration failed. Please try again.');
  }
}

function redirectToLogin() {
  alert('Please login to use this feature');
  openModal('login-modal');
}

// Initialize app state with user token
window.addEventListener('load', () => {
  const token = localStorage.getItem('authToken');
  if (token) {
    // Verify token and get user info
    fetch('/api/me', {
      headers: { 'Authorization': `Bearer ${token}` }
    })
    .then(response => response.json())
    .then(data => {
      if (data.success) {
        appState.currentUser = data.user;
      }
    });
  }
  
  // Run demo prediction after 1 second
  setTimeout(() => {
    runAIPrediction(2.10, 3.25, 3.80, "Arsenal vs Chelsea");
  }, 1000);
});