# 📁 Project File Structure & Summary

## ✅ All Critical Files Intact - Project Ready to Run!

---

## 🎯 Core Application Files

### **Backend (Python/Flask)**

| File | Purpose | Status |
|------|---------|--------|
| **app.py** | Main Flask server, API endpoints (/predict, /health), model loading, audio processing | ✅ Present |
| **train_parallel_cnn.py** | Model architecture (ParallelCNN), training utilities, data processing | ✅ Present |
| **train_model_torch.py** | Training helper functions (choose_device, set_seed) | ✅ Present |
| **requirements.txt** | Python dependencies (PyTorch, Flask, librosa, OpenL3) | ✅ Present |

### **Pre-trained Model**

| File | Purpose | Size | Status |
|------|---------|------|--------|
| **torch_models/parallel_genre_classifier_torch.pt** | Trained model checkpoint with 88.83% test accuracy | 56 MB | ✅ Present |

### **Frontend (React/Vite)**

| File | Purpose | Status |
|------|---------|--------|
| **frontend/package.json** | npm dependencies and scripts | ✅ Present |
| **frontend/vite.config.js** | Vite build configuration (port 3000, proxy setup) | ✅ Present |
| **frontend/index.html** | Main HTML entry point | ✅ Present |
| **frontend/src/main.jsx** | React app entry point | ✅ Present |
| **frontend/src/App.jsx** | Main app component with routing | ✅ Present |

---

## 📂 Detailed File Breakdown

### **Root Directory Files**

#### **Backend Core**
```
app.py                          # Flask API server (31.5 KB)
├─ Endpoints:
│  ├─ POST /predict            # Audio analysis endpoint
│  └─ GET /health              # Health check endpoint
├─ Features:
│  ├─ Audio loading & validation (MP3, WAV, FLAC, M4A, OGG)
│  ├─ 3-second segment processing (up to 30 seconds)
│  ├─ Model inference with GPU acceleration
│  ├─ Spectrogram generation (Base64)
│  ├─ SNR calculation
│  └─ Audio quality checks (noise detection)
```

```
train_parallel_cnn.py           # Model architecture (35 KB)
├─ Classes:
│  ├─ ParallelCNN               # Main model class
│  ├─ MelSliceEncoder           # Mel-spectrogram encoder
│  ├─ EmbeddingSliceEncoder     # OpenL3 embedding encoder
│  ├─ DatasetConfig             # Configuration dataclass
│  └─ OpenL3Config              # OpenL3 settings
├─ Functions:
│  ├─ compute_mel_slices()      # Mel-spectrogram computation
│  ├─ load_openl3_model()       # Load OpenL3 embeddings
│  └─ choose_device()           # GPU/CPU selection
```

```
train_model_torch.py            # Training utilities (19 KB)
├─ Functions:
│  ├─ choose_device()           # Device selection helper
│  ├─ set_seed()                # Reproducibility seeding
│  └─ split_datasets()          # Train/val/test split
```

#### **Configuration & Setup**
```
requirements.txt                # Python dependencies
├─ Core:
│  ├─ torch==2.6.0+cu124       # PyTorch with CUDA 12.4
│  ├─ librosa>=0.10.0          # Audio processing
│  ├─ openl3>=0.4.1            # Pre-trained embeddings
│  ├─ flask>=3.0.0             # Web framework
│  └─ numpy, scikit-learn, matplotlib, pillow

.gitignore                      # Git ignore rules
├─ Excludes: .venv, node_modules, __pycache__, *.pyc
├─ Tracks: torch_models/*.pt (exception rule)
```

```
install_dependencies.ps1        # Automated dependency installer (PowerShell)
setup_python310_env.ps1/sh      # Python 3.10 environment setup scripts
START_BACKEND.ps1               # Quick backend start script
```

#### **Model & Cache**
```
torch_models/
└─ parallel_genre_classifier_torch.pt   # Pre-trained model (56 MB)
   ├─ Architecture: Parallel CNN + OpenL3
   ├─ Accuracy: 88.83% test, 83.73% validation
   ├─ Genres: 10 classes
   └─ Feature type: OpenL3 (512-dim embeddings)

torch_cache/                    # Cached preprocessed features
uploads/                        # Temporary audio uploads
__pycache__/                    # Python bytecode cache
```

#### **Data**
```
Data/                           # Dataset directory (GTZAN)
genres_original/                # Original genre folders
archive/                        # Archived/backup files
data.json                       # Legacy preprocessed data
```

---

### **Frontend Directory Structure**

```
frontend/
├─ package.json                 # npm dependencies
│  ├─ react: 18.2.0
│  ├─ vite: 7.2.2
│  ├─ tailwindcss: 3.3.6
│  ├─ chart.js: 4.4.0
│  ├─ wavesurfer.js: 7.6.0
│  └─ framer-motion, react-router-dom, react-hot-toast
│
├─ vite.config.js               # Vite configuration
│  ├─ Port: 3000
│  ├─ Proxy: /api → localhost:5000
│  └─ React plugin setup
│
├─ tailwind.config.js           # Tailwind CSS configuration
├─ postcss.config.js            # PostCSS configuration
├─ index.html                   # HTML entry point
│
├─ src/
│  ├─ main.jsx                  # React entry point
│  ├─ App.jsx                   # Main app with routing
│  ├─ index.css                 # Global styles (Tailwind imports)
│  │
│  ├─ pages/                    # Route pages
│  │  ├─ Landing.jsx            # Home page
│  │  ├─ Login.jsx              # Authentication page
│  │  ├─ Analysis.jsx           # Audio analysis page (main feature)
│  │  └─ History.jsx            # Analysis history page
│  │
│  ├─ components/               # Reusable components
│  │  ├─ AudioInput.jsx         # File upload & recording
│  │  ├─ ResultsDisplay.jsx     # Genre predictions display
│  │  ├─ WaveformTimeline.jsx   # Audio waveform player
│  │  ├─ ResultsDashboard.jsx   # Metrics visualization
│  │  ├─ Header.jsx             # Navigation header
│  │  ├─ Footer.jsx             # Footer component
│  │  └─ ThemeToggle.jsx        # Dark/light mode toggle
│  │
│  ├─ context/                  # React Context API
│  │  ├─ AuthContext.jsx        # Authentication state
│  │  └─ ThemeContext.jsx       # Theme state (dark/light)
│  │
│  ├─ config/                   # Configuration
│  │  └─ api.js                 # API endpoints configuration
│  │
│  └─ utils/                    # Utility functions
│     ├─ pdfExport.js           # PDF report generation
│     └─ validation.js          # Input validation helpers
│
├─ dist/                        # Production build output
└─ node_modules/                # npm packages
```

---

## 📚 Documentation Files

| File | Purpose |
|------|---------|
| **README.md** | Main project documentation with setup instructions |
| **README_WEB_APP.md** | Web application specific guide |
| **PROJECT_ANALYSIS.md** | Comprehensive project structure analysis |
| **SETUP_GUIDE.md** | Detailed setup instructions |
| **QUICK_START.md** | Quick start guide |
| **TECH_STACK_ANALYSIS.md** | Complete technology stack breakdown |
| **MODEL_PERFORMANCE_METRICS.md** | Model accuracy and performance metrics |
| **INSTALL_OPENL3.md** | OpenL3 installation guide |
| **SETUP_WITH_OPENL3.md** | Setup with OpenL3 features |
| **RUN_COMMANDS.md** | Quick reference for run commands |
| **BUG_FIXES.md** | Bug fixes and solutions |
| **EDGE_CASE_FIXES.md** | Edge case handling documentation |
| **FEATURES_IMPLEMENTED.md** | List of implemented features |
| **IMPROVED_NOISE_DETECTION.md** | Noise detection improvements |
| **NOISE_AND_TRAFFIC_DETECTION.md** | Traffic sound detection |
| **USER_HISTORY_FIXES.md** | History feature fixes |
| **REPOSITORY_STATUS.md** | Repository status and deployment info |

---

## 🔧 Utility Files

| File | Purpose |
|------|---------|
| **check_metrics.py** | Script to extract model metrics from checkpoint |
| **genre_classifier.keras** | Legacy TensorFlow model (not used in current version) |
| **tmp_*.wav** | Temporary test audio files |

---

## 📊 What Each Key File Does

### **app.py - Backend Server (Most Important)**
```python
# Functions:
load_model()                    # Load PyTorch model and config
allowed_file()                  # Validate file extensions
check_audio_quality()           # SNR, noise, traffic detection
compute_snr()                   # Signal-to-noise ratio calculation
prepare_audio_for_segmentation() # Limit to 30s, loop if short
process_3s_segments()           # Split audio into 3-second segments
generate_spectrogram_image()    # Create mel-spectrogram visualization

# Endpoints:
@app.route('/health')           # Health check
@app.route('/predict')          # Audio analysis (main endpoint)

# Configuration:
MODEL_PATH                      # torch_models/parallel_genre_classifier_torch.pt
MAX_FILE_SIZE                   # 50 MB limit
MAX_ANALYSIS_DURATION           # 30 seconds
ALLOWED_EXTENSIONS              # mp3, wav, flac, m4a, ogg
```

### **frontend/src/pages/Analysis.jsx - Main UI**
```javascript
// Features:
- File upload (drag & drop)
- Audio recording (10s max)
- Backend health check before analysis
- Progress tracking with toast notifications
- Error handling (timeout, short audio, network issues)
- Results display with visualizations
- Export to PDF
- Save to history (localStorage)

// State Management:
- audioFile, recordedAudio, analysisState
- analysisResults, errorMessage, audioUrl
```

### **frontend/src/components/ResultsDisplay.jsx**
```javascript
// Visualizations:
- Radar chart (genre confidence)
- Bar chart (top genres)
- Confidence scores with color coding
- SNR meter
- Duration and segment info
- Social sharing buttons
```

### **frontend/src/components/WaveformTimeline.jsx**
```javascript
// Features:
- WaveSurfer.js integration
- Audio playback controls
- Timeline heatmap (genre predictions over time)
- Waveform visualization
- Time cursor tracking
```

---

## 🚀 How to Run (Quick Reference)

### **Option 1: Manual Start (2 Terminals)**

**Terminal 1 - Backend:**
```powershell
cd "V:/vishnu/notes pdf/5th SEM/mini project(3c)/project-1"
.\.venv\Scripts\Activate.ps1
python app.py
```
✅ Backend runs on http://127.0.0.1:5000

**Terminal 2 - Frontend:**
```powershell
cd "V:/vishnu/notes pdf/5th SEM/mini project(3c)/project-1/frontend"
npm run dev
```
✅ Frontend runs on http://localhost:3000

### **Option 2: Quick Start Script**
```powershell
.\START_BACKEND.ps1    # Starts backend only
```

---

## ✅ System Status Check

| Component | Status | Details |
|-----------|--------|---------|
| **Backend Core** | ✅ Ready | app.py, model files present |
| **Model Checkpoint** | ✅ Ready | 56 MB, 88.83% accuracy |
| **Training Scripts** | ✅ Ready | train_parallel_cnn.py, train_model_torch.py |
| **Frontend Core** | ✅ Ready | All React components present |
| **Dependencies** | ✅ Ready | requirements.txt, package.json |
| **Documentation** | ✅ Complete | 17 markdown files |
| **Configuration** | ✅ Ready | vite.config.js, API endpoints |
| **Git Repository** | ✅ Ready | pro1 branch, .gitignore configured |

---

## 🎯 Critical Dependencies

### **Backend Must-Have:**
- Python 3.11 with .venv
- PyTorch 2.6.0 (CUDA 12.4)
- librosa, Flask, OpenL3
- NVIDIA GPU (optional but recommended)

### **Frontend Must-Have:**
- Node.js 16+
- npm packages (react, vite, tailwindcss, chart.js, wavesurfer.js)

---

## 📝 Summary

**Total Files**: 61 tracked files  
**Backend Files**: 3 core Python scripts  
**Frontend Files**: 35+ React/JS files  
**Documentation**: 17 markdown files  
**Model Size**: 56 MB  

**Project Status**: ✅ **100% Ready to Run**

All critical files are intact. The project is fully functional and can be started immediately using the commands above.

---

**Last Verified**: December 29, 2025  
**Repository**: Music_Genre_Classification (pro1 branch)  
**Owner**: Vishnu-kashyap-D
