<!-- <p align="center">
  <img src="documentation/system_architecture.png" alt="Rising Waters Banner" width="800"/>
</p>

<h1 align="center">Rising Waters: A Machine Learning Approach to Flood Prediction</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Flask-2.3.2-000000?style=for-the-badge&logo=flask&logoColor=white" alt="Flask"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-1.3.0-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-learn"/>
  <img src="https://img.shields.io/badge/XGBoost-1.7.6-006600?style=for-the-badge&logo=xgboost&logoColor=white" alt="XGBoost"/>
  <img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge" alt="License"/>
</p> -->

<p align="center">
  <b>An intelligent flood prediction system powered by ensemble machine learning models, delivering real-time risk assessments through an intuitive web interface.</b>
</p>

---

## Project Overview

**Rising Waters** is a comprehensive machine learning system designed to predict flood occurrences based on meteorological and environmental parameters. The system leverages multiple ML algorithms — including Decision Trees, Random Forests, K-Nearest Neighbors, and XGBoost — to analyze weather data and generate accurate flood risk predictions.

Built with a Flask-powered web interface, the system enables users to input real-time weather conditions and receive instant flood risk assessments, making it a practical tool for disaster preparedness and early warning systems.

---

##  Features

| Feature | Description |
|---------|-------------|
|  **Multi-Model Prediction** | Employs 4 ML models with automatic best-model selection |
|  **Data Visualization** | Comprehensive EDA with correlation heatmaps, distribution plots, and feature analysis |
|  **Web Interface** | Clean, responsive Flask-based UI for real-time predictions |
|  **Model Comparison** | Side-by-side accuracy metrics across all trained models |
|  **XGBoost Champion** | Best-in-class accuracy of **96.55%** with XGBoost |
|  **Preprocessing Pipeline** | Automated feature scaling, encoding, and train-test splitting |
|  **Model Persistence** | Trained models saved via `joblib` for instant loading |

---

##  System Architecture

The system follows a layered architecture pattern ensuring clean separation of concerns:

![System Architecture](documentation/system_architecture.png)

**Architecture Layers:**

| Layer | Components | Responsibility |
|-------|-----------|----------------|
| 🧑‍💻 **User Layer** | End Users | Interacts with the web interface |
| 🎨 **Presentation Layer** | HTML / CSS / JavaScript | Renders UI and captures user input |
| ⚙️ **Application Layer** | Flask Framework | Routes, request handling, business logic |
| 🧠 **Machine Learning Layer** | Trained Models + Scaler | Prediction engine and data transformation |
| 💾 **Data Layer** | CSV Dataset | Source data for training and evaluation |

---

##  Entity Relationship Diagram

The ER diagram below illustrates the data model and relationships between system entities:

![Entity Relationship Diagram](documentation/ER_Diagram.png)

**Entities:**
- **User** — Stores user information (`user_id` PK, `name`, `email`)
- **Weather_Input** — Captures weather parameters (`input_id` PK, `user_id` FK, `rainfall`, `cloud_visibility`, `seasonal_rainfall`, `timestamp`)
- **Prediction** — Stores prediction results (`prediction_id` PK, `input_id` FK, `result`, `probability_score`, `model_used`)

**Relationships:**
- `User` ➜ `Weather_Input` — One-to-Many
- `Weather_Input` ➜ `Prediction` — One-to-One

---

##  Project Flow

The ML pipeline follows a structured, end-to-end workflow:

![Project Flow](documentation/project_flow.png)

```
📁 Dataset Collection
    ↓
📊 Data Analysis & Visualization
    ↓
🔧 Data Preprocessing (Scaling, Encoding, Splitting)
    ↓
🤖 Model Training (Decision Tree → Random Forest → KNN → XGBoost)
    ↓
📈 Model Evaluation (Accuracy, Confusion Matrix, Classification Report)
    ↓
🏆 Best Model Selection (XGBoost @ 96.55%)
    ↓
🌐 Flask Deployment
    ↓
🧑‍💻 User Prediction
```

---

##  Tech Stack

| Category | Technology |
|----------|-----------|
| **Language** | ![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white) Python 3.10+ |
| **Web Framework** | ![Flask](https://img.shields.io/badge/Flask-000000?style=flat-square&logo=flask&logoColor=white) Flask 2.3.2 |
| **ML Libraries** | ![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white) Scikit-learn 1.3.0 |
| **Boosting** | ![XGBoost](https://img.shields.io/badge/XGBoost-006600?style=flat-square) XGBoost 1.7.6 |
| **Data Processing** | ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white) Pandas 2.0.3 / NumPy 1.24.3 |
| **Visualization** | ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square) Matplotlib 3.7.2 / Seaborn 0.12.2 |
| **Frontend** | HTML5 / CSS3 / JavaScript |

---

## Folder Structure

```
Flood-Prediction-System/
│
├── 📁 dataset/
│   └── flood_data.csv                  # Training dataset
│
├── 📁 models/
│   ├── decision_tree_model.pkl         # Trained Decision Tree
│   ├── random_forest_model.pkl         # Trained Random Forest
│   ├── knn_model.pkl                   # Trained KNN
│   ├── xgboost_model.pkl              # Trained XGBoost (Best)
│   └── scaler.pkl                      # Fitted StandardScaler
│
├── 📁 notebooks/
│   ├── data_visualization.py           # EDA & visualization scripts
│   └── model_training.py              # Model training & evaluation
│
├── 📁 flask_app/
│   ├── app.py                          # Flask application entry point
│   ├── 📁 templates/
│   │   └── index.html                  # Web interface template
│   └── 📁 static/
│       └── style.css                   # Application styles
│
├── 📁 documentation/
│   ├── system_architecture.png         # Architecture diagram
│   ├── ER_Diagram.png                  # Entity relationship diagram
│   └── project_flow.png               # ML pipeline flow diagram
│
├── requirements.txt                    # Python dependencies
├── README.md                           # Project documentation
└── LICENSE                             # MIT License
```

---

## Installation & Setup

### Prerequisites
- Python 3.10 or higher
- pip (Python package manager)
- Git

### Step-by-Step Installation

**1. Clone the Repository**
```bash
git clone https://github.com/yourusername/Flood-Prediction-System.git
cd Flood-Prediction-System
```

**2. Create a Virtual Environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

**4. Train the Models**
```bash
python notebooks/model_training.py
```

**5. Launch the Flask Application**
```bash
python flask_app/app.py
```

**6. Open in Browser**
```
http://127.0.0.1:5000
```

---

##  ML Models Used

| # | Model | Algorithm Type | Description |
|---|-------|---------------|-------------|
| 1 | **Decision Tree** | Tree-based | Hierarchical splitting based on feature thresholds; interpretable and fast |
| 2 | **Random Forest** | Ensemble (Bagging) | Aggregates multiple decision trees to reduce overfitting and improve generalization |
| 3 | **K-Nearest Neighbors (KNN)** | Instance-based | Classifies based on majority vote of k nearest data points in feature space |
| 4 | **XGBoost** | Ensemble (Boosting) | Gradient-boosted trees with regularization; delivers state-of-the-art performance |

---

##  Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | Status |
|-------|----------|-----------|--------|----------|--------|
| Decision Tree | ~90.00% | ~0.89 | ~0.90 | ~0.89 | ✅ Good |
| Random Forest | ~94.00% | ~0.93 | ~0.94 | ~0.93 | ✅ Very Good |
| KNN | ~88.00% | ~0.87 | ~0.88 | ~0.87 | ✅ Good |
| **XGBoost** | **~96.55%** | **~0.96** | **~0.97** | **~0.96** | 🏆 **Best** |

> ** Champion Model:** XGBoost achieves the highest accuracy at **96.55%** and is selected as the default prediction model in the deployed application.

---

##  Usage

### Training the Models
```bash
# Run the complete training pipeline
python notebooks/model_training.py
```
This script will:
- Load and preprocess the flood dataset
- Train all 4 ML models
- Evaluate and compare model performance
- Save the best model and scaler to the `models/` directory

### Running Data Visualization
```bash
# Generate EDA plots and visualizations
python notebooks/data_visualization.py
```

### Launching the Web App
```bash
# Start the Flask development server
python flask_app/app.py
```
Navigate to `http://127.0.0.1:5000` and enter weather parameters to get predictions.

---

##  Usage Scenarios

### Scenario 1: Heavy Rainfall Alert
> **Input:** High rainfall (200mm), low cloud visibility (2km), high seasonal rainfall (500mm)
>
> **Output:**  **Flood Predicted** — Probability: 94.7%
>
> **Action:** Issue early warning, activate emergency protocols

### Scenario 2:  Clear Weather Check
> **Input:** Low rainfall (10mm), high cloud visibility (15km), normal seasonal rainfall (100mm)
>
> **Output:**  **No Flood** — Probability: 3.2%
>
> **Action:** Normal operations, no intervention needed

### Scenario 3:  Moderate Conditions Assessment
> **Input:** Moderate rainfall (80mm), moderate cloud visibility (7km), above-average seasonal rainfall (300mm)
>
> **Output:** **Flood Possible** — Probability: 61.3%
>
> **Action:** Monitor conditions closely, prepare contingency plans

---

## System Requirements

### Hardware Requirements
| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **Processor** | Intel i3 / AMD Ryzen 3 | Intel i5 / AMD Ryzen 5 or higher |
| **RAM** | 4 GB | 8 GB or higher |
| **Storage** | 2 GB free space | 5 GB free space |
| **Display** | 1280 × 720 | 1920 × 1080 |

### Software Requirements
| Software | Version |
|----------|---------|
| **Operating System** | Windows 10+ / macOS 12+ / Ubuntu 20.04+ |
| **Python** | 3.10 or higher |
| **pip** | 21.0 or higher |
| **Web Browser** | Chrome 90+ / Firefox 88+ / Edge 90+ |
| **Git** | 2.30+ (optional) |

---



<p align="center">
  Made with ❤️ for disaster preparedness and community safety
</p>

<p align="center">
  <a href="#-rising-waters-a-machine-learning-approach-to-flood-prediction">⬆️ Back to Top</a>
</p>
