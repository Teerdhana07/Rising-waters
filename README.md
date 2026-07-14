#  Flood Prediction System using Machine Learning

##  Project Overview

The **Flood Prediction System** is a Machine Learning-based web application that predicts the likelihood of floods using historical weather data. The system applies the **XGBoost** algorithm to analyze weather parameters and generate accurate flood predictions through a simple and interactive Flask web interface.

---

##  Features

- Historical weather data analysis
- Data preprocessing and feature scaling
- Machine Learning model training
- Flood prediction using the trained XGBoost model
- Flask-based web application
- User-friendly prediction interface
- Performance testing using Apache JMeter

---

##  Technologies Used

- Python
- Flask
- Scikit-learn
- XGBoost
- Pandas
- NumPy
- Matplotlib
- HTML
- CSS
- Apache JMeter
- Jupyter Notebook

---

# Repository Structure

```text
Rising-Waters/
│
├── 1.Brainstorming & Ideation/
│   ├── Brainstorming & Idea Prioritization.pdf
│   ├── Define Problem Statements.pdf
│   └── Empathy Map.pdf
│
├── 2.Requirement Analysis/
│   ├── Customer Journey Map.pdf
│   ├── Data Flow Diagram.pdf
│   ├── Solution Requirements.pdf
│   └── Technology Stack.pdf
│
├── 3.Project Design Phase/
│   ├── Problem-Solution Fit.pdf
│   ├── Proposed Solution.pdf
│   └── Solution Architecture.pdf
│
├── 4.Project Planning Phase/
│   └── Project Planning.pdf
│
├── 5.Project Development Phase/
│   ├── Code Layout, Readability and Reusability.pdf
│   ├── Coding & Solution.pdf
│   └── No. of Functional Features Included.pdf
│
├── 6.Project Testing/
│   └── Performance Testing.pdf
│
├── 7.Project Documentation/
│   ├── Project Executable Files.pdf
│   └── Sample Project Documentation.pdf
│
├── 8.Project Demonstration/
│   ├── Communication.pdf
│   ├── Demonstration of Proposed Features.pdf
│   ├── Project Demo Planning.pdf
│   ├── Scalability & Future Plan.pdf
│   └── Team Involvement in Demonstration.pdf
│
├── app.py
├── README.md
├── requirements.txt
├── dataset/
├── documentation/
├── models/
├── notebooks/
├── static/
└── templates/
```

---

# Folder Description

### 1. Brainstorming & Ideation
- Brainstorming & Idea Prioritization
- Problem Definition
- Empathy Map

### 2. Requirement Analysis
- Customer Journey Map
- Data Flow Diagram (DFD)
- Functional & Non-Functional Requirements
- Technology Stack

### 3. Project Design Phase
- Problem-Solution Fit
- Proposed Solution
- Solution Architecture

### 4. Project Planning Phase
- Project Planning and Scheduling

### 5. Project Development Phase
- Source Code Implementation
- Code Quality & Readability
- Functional Features
- Machine Learning Model Development

### 6. Project Testing
- Performance Testing using Apache JMeter

### 7. Project Documentation
- Project Executable Files
- Project Documentation

### 8. Project Demonstration
- Communication Strategy
- Demonstration of Proposed Features
- Project Demo Planning
- Scalability & Future Plan
- Team Involvement

---

#  How to Run

## Prerequisites

- Python 3.10 or above
- pip

---

## Navigate to the Project Folder

```bash
cd Rising-Waters
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Application

```bash
python app.py
```

---

## Open in Browser

```
http://127.0.0.1:5000
```

---

# Machine Learning Models Used

- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- **XGBoost (Final Selected Model)**

---

# Testing

- Functional Testing
- Performance Testing using Apache JMeter
- Prediction Validation
- Flood Risk & No Flood Risk Verification

---

# Performance Testing

The application was tested using **Apache JMeter** to evaluate response time, throughput, and error rate.

### Test Summary

- Testing Tool: Apache JMeter
- Average Response Time: **17 ms**
- Maximum Response Time: **59 ms**
- Throughput: **11.2 Requests/sec**
- Error Rate: **0.00%**

---

# Future Enhancements

- Cloud Deployment
- Live Weather API Integration
- User Authentication
- Mobile Application
- Advanced Analytics Dashboard
-
# Demo link
- https://drive.google.com/file/d/19lvLWJkaWydopGWNJsB8Y-s9QYOrdfFQ/view
