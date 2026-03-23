# 🚗 AI Vehicle Health Monitoring System

## 📌 Project Overview
This project is an AI-based Vehicle Health Monitoring System that predicts:
- 🔧 Maintenance Requirement
- ⚠️ Machine Failure Risk

The system uses machine learning models and an interactive web dashboard to analyze vehicle data and sensor inputs in real-time.

---

## 🚀 Live Demo
👉 [Click here to use the app]([YOUR_STREAMLIT_LINK_HERE](https://vehicle-health-monitor-ypedb3vfpjmpwvwrjmv29c.streamlit.app/))

---

## 🧠 Features
- Predicts vehicle maintenance requirement using ML model
- Predicts machine failure risk based on sensor data
- Displays probability-based results (not just 0/1)
- Interactive dashboard built with Streamlit
- Visualizations:
  - Feature Importance Graph
  - Sensor Data Graph
  - Risk Progress Bars

---

## 🏗️ Tech Stack
- **Programming Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib
- **Framework:** Streamlit
- **Deployment:** Streamlit Cloud
- **Version Control:** Git & GitHub

---

## 📊 Machine Learning Models

### 1️⃣ Maintenance Prediction Model
- Algorithm: Decision Tree Classifier
- Input: Vehicle details (age, mileage, battery, brake, etc.)
- Output: Maintenance risk probability

### 2️⃣ Failure Prediction Model
- Algorithm: Random Forest Classifier
- Input: Sensor data (temperature, RPM, torque, tool wear)
- Output: Failure risk probability

---

## ⚙️ How It Works
1. User enters vehicle and sensor data
2. Data is preprocessed and encoded
3. ML models generate predictions
4. Results are displayed with probabilities and graphs

---

## 📂 Project Structure

Vehicle_Health_Monitor/
│
├── frontend/
│ └── app.py
├── model/
│ ├── maintenance_model.pkl
│ └── failure_model.pkl
├── datasets/
├── notebooks/
└── requirements.txt


---

## ⚠️ Challenges Faced
- Handling imbalanced dataset
- Model bias towards majority class
- Deployment errors (joblib, file paths)
- Feature encoding mismatch
- Large model file size

---

## 💡 Key Learnings
- End-to-end ML project development
- Model training and evaluation
- Streamlit dashboard creation
- Deployment on cloud platform
- Debugging real-world issues

---

## 📈 Future Improvements
- Add user authentication system
- Store prediction history in database
- Integrate real-time IoT sensor data
- Convert into mobile application

---

## 👨‍💻 Author
**Samrat Sachin Maurya**

---

## ⭐ If you like this project
Give it a star ⭐ on GitHub!
