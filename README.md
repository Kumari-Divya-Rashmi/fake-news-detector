# 📰 Fake News Detection API

A Flask-based backend API that predicts whether a given news headline is **real** or **fake** using machine learning. Deployed on Render.

## 🔗 Live Demo

👉 ** Visit API **: [https://fake-news-detector-3nx7.onrender.com]

## 🚀 Features

- Built with **Flask**
- Trained on a labeled dataset of real vs fake news
- Uses **TF-IDF Vectorizer** + **Logistic Regression**
- JSON-based API for easy frontend integration
- Deployed on **Render** with `gunicorn`

## 📦 Installation (Local Setup)
git clone https://github.com/Kumari-Divya-Rashmi/fake-news-detector.git
cd fake-news-detector
pip install -r requirements.txt
python app.py

🧠 Model Info

Vectorizer: TfidfVectorizer
Classifier: LogisticRegression
Dataset: Preprocessed Kaggle dataset (Real vs Fake news)

📁 Project Structure

fake-news-detector
├── app.py                # Flask app with /predict route
├── model.pkl             # Trained ML model
├── vectorizer.pkl        # TF-IDF vectorizer
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation



🌐 Deployment

Deployed on Render using:
Build Command: pip install -r requirements.txt
Start Command: gunicorn app:app

👤 Author
Your Name
📧 kumaridivyarashmi306@gmail.com.com
🔗 https://www.linkedin.com/in/kumari-divya-rashmi-673b24291/
🔗 https://github.com/Kumari-Divya-Rashmi

