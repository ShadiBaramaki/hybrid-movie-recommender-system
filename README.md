# 🎬 Hybrid Movie Recommender System

![Python](https://img.shields.io/badge/Python-3.11-blue)
![ML](https://img.shields.io/badge/Machine%20Learning-Recommender%20System-orange)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![Live Demo](https://img.shields.io/badge/Live%20Demo-Online-brightgreen)

A **Hybrid Movie Recommendation System** that combines **Content-Based Filtering** and **Collaborative Filtering** using Machine Learning techniques, deployed with a Streamlit web interface.

🔗 **Live Demo:** [hybrid-movie-recommender-system-project.streamlit.app](https://hybrid-movie-recommender-system-project.streamlit.app/)

---

## 📸 UI Preview

### 🔍 Movie Search & Recommendation Interface
![UI Screenshot](assets/ui-preview.png)

---

## 🧠 How It Works

This system is a **Hybrid Recommendation Engine** that uses two complementary approaches to improve recommendation quality.

---

### 🎯 1. Content-Based Filtering

This method recommends movies based on their **features (genres)**.

#### 📌 Process:
- Movies are represented using their genres
- Genres are converted into numerical vectors using `CountVectorizer`
- Similarity between movies is calculated using **Cosine Similarity**
- Movies with highest similarity scores are recommended

#### 📌 Intuition:
> If you like a movie, you will likely enjoy similar movies with the same characteristics.

---

### 👥 2. Collaborative Filtering

This method recommends movies based on **user behavior patterns**.

#### 📌 Process:
- Uses user rating history from the dataset
- Builds a **sparse user-item interaction matrix**
- Calculates similarity between movies based on user rating patterns
- Recommends movies that similar users also enjoyed

#### 📌 Intuition:
> Users with similar preferences in the past will likely agree in the future.

---

### ⚡ 3. Hybrid Model

To improve recommendation accuracy, both methods are combined.

#### 📌 Formula:
```
Final Score = (Content-Based Score + Collaborative Score) / 2
```

#### 📌 Why Hybrid?
- Content-Based → solves item similarity
- Collaborative → captures user behavior
- Hybrid → balances both approaches and improves overall accuracy

---

## 📂 Dataset

This project uses the **MovieLens Small Dataset**, a widely used benchmark in recommendation systems research.

### 📁 Files:
- `movies.csv` → movie metadata (title, genres)
- `ratings.csv` → 100,000+ user ratings

### 📌 Source:
[https://grouplens.org/datasets/movielens/](https://grouplens.org/datasets/movielens/)

---

## 🧰 Tech Stack

| Tool | Purpose |
|------|---------|
| 🐍 Python | Core language |
| 📊 Pandas | Data manipulation |
| 🔢 NumPy | Numerical computation |
| 🤖 Scikit-learn | ML & similarity metrics |
| 🔬 SciPy | Sparse matrix optimization |
| 🎛️ Streamlit | Web UI & deployment |

---

## 📊 Machine Learning Concepts

This project implements:
- Feature Engineering (Genre Encoding)
- Cosine Similarity
- Sparse User-Item Matrix
- Collaborative Filtering
- Content-Based Filtering
- Hybrid Recommendation Systems

---

## 🚀 Run Locally

```bash
git clone https://github.com/ShadiBaramaki/hybrid-movie-recommender-system.git
cd hybrid-movie-recommender-system
pip install -r requirements.txt
streamlit run app.py
```
