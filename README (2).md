# 🎬 Movie Recommendation System

A **content-based Movie Recommendation System** in **Python** that suggests movies similar to a selected movie using features like genres, keywords, cast, and crew.

---

## 🚀 Features

* Interactive **Streamlit** web interface  
* Recommends movies similar to the chosen one  
* Uses **cosine similarity** for content-based recommendations  
* Preprocessed dataset for fast responses

---

## 🧩 Tech Stack

| Component | Technology |
|----------|------------|
| Language | Python 🐍 |
| Framework | Streamlit |
| Libraries | Pandas, NumPy, Scikit-learn |
| Algorithm | Cosine Similarity |
| IDE | VS Code |
| Version Control | Git + GitHub |

---

## 📂 Folder Structure

```
movie-recommendation-system/
│
├── app.py
├── preprocess.py
├── dataset.csv
├── movie_dict.pkl
├── similarity.pkl
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Run

```bash
git clone https://github.com/<your-username>/movie-recommendation-system.git
cd movie-recommendation-system
pip install -r requirements.txt
python preprocess.py       # if not done
streamlit run app.py
```

Open in browser: `http://localhost:8501`

---

## 🎯 How It Works

1. Combine movie metadata (genres, keywords, cast, crew)  
2. Create a **content matrix** using CountVectorizer  
3. Compute **cosine similarity**  
4. Recommend top 5 movies based on selection

---

## 👨‍💻 Developed By

**Uday Kumar Dubey**  
B.Tech CSE (CSBS), Chandigarh University

