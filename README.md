# Collaborative Filtering Recommender System

A multi-domain deep learning application that provides personalized content recommendations for Books and Movies. This system uses **Neural Collaborative Filtering (NCF)** specifically the **Generalized Matrix Factorization (GMF)** architecture to learn high-dimensional latent embeddings, serving predictions via a **FastAPI** API service and a **Streamlit** dashboard.

---

## 🚀 Key Features

* **⚡ Deep Learning Engine:** Custom **PyTorch** architecture utilizing Matrix Factorization with learnable user/item embeddings.
* **🎬 Multi-Domain Support:** Seamlessly switches between a **Book Recommender** and a **Movie Recommender**.

---

## 🛠️ Tech Stack

* **Machine Learning:** PyTorch, Scikit-Learn (Nearest Neighbors), Pandas
* **Interface:** Streamlit
* **Environment:** Python 3.13+

---

## 📂 Project Structure

```text
├── docs/images             # Experiment images
├── models/                 # Saved Model Files (.pth) & Encoders (.joblib)
├── src/
│   ├── app.py              # Streamlit Frontend (The Client)
│   ├── movie_model.py      # PyTorch Model Architecture (128-dim)
│   ├── model.py            # PyTorch Model Architecture (50-dim)
│   ├── movie_dataset.py    # Torch Dataset & Dataloader Logic
│   ├── dataset.py          # Torch Dataset & Dataloader Logic
│   ├── movie_train.py      # Torch Dataset & Dataloader Logic
│   └── train.py            # Training Loop & Checkpointing
├── utils/
│   └── recommenders/       # Inference Engines (Decoupled Logic)
│       ├── book_recommender.py
│       └── movie_recommender.py
├── EXPERIMENTS.md          # Experiment logs
└── pyproject.toml          # Dependencies

## ⚡ Installation & Setup

### 1. Prerequisites
* Python 3.13+ installed.
* Poetry installed for dependency management.

### 2. Clone and Install
```bash
git clone [https://github.com/yourusername/NextPick.git](https://github.com/yourusername/NextPick.git)
cd NextPick
poetry install
```
### 4. Setup Models
Place the following files in the models/ directory:
* movie_matrix_factorization_checkpoint.pth
* matrix_factorization_checkpoint.pth
* movie_encoder.joblib & movie_user_encoder.joblib
* book_encoder.joblib & book_user_encoder.joblib

## 🏃‍♂️ How to Run
This application follows a microservice pattern and requires two terminal windows running simultaneously.

### Terminal 1: Start the Backend API
This launches the FastAPI server (The Brain).
```bash
uvicorn services.app:app --reload
```

### Terminal 2: Start the Frontend UI
You can run the full interactive dashboard using Streamlit:
```bash
streamlit run streamlit_app/Home.py
```

### 🧠 Model Performance
The Book Recommender was trained on the Book-Crossing Kaggle dataset using a Scaled Matrix Factorization architecture
* Final Training Loss (MSE): 1.5275
* Final Validation Loss (MSE): 3.1901
* Embedding Dimension: 50

The Movie Recommender was trained on the Movie Recommendation System Kaggle dataset using a Scaled Matrix Factorization architecture
* Final Training Loss (MSE): 0.7058
* Final Validation Loss (MSE): 0.7072
* Embedding Dimension: 128

### 🔮 Future Improvements
* Expand Domain: Add more categories/domains.

* Hybrid Filtering: Combine Collaborative Filtering with Content-Based filtering (Genre/Author metadata) to solve the "Cold Start" problem.

* Dockerize the application for cloud deployment.

