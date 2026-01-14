import streamlit as st

st.set_page_config(
    page_title="NextPick", page_icon="😎", layout="wide"
)  # Added 'wide' for better spacing

st.title("NextPick 🎯")
st.caption("Your AI-powered recommendation engine.")
st.info(
    "🛠️ **Internal Demo Interface:** This app is designed for testing the model architecture and API logic."
)

st.divider()

st.markdown("### The Intelligence Engine")

st.success(
    """
    **No hard-coded rules.** NextPick uses **Neural Collaborative Filtering** to discover hidden patterns in user behavior.
    It doesn't just know that you like *Sci-Fi*; it knows you like *90s Cyberpunk Sci-Fi with a philosophical twist*.
    """
)

st.divider()

st.subheader("Technical Specs")

tab1, tab2 = st.tabs(["📚 Book Model", "🎬 Movie Model"])

with tab1:
    st.markdown("#### **Neural Collaborative Filtering (NCF)**")
    st.markdown(
        """
    | Feature | Specification |
    | :--- | :--- |
    | **Architecture** | General Matrix Factorization (`torch.nn.Module`) |
    | **Dataset** | [Book-Crossing Dataset (Kaggle)](https://www.kaggle.com/datasets/ruchi798/bookcrossing-dataset) |
    | **Embedding Strategy** | Learned **User & Item Vectors** initialized via **Xavier Uniform**. Includes **Bias Terms** to account for user leniency. |
    | **Inference Logic** |  **Nearest Neighbor (k-NN)** search on the latent space to find books mathematically close to the query using the cosine similarity metric. |
    """
    )

with tab2:
    st.markdown("#### **Scaled Matrix Factorization**")
    st.markdown(
        """
    | Feature | Specification |
    | :--- | :--- |
    | **Architecture** | General Matrix Factorization (`torch.nn.Module`) |
    | **Dataset** | [Movie Recommendation System (Kaggle)](https://www.kaggle.com/datasets/parasharmanas/movie-recommendation-system) |
    | **Embedding Strategy** | **High-Fidelity (128-Dim)** latent space. Captures complex nuances (e.g., "90s Crime Drama") beyond simple genre tags. |
    | **Inference Logic** | **Nearest Neighbor (k-NN)** search on the latent space to find books mathematically close to the query using the cosine similarity metric. |
    """
    )

st.divider()
st.caption("👈 Select 'Books' or 'Movies' in the sidebar to test the inference.")
