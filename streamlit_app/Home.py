import streamlit as st

st.set_page_config(page_title="NextPick", page_icon="😎")

st.title("NextPick 🎯")
st.caption("Your AI-powered recommendation engine.")
st.info(
    "🛠️ **Internal Demo Interface:** This app is designed for testing the model architecture and API logic."
)

st.divider()

st.markdown(
    """
### How it works
NextPick goes beyond simple keyword matching. We use **Deep Learning** to understand the *context* and *vibe* of the content you love.

Select a category in the sidebar to get started!
"""
)

with st.expander("🤓 Technical details (The Models)"):
    st.markdown(
        """
    **Book Recommender Model:**
    - **Architecture:** General Matrix Factorization (Neural Collaborative Filtering).
    - **Framework:** PyTorch (`nn.Module`).
    - **Mechanism:** - Learns unique **User & Item Embeddings** (vectors) during training.
        - Uses **Xavier Initialization** for weight convergence.
        - Includes **Bias Terms** to account for user leniency and item popularity.
    - **How it Recommends:** We map these vectors into a high-dimensional space and use a **Nearest Neighbor** search to find books that are mathematically closest to your favorites.
    - **Prediction:** Calculates the **Dot Product** of user/item vectors + bias, passed through a sigmoid activation to predict your rating.
    """
    )
