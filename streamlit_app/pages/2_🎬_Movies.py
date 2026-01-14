import streamlit as st
import requests

st.title("🎬 Movie Recommender")

movie_id = st.text_input("Enter Movie ID (e.g., 1, 2, 3...)")

if st.button("Get Recommendations"):
    if movie_id:
        try:

            with st.spinner("Fetching Recommendations"):
                response = requests.get(
                    f"http://127.0.0.1:8000/recommend/movie/{movie_id}"
                )
                response.raise_for_status()
                result = response.json()

            recommendations = result.get("data", [])
            title = result.get("title", "Movie")

            if not recommendations:
                st.warning(f"No recommendations found for Movie ID {movie_id}.")

            else:
                st.write(f"**Recommendations based on '{title}':**")

                with st.container():
                    st.json(result)
        except requests.exceptions.HTTPError:
            st.error(f"Movie with the ID {movie_id} not found in the database")
            st.info("Please try another Movie ID.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a valid Movie ID.")
