import streamlit as st
import requests

st.title("Book Recommender")

isbn = st.text_input("Enter ISBN of a book")


if st.button("Get Recommendations"):
    if isbn:
        try:
            response = requests.get(f"http://127.0.0.1:8000/recommend/{isbn}")
            response.raise_for_status()
            result = response.json()

            with st.container():
                st.subheader("Recommendations:")
                st.write(result)

        except requests.exceptions.HTTPError as err:
            st.error(f"Book not found or API error : {err}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a valid ISBN.")
