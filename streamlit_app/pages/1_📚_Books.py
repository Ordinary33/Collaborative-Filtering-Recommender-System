import streamlit as st
import requests

st.title("Book Recommender")

isbn = st.text_input("Enter 10-digit ISBN of a book")


if st.button("Get Recommendations"):
    if isbn:
        try:
            response = requests.get(f"http://127.0.0.1:8000/recommend/{isbn}")
            response.raise_for_status()
            result = response.json()
            recommendations = result.get("data", [])

            st.subheader(f"Recommendations for {result.get('title', 'the book')}:")

            with st.container():
                st.write(result)

        except requests.exceptions.HTTPError:
            st.error(f"Book with the isbn {isbn} not found in the database")
            st.info("Please try another ISBN.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a valid ISBN.")
