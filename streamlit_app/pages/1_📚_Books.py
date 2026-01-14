import streamlit as st
import requests

st.title("📚 Book Recommender")

isbn = st.text_input("Enter 10-digit ISBN of a book")


if st.button("Get Recommendations"):
    if isbn:
        try:

            with st.spinner("Fetching recommendations..."):
                response = requests.get(f"http://127.0.0.1:8000/recommend/book/{isbn}")
                response.raise_for_status()
                result = response.json()

            recommendations = result.get("data", [])
            title = result.get("title", "The Book")

            if not recommendations:
                st.warning(f"No recommendations found for ISBN {isbn}.")

            else:
                st.write(f"**Recommendations based on '{title}':**")
                with st.container():
                    for data in recommendations:
                        st.markdown("---")
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.image(data["Image"], width=100)
                        with col2:
                            st.markdown(f"**Title:** {data['Title']}")
                            st.markdown(f"**Author:** {data['Author']}")
                            st.markdown(f"**Year:** {data['Year']}")
                            st.markdown(f"**ISBN:** {data['ISBN']}")
                    st.markdown("---")
                    st.markdown("Json Response:")
                    st.json(result)

        except requests.exceptions.HTTPError:
            st.error(f"Book with the isbn {isbn} not found in the database")
            st.info("Please try another ISBN.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a valid ISBN.")
