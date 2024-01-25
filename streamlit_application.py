import streamlit as st

# Import your modularized applications
from python_sentiment_inference import run_sentiment_analysis
from recommender_streamlit import run_company_recommendation

# Function to render the sidebar and handle navigation
def main():
    st.sidebar.title("Navigation")
    app_choice = st.sidebar.radio("Go to", ["Home", "Sentiment Analysis", "Company Recommendation"])

    if app_choice == "Home":
        st.title("Welcome to the Multi-App Interface")
        st.write("Please select an application from the sidebar.")
    elif app_choice == "Sentiment Analysis":
        run_sentiment_analysis()
    elif app_choice == "Company Recommendation":
        run_company_recommendation()

if __name__ == "__main__":
    main()
