


import streamlit as st
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def run_company_recommendation(): 

    # Streamlit app title
    st.title("Company Recommendation System")

    # Load the dataset
    file_path = 'recommandation4.csv'
    df = pd.read_csv(file_path)
    df['Review'] = df['Review'].apply(lambda x: '' if not isinstance(x, str) else x)
    aggregated_reviews = df.groupby('Name')['Review'].apply(lambda x: ' '.join(x)).reset_index()

    # Load BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # Function to encode text using BERT
    def encode_text(text):
        input_ids = tokenizer.encode(text, add_special_tokens=True, max_length=512, truncation=True)
        input_ids = torch.tensor([input_ids])
        with torch.no_grad():
            outputs = model(input_ids)
            last_hidden_states = outputs.last_hidden_state
            sentence_embedding = torch.mean(last_hidden_states, dim=1)
        return sentence_embedding

    # User input
    user_query = st.text_input("Enter your query:", "looking for a nice dress")
    user_query_embedding = encode_text(user_query)

    # Process and display recommendations
    if st.button("Recommend"):
        company_embeddings = aggregated_reviews['Review'].apply(lambda x: encode_text(x).numpy())
        cosine_similarities = [cosine_similarity(user_query_embedding, company_embedding)[0][0] for company_embedding in company_embeddings]
        highest_score_index = cosine_similarities.index(max(cosine_similarities))
        recommended_company_bert = aggregated_reviews.iloc[highest_score_index]['Name']
        st.write(f"Recommended Company: {recommended_company_bert}")

    # Run this script with `streamlit run your_script.py`
