import tensorflow as tf
# !pip install transformers
from transformers import DistilBertTokenizerFast
from transformers import TFDistilBertForSequenceClassification

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
# loaded_model = TFDistilBertForSequenceClassification.from_pretrained("/sentiment")
loaded_model = TFDistilBertForSequenceClassification.from_pretrained("C:/Users/hugod/OneDrive/Documents/Annee5_DIA/nlp_project_hasna_hugo_rendu/sentiment_2")
# "C:\Users\hugod\OneDrive\Documents\Annee5_DIA\nlp_project_hasna_hugo\sentiment"


# Import necessary librariesx
import streamlit as st




# # test_sentence = "i waste my money"
# test_sentence = "this is not good at all"

# predict_input = tokenizer.encode(test_sentence,
#                                  truncation=True,
#                                  padding=True,
#                                  return_tensors="tf")

# # Get the model's prediction
# tf_output = loaded_model.predict(predict_input)[0]

# # Since it's a binary classification, the output is a single probability value
# # for the positive class. We use 0.5 as a threshold to decide the class.
# predicted_prob = tf.sigmoid(tf_output).numpy()[0][0]
# predicted_class = 'positif' if predicted_prob >= 0.5 else 'negatif'

# print(f"Prediction: {predicted_class} (Probability: {predicted_prob:.2f})")




def run_sentiment_analysis():
        

    def predict_sentiment(user_input):
        predict_input = tokenizer.encode(user_input, truncation=True, padding=True, return_tensors="tf")
        tf_output = loaded_model.predict(predict_input)[0]

        predicted_prob = tf.sigmoid(tf_output).numpy()[0][0]
        predicted_class = 'positif' if predicted_prob >= 0.5 else 'negatif'

        # tf_prediction = tf.nn.softmax(tf_output, axis=1)
        # labels = ['0', '1']
        # label = tf.argmax(tf_prediction, axis=1)
        # prediction = labels[label[0]]


        return predicted_class

    # Streamlit app
    st.title("Sentiment Analysis Reviews")

    # Input for user to enter a review
    user_input = st.text_area("Enter your review:")

    # Button to trigger sentiment analysis
    if st.button("Predict Sentiment"):
        if user_input:
            # Perform sentiment analysis
            result = predict_sentiment(user_input)
            st.success(f"Sentiment: {result}")
        else:
            st.warning("Please enter a review.")