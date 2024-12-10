# Force CPU usage - must be at the very top of the file
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    DistilBertTokenizer, DistilBertForSequenceClassification,
    RobertaTokenizer, RobertaForSequenceClassification
)
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Set page config
st.set_page_config(
    page_title="Turkish News Classifier",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    nltk.download('stopwords')
    nltk.download('punkt')
    return set(stopwords.words('english'))

stop_words = download_nltk_data()
porter_stemmer = PorterStemmer()

# Define class labels with emojis
CLASS_LABELS = {
    1: "🌍 Dünya Haberleri",
    2: "⚽ Spor",
    3: "💼 İş Dünyası",
    4: "🔬 Bilim ve Teknoloji"
}

# Preprocessing functions
def preprocess_text(text, is_description=False):
    # Remove HTML-like tags and special codes
    text = re.sub(r'&lt;[^&]+&gt;', '', text)
    text = re.sub(r'<[^>]*>', '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove source information
    if is_description:
        text = re.sub(r'^\w+\s*-\s*', '', text)  # Remove source at beginning
        text = re.sub(r'\s*-\s*\w+\s*$', '', text)  # Remove source at end
    
    # Remove numbers and special characters
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    
    # Apply stemming
    tokens = [porter_stemmer.stem(word) for word in tokens]
    
    # Join tokens back into text
    return ' '.join(tokens)

# Load vectorizers
@st.cache_resource
def load_vectorizers():
    vectorizers = {
        'tfidf_ml': joblib.load('models/ml/tf_idf_vectorizer_fitted.joblib'),
        'tfidf_dl': joblib.load('models/dl/tfidf_vectorizer.joblib'),
        'w2v': np.load('models/dl/embedding_matrix_w2v.npy'),
        'fasttext': np.load('models/dl/embedding_matrix_ft.npy')
    }
    return vectorizers

# Load ML models
@st.cache_resource
def load_ml_models():
    models = {
        'Multinomial Naive Bayes': joblib.load('models/ml/mnb_model_tf_idf.joblib'),
        'Random Forest': joblib.load('models/ml/rf_model_tf_idf.joblib'),
        'Support Vector Classifier': joblib.load('models/ml/svm_model_tf_idf.joblib'),
        'K-Nearest Neighbours': joblib.load('models/ml/knn_model_tf_idf.joblib'),
        'Logistic Regression': joblib.load('models/ml/logistic_model_tf_idf.joblib'),
        'XGBoost': joblib.load('models/ml/xgb_model_tf_idf.joblib')
    }
    return models

# Load DL models
@st.cache_resource
def load_dl_models():
    try:
        models = {
            'LSTM-TFIDF': tf.keras.models.load_model('models/dl/lstm_tfidf.h5'),
            'RNN-TFIDF': tf.keras.models.load_model('models/dl/rnn_tfidf.h5')
        }
        return models
    except Exception as e:
        st.error(f"Error loading DL models: {str(e)}")
        st.error("Make sure all .h5 files are present in the models/dl directory")
        raise e

# Load transformer models
@st.cache_resource
def load_transformer_models():
    try:
        models = {
            'BERT': (
                BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4),
                BertTokenizer.from_pretrained('bert-base-uncased')
            ),
            'DistilBERT': (
                DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=4),
                DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            ),
            'RoBERTa': (
                RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=4),
                RobertaTokenizer.from_pretrained('roberta-base')
            )
        }
        return models
    except Exception as e:
        st.error(f"Error loading transformer models: {str(e)}")
        raise e

def predict_text(title, description, model_name, model_category):
    # Combine preprocessed title and description
    combined_text = f"{title} {description}"
    
    if model_category == "Machine Learning":
        vectorizers = load_vectorizers()
        models = load_ml_models()
        
        # ML models use their own TF-IDF
        text_vectorized = vectorizers['tfidf_ml'].transform([combined_text])
        
        # Get prediction
        model = models[model_name]
        prediction = model.predict(text_vectorized)[0]
        
        # Handle confidence score based on model type
        if model_name == "Support Vector Classifier":
            # For SVC, use decision_function if predict_proba is not available
            try:
                confidence = np.max(model.predict_proba(text_vectorized))
            except AttributeError:
                # Normalize decision function scores to [0,1] range
                decision_scores = model.decision_function(text_vectorized)
                confidence = float(1 / (1 + np.exp(-np.max(np.abs(decision_scores)))))
        elif model_name == "Random Forest":
            # Get probability scores from all trees and average them
            proba = model.predict_proba(text_vectorized)[0]
            # Take the average probability for the predicted class
            predicted_class_proba = proba[prediction]
            confidence = float(predicted_class_proba)
        else:
            confidence = float(np.max(model.predict_proba(text_vectorized)))
        
    elif model_category == "Deep Learning":
        models = load_dl_models()
        vectorizers = load_vectorizers()
        model = models[model_name]
        
        # For TFIDF, we need a 2D input (batch_size, features)
        text_vectorized = vectorizers['tfidf_dl'].transform([combined_text]).toarray()
        
        # Make prediction using Keras model with CPU
        with tf.device('/CPU:0'):
            predictions = model(text_vectorized, training=False)
            if isinstance(predictions, tf.Tensor):
                predictions = predictions.numpy()
            prediction = np.argmax(predictions[0])
            confidence = float(np.max(predictions[0]))
        
    else:  # Transformers
        models = load_transformer_models()
        model, tokenizer = models[model_name]
        
        # Transformer models handle their own tokenization
        inputs = tokenizer(
            combined_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Get prediction
        with torch.no_grad():
            outputs = model(**inputs)
            prediction = outputs.logits.argmax().item()
            confidence = torch.softmax(outputs.logits, dim=1).max().item()
    
    # Ensure confidence is between 0 and 1 before converting to percentage
    confidence = min(max(confidence, 0), 1)
    
    return prediction, confidence

# Enhanced Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 0.8rem;
        border-radius: 0.5rem;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .model-container {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        margin-bottom: 1rem;
        border: 1px solid #dee2e6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .prediction {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        margin-top: 1rem;
        border: 1px solid #4CAF50;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .prediction h3 {
        color: #2E7D32;
        margin-bottom: 1rem;
    }
    .stTextArea>div>div>textarea {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.5rem;
    }
    .category-label {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1a1a1a;
    }
    .confidence-score {
        font-size: 1rem;
        color: #666;
    }
    </style>
""", unsafe_allow_html=True)

# Main content
st.markdown("""
    <div style='text-align: center; padding: 1rem; background-color: #f8f9fa; border-radius: 0.5rem; margin-bottom: 2rem;'>
        <h1 style='color: #1a1a1a;'>📰 Turkish News Classifier</h1>
        <p style='color: #666; font-size: 1.1rem;'>Enter the title and description of the news article to classify</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar with model selection
with st.sidebar:
    st.markdown("""
        <div style='text-align: center; padding: 1rem; background-color: #f8f9fa; border-radius: 0.5rem; margin-bottom: 1rem;'>
            <h2 style='color: #1a1a1a;'>Model Selection</h2>
        </div>
    """, unsafe_allow_html=True)
    
    model_category = st.radio(
        "Select Model Category",
        ["Machine Learning", "Deep Learning", "Transformers"],
        help="Choose the type of model you want to use for classification"
    )
    
    if model_category == "Machine Learning":
        selected_model = st.selectbox(
            "Choose ML Model",
            ["Multinomial Naive Bayes", "Random Forest", "Support Vector Classifier",
             "K-Nearest Neighbours", "Logistic Regression", "XGBoost"]
        )
    elif model_category == "Deep Learning":
        selected_model = st.selectbox(
            "Choose DL Model",
            ["LSTM-TFIDF", "RNN-TFIDF"]
        )
    else:
        selected_model = st.selectbox(
            "Choose Transformer Model",
            ["BERT", "DistilBERT", "RoBERTa"]
        )

# Input fields
input_title = st.text_input(
    "",
    placeholder="Enter the news title here..."
)

input_description = st.text_area(
    "",
    height=200,
    placeholder="Enter the news description here..."
)

# Prediction button
if st.button("Classify News Article", type="primary", use_container_width=True):
    if input_title and input_description:
        with st.spinner('Analyzing the text...'):
            try:
                # Preprocess inputs
                processed_title = preprocess_text(input_title, is_description=False)
                processed_description = preprocess_text(input_description, is_description=True)
                
                # Get prediction with all required arguments
                prediction, confidence = predict_text(
                    processed_title, 
                    processed_description, 
                    selected_model,  # from sidebar selection
                    model_category   # from sidebar selection
                )
                
                st.markdown(f"""
                    <div class='prediction'>
                        <h3>📊 Classification Results</h3>
                        <div class='category-label'>Category: {CLASS_LABELS[prediction]}</div>
                        <div class='confidence-score'>Confidence: {confidence * 100:.2f}%</div>
                        <hr>
                        <p>Model Used: {selected_model} ({model_category})</p>
                    </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"An error occurred during classification: {str(e)}")
    else:
        st.warning("⚠️ Please enter both title and description!")

# Footer with enhanced styling
st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 1rem; background-color: #f8f9fa; border-radius: 0.5rem; margin-top: 2rem;'>
        <p style='color: #666;'>Created by Your Team Name | Turkish News Classification Project</p>
        <p style='color: #666; font-size: 0.8rem;'>Powered by Machine Learning, Deep Learning, and Transformer Models</p>
    </div>
""", unsafe_allow_html=True)