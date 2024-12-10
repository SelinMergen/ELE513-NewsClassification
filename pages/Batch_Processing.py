import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from pages.Manual_Entry import predict_text, CLASS_LABELS

st.set_page_config(
    page_title="Batch Processing - Turkish News Classifier",
    page_icon="📊",
    layout="wide"
)

# Enhanced Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .upload-container {
        padding: 2rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        margin-bottom: 1rem;
        border: 1px solid #dee2e6;
    }
    .results-container {
        padding: 2rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        margin-top: 1rem;
        border: 1px solid #4CAF50;
    }
    .metrics-container {
        display: flex;
        justify-content: space-around;
        margin: 2rem 0;
    }
    .metric-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Update CLASS_LABELS to match all possible classes
CLASS_LABELS = {
    0: "🌍 Dünya Haberleri",
    1: "⚽ Spor",
    2: "💼 İş Dünyası",
    3: "🔬 Bilim ve Teknoloji",
    4: "❓ Diğer"  # Add any additional class if needed
}

CLASS_LABELS_WITHOUT_EMOJI = {
    0: "Dünya Haberleri",
    1: "Spor",
    2: "İş Dünyası",
    3: "Bilim ve Teknoloji",
    4: "Diğer"
}

def process_batch(df, model_name, model_category):
    predictions = []
    confidences = []
    
    for _, row in df.iterrows():
        pred, conf = predict_text(row['title'], row['description'], model_name, model_category)
        predictions.append(pred)
        confidences.append(conf)
    
    df['predicted_label'] = predictions
    df['confidence'] = confidences
    df['predicted_category'] = df['predicted_label'].map(CLASS_LABELS)
    
    return df

def plot_confusion_matrix(y_true, y_pred):
    # Get unique labels from both true and predicted
    labels = sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[CLASS_LABELS_WITHOUT_EMOJI.get(i, f"Class {i}") for i in labels],
                yticklabels=[CLASS_LABELS_WITHOUT_EMOJI.get(i, f"Class {i}") for i in labels])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    return plt

st.title("📊 Batch Processing - Turkish News Classifier")

# Model selection (same as in manual entry)
with st.sidebar:
    st.markdown("""
        <div style='text-align: center; padding: 1rem; background-color: #f8f9fa; border-radius: 0.5rem; margin-bottom: 1rem;'>
            <h2 style='color: #1a1a1a;'>Model Selection</h2>
        </div>
    """, unsafe_allow_html=True)
    
    model_category = st.radio(
        "Select Model Category",
        ["Machine Learning", "Deep Learning", "Transformers"]
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

# Instructions
with st.expander("📖 Instructions"):
    st.markdown("""
    1. Upload a CSV file containing at least 'title' and 'description' columns
    2. Optionally include a 'label' column for performance evaluation
    3. Click 'Process Data' to classify the news articles
    4. Download the results or view performance metrics
    """) 
    
# File upload
st.subheader("Upload Data")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Check if required columns exist
    required_cols = ['title', 'description']
    df.columns = df.columns.str.lower()
    if not all(col in df.columns for col in required_cols):
        st.error("CSV file must contain 'title' and 'description' columns!")
    else:
        # Process button
        if st.button("Process Data"):
            with st.spinner('Processing data...'):
                df = df[required_cols] if 'label' not in df.columns else df[required_cols + ['label']]
                # Process the data
                results_df = process_batch(df.copy(), selected_model, model_category)
                
                # Display results
                st.subheader("Results")
                st.dataframe(results_df)
                
                # Download button for results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Results",
                    data=csv,
                    file_name="classified_news.csv",
                    mime="text/csv"
                )
                
                # If true labels are available
                if 'label' in df.columns:
                    st.subheader("Performance Metrics")
                    
                    # Get unique labels from both true and predicted values
                    unique_labels = sorted(list(set(df['label']) | set(results_df['predicted_label'])))
                    
                    # Classification report with proper labels
                    report = classification_report(
                        df['label'],
                        results_df['predicted_label'],
                        labels=unique_labels,
                        target_names=[CLASS_LABELS.get(i, f"Class {i}") for i in unique_labels],
                        output_dict=True
                    )
                    
                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Accuracy", f"{report['accuracy']:.2%}")
                    with col2:
                        st.metric("Macro F1-Score", f"{report['macro avg']['f1-score']:.2%}")
                    with col3:
                        st.metric("Weighted F1-Score", f"{report['weighted avg']['f1-score']:.2%}")
                    
                    # Confusion Matrix
                    st.subheader("Confusion Matrix")
                    fig = plot_confusion_matrix(df['label'], results_df['predicted_label'])
                    st.pyplot(fig)

