import streamlit as st

st.set_page_config(
    page_title="Turkish News Classifier",
    page_icon="📰",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .header-container {
        text-align: center;
        padding: 2rem;
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        margin-bottom: 2rem;
        border: 1px solid #dee2e6;
    }
    .section-container {
        padding: 1.5rem;
        background-color: white;
        border-radius: 0.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid #dee2e6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .feature-box {
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border: 1px solid #e9ecef;
    }
    .citation {
        padding: 1rem;
        background-color: #f8f9fa;
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="header-container">
        <h1>📰 Turkish News Classifier</h1>
        <p style='font-size: 1.2rem; color: #666;'>
            An Advanced Multi-Model Approach to Turkish News Classification
        </p>
        <p style='font-size: 1rem; color: #666;'>
            ELE513 Course Project
        </p>
    </div>
""", unsafe_allow_html=True)

# Project Overview
st.markdown("""
    <div class="section-container">
        <h2>🎯 Project Overview</h2>
        <p>
            This project implements a comprehensive Turkish news classification system using various machine learning approaches. 
            Our system can categorize news articles into four main categories: World News, Sports, Business, and Science & Technology.
        </p>
    </div>
""", unsafe_allow_html=True)

# Features
st.markdown("""
    <div class="section-container">
        <h2>✨ Key Features</h2>
        <div class="feature-box">
            <h4>🤖 Multiple Model Support</h4>
            <ul>
                <li>Traditional Machine Learning (6 models)</li>
                <li>Deep Learning (LSTM & RNN with TF-IDF)</li>
                <li>Transformer Models (BERT, DistilBERT, RoBERTa)</li>
            </ul>
        </div>
        <div class="feature-box">
            <h4>🛠️ Two Operating Modes</h4>
            <ul>
                <li><strong>Manual Entry:</strong> Test individual articles</li>
                <li><strong>Batch Processing:</strong> Analyze multiple articles with performance metrics</li>
            </ul>
        </div>
        <div class="feature-box">
            <h4>📊 Advanced Analytics</h4>
            <ul>
                <li>Confusion Matrix Visualization</li>
                <li>Performance Metrics (Accuracy, F1-Score)</li>
                <li>Confidence Scores for Predictions</li>
            </ul>
        </div>
    </div>
""", unsafe_allow_html=True)

# Research Background
st.markdown("""
    <div class="section-container">
        <h2>📚 Research Background</h2>
        
        <h4>Original Paper</h4>
        <div class="citation">
            <p><em>"Machine learning models for news article classification"</em></p>
            <p>Authors: Naseeba Beebi, et al.</p>
            <p>Published in: 2023 5th International Conference on Smart Systems and Inventive Technology (ICSSIT)</p>
            <p>Publisher: IEEE, 2023</p>
        </div>
        
        <h4>Our Implementation</h4>
        <div class="citation">
            <p><em>"Haber Metni Sınıflandırma"</em></p>
            <p>Authors: 
                <ul>
                    <li>Gökçe Başak Demirok (g.demirok@etu.edu.tr)</li>
                    <li>Selin Mergen (s.mergen@etu.edu.tr)</li>
                </ul>
            </p>
            <p>Department: Computer Engineering</p>
            <p>Institution: TOBB Economics and Technology University</p>
        </div>
        
        <h4>Project Resources</h4>
        <ul>
            <li><a href="https://drive.google.com/file/d/1jpjE8BU40m7XPHKa-bL6Zyt8WktgadQy/view?usp=sharing" target="_blank">Project Code Repository</a></li>
            <li><a href="https://drive.google.com/file/d/1EXZbEocHjJkbMntEjVW8oeSQTs0bQk6f/view?usp=sharing" target="_blank">Project Paper</a></li>
        </ul>
    </div>
""", unsafe_allow_html=True)

# Dataset Information
st.markdown("""
    <div class="section-container">
        <h2>📊 Dataset Information</h2>
        <p>The system is trained on a comprehensive Turkish news dataset containing:</p>
        <ul>
            <li>Categories: 4 main categories (World News, Sports, Business, Science & Technology)</li>
            <li>Balanced distribution across categories</li>
            <li>Turkish language news articles from various sources</li>
        </ul>
    </div>
""", unsafe_allow_html=True)

# Team Information
st.markdown("""
    <div class="section-container">
        <h2>👥 Team Information</h2>
        <h4>Project Team</h4>
        <ul>
            <li>Gökçe Başak Demirok - Computer Engineering</li>
            <li>Selin Mergen - Computer Engineering</li>
        </ul>
        
        <h4>Institution</h4>
        <p>TOBB Economics and Technology University</p>
        <p>Department of Computer Engineering</p>
        <p>ELE513 Course Project</p>
    </div>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
    <div style='text-align: center; padding: 1rem; background-color: #f8f9fa; border-radius: 0.5rem; margin-top: 2rem;'>
        <p style='color: #666;'>© 2024 TOBB ETU - Turkish News Classification Project</p>
        <p style='color: #666; font-size: 0.8rem;'>
            For academic and research purposes only. All rights reserved.
        </p>
    </div>
""", unsafe_allow_html=True)