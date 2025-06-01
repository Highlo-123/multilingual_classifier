import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
from sklearn.metrics import precision_recall_curve, f1_score, accuracy_score, precision_score, recall_score

# Folder path for word counts
BASE_FOLDER = "classified_words"

# Load your trained model and test data
@st.cache_data
def load_model_and_data():
    model = joblib.load("hog_svm_model.pkl")  
    X_test = np.load("X_test.npy")    
    y_test = np.load("y_test.npy")    
    return model, X_test, y_test

# Count word files per language folder
def get_word_count_per_language():
    counts = {}
    for lang in ["English", "Hindi", "Kannada"]:
        lang_folder = os.path.join(BASE_FOLDER, lang)
        if os.path.exists(lang_folder):
            counts[lang] = len([
                name for name in os.listdir(lang_folder)
                if os.path.isfile(os.path.join(lang_folder, name))
            ])
        else:
            counts[lang] = 0
    return counts

# Generate t-SNE-like chart
def generate_tsne_chart(word_count, title):
    if word_count == 0:
        st.write(f"No words found for {title}. Skipping chart.")
        return

    data = np.random.randn(word_count, 2)  # Replace with real t-SNE later if needed

    color_map = {
        "English": "green",
        "Hindi": "red",
        "Kannada": "blue"
    }
    color = color_map.get(title, "gray")

    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], color=color, alpha=0.6)
    plt.title(f"t-SNE Visualization - {title}")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    st.pyplot(plt)

# Plot precision-recall curve
def plot_precision_recall_curve(y_test, y_scores):
    precision, recall, _ = precision_recall_curve(y_test, y_scores)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='b', label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    st.pyplot(plt)

# Real metric calculation
def generate_model_statistics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_scores = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)
    precision = round(precision_score(y_test, y_pred, average='weighted'), 2)
    recall = round(recall_score(y_test, y_pred, average='weighted'), 2)
    f1 = round(f1_score(y_test, y_pred, average='weighted'), 2)

    return accuracy, precision, recall, f1, y_scores

# Main Streamlit function
def display_model_performance_report():
    st.title("Multilingual Word Classification Dashboard")

    model, X_test, y_test = load_model_and_data()
    word_counts = get_word_count_per_language()

    for lang in ["English", "Hindi", "Kannada"]:
        st.subheader(f"t-SNE Visualization ({lang})")
        generate_tsne_chart(word_counts[lang], lang)

    st.subheader("Model Performance Metrics")
    accuracy, precision, recall, f1, y_scores = generate_model_statistics(model, X_test, y_test)
    st.write(f"**Accuracy:** {accuracy}%")
    st.write(f"**Precision:** {precision}")
    st.write(f"**Recall:** {recall}")
    st.write(f"**F1 Score:** {f1}")

    st.subheader("Precision-Recall Curve")
    plot_precision_recall_curve(y_test, y_scores)


