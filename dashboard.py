import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Load metrics.json
try:
    with open("outputs/metrics.json") as f:
        metrics = json.load(f)
except FileNotFoundError:
    st.error("metrics.json not found. Run your pipeline first.")
    st.stop()

st.title("📊 ML Pipeline Dashboard")

# Show scalar metrics
st.subheader("Model Performance")
st.write(f"Accuracy: {metrics.get('accuracy', 'N/A')}")
st.write(f"Precision: {metrics.get('precision', 'N/A')}")
st.write(f"Recall: {metrics.get('recall', 'N/A')}")

# Confusion matrix
if "confusion_matrix" in metrics:
    st.subheader("Confusion Matrix")
    cm = pd.DataFrame(metrics["confusion_matrix"])
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

# Training history
if "train_history" in metrics:
    st.subheader("Training History")
    history = metrics["train_history"]
    fig, ax = plt.subplots()
    ax.plot(history["epoch"], history["loss"], marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    st.pyplot(fig)

# Optional: interactive prediction demo
st.subheader("Try Predictions")
uploaded = st.file_uploader("Upload CSV for inference", type="csv")
if uploaded:
    df = pd.read_csv(uploaded)
    st.write("Uploaded data preview:", df.head())
    st.info("Hook this into your inference script to show predictions live.")
