import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(page_title="Segmentasi Konsumen")

st.title("Aplikasi Segmentasi Konsumen")
st.write("Segmentasi pelanggan menggunakan metode K-Means Clustering")

# CEK FILE MODEL
if not os.path.exists("kmeans_model.pkl"):
    st.error("File kmeans_model.pkl TIDAK ditemukan!")
    st.stop()

if not os.path.exists("scaler.pkl"):
    st.error("File scaler.pkl TIDAK ditemukan!")
    st.stop()

# LOAD MODEL (AMAN)
kmeans = pickle.load(open("kmeans_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Upload data
uploaded_file = st.file_uploader("Upload Dataset Mall Customers", type=["csv"])

if uploaded_file is None:
    st.info("Silakan upload file CSV untuk memulai")
    st.stop()

df = pd.read_csv(uploaded_file)

st.subheader("Preview Dataset")
st.dataframe(df.head())

# CEK KOLOM
required_cols = ['Annual Income (k$)', 'Spending Score (1-100)']
if not all(col in df.columns for col in required_cols):
    st.error("Kolom dataset tidak sesuai dengan Mall_Customers.csv")
    st.write("Kolom ditemukan:", df.columns.tolist())
    st.stop()

X = df[required_cols]
X_scaled = scaler.transform(X)
df['Cluster'] = kmeans.predict(X_scaled)

st.subheader("Hasil Segmentasi Konsumen")
st.dataframe(df)

st.subheader("Visualisasi Cluster")
fig, ax = plt.subplots()
sns.scatterplot(
    x=required_cols[0],
    y=required_cols[1],
    hue='Cluster',
    data=df,
    palette='Set2',
    ax=ax
)
st.pyplot(fig)

st.subheader("Prediksi Cluster Pelanggan Baru")
income = st.number_input("Annual Income (k$)", min_value=0)
score = st.number_input("Spending Score (1-100)", min_value=0, max_value=100)

if st.button("Prediksi"):
    data = scaler.transform([[income, score]])
    cluster = kmeans.predict(data)
    st.success(f"Pelanggan termasuk dalam Cluster {cluster[0]}")
