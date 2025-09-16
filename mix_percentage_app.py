# mix_percentage_app.py
import streamlit as st
from PIL import Image
import numpy as np
import cv2
from sklearn.cluster import KMeans
import io

st.title("Mixing Percentage from Image — Quick Demo")

uploaded = st.file_uploader("Upload image (top-down or clear view)", type=["jpg","jpeg","png"])
n_clusters = st.slider("Number of color clusters (start with 2)", 2, 6, 2)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Original image", use_column_width=True)
    img = np.array(image)
    h, w = img.shape[:2]
    # Optional: Let user define ROI by cropping (simpler: use whole image)
    # Preprocess
    img_small = cv2.resize(img, (400, int(400*h/w)))
    img_lab = cv2.cvtColor(img_small, cv2.COLOR_RGB2LAB)
    # reshape for clustering
    X = img_lab.reshape((-1,3)).astype(np.float32)
    km = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    labels = km.labels_.reshape(img_small.shape[0], img_small.shape[1])
    # Count pixels per cluster
    counts = [(labels==i).sum() for i in range(n_clusters)]
    total = sum(counts)
    # Map cluster centers to colors (for visualization)
    centers_lab = km.cluster_centers_.astype(np.uint8)
    centers_rgb = cv2.cvtColor(centers_lab.reshape(1,-1,3), cv2.COLOR_LAB2RGB).reshape(-1,3)
    # Build segmented preview
    seg_img = np.zeros_like(img_small)
    for i in range(n_clusters):
        seg_img[labels==i] = centers_rgb[i]
    st.image(seg_img, caption="Segmented preview", use_column_width=True)
    # Show percentages sorted by area
    perc = [(i, counts[i]/total*100, centers_rgb[i].tolist()) for i in range(n_clusters)]
    perc_sorted = sorted(perc, key=lambda x: x[1], reverse=True)
    st.write("Estimated composition (by area):")
    for idx, p, col in perc_sorted:
        st.write(f"Cluster {idx}: {p:.2f}% — representative color: {col}")
    st.write("Notes:")
    st.write("- This measures fraction of the image area belonging to each visual cluster.")
    st.write("- For better accuracy: crop to the mixture region, use good lighting, and choose correct #clusters.")
