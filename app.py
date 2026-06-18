# -*- coding: utf-8 -*-

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import pickle
import easyocr

from tensorflow.keras.preprocessing.sequence import pad_sequences


# ======================================
# PAGE CONFIG
# ======================================
st.set_page_config(
    page_title="Fake Job Detection",
    page_icon="🕵️",
    layout="wide"
)


# ======================================
# HEADER
# ======================================
st.title("🕵️ Fake Job Posting Detection")
st.divider()


# ======================================
# LOAD MODEL + TOKENIZER + OCR
# ======================================
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model(
        "fake_job_multimodal_model.keras"
    )

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    # إضافة gpu=False أو True حسب السيرفر لتسريع التحميل
    reader = easyocr.Reader(['en'], gpu=tf.config.list_physical_devices('GPU') != [])

    return model, tokenizer, reader


try:
    model, tokenizer, reader = load_assets()
except Exception as e:
    st.error(f"Loading Error:\n{e}")
    st.stop()


# ======================================
# MAIN LAYOUT
# ======================================
col1, col2 = st.columns([1, 1.2])


# ======================================
# LEFT COLUMN (Inputs)
# ======================================
with col1:
    st.subheader("Upload Job Image")

    uploaded_file = st.file_uploader(
        "Choose Job Advertisement",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        st.image(
            uploaded_file,
            caption="Uploaded Advertisement Preview",
            use_container_width=True
        )


# ======================================
# RIGHT COLUMN (Controls & Results)
# ======================================
with col2:
    st.subheader("AI Detection Control")

    detect = st.button(
        "Run Auto-Detection",
        use_container_width=True,
        type="primary"
    )

    # نضع مساحة فارغة (Placeholder) للنتائج حتى تظهر بشكل مرتب تحت الزر مباشرة
    results_container = st.container()


# ======================================
# PREDICTION & RESULTS
# ======================================
if detect:
    if uploaded_file is None:
        st.warning("Please upload an image first.")
    else:
        with st.spinner("Analyzing Job Posting components..."):
            try:
                # =========================
                # IMAGE READING
                # =========================
                file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # =========================
                # OCR TEXT EXTRACTION
                # =========================
                # الأفضل تمرير img_rgb لـ easyocr لضمان دقة أعلى
                ocr_results = reader.readtext(img_rgb)
                combined_text = " ".join([res[1] for res in ocr_results])

                # حماية في حال كانت الصورة فارغة أو لم يجد OCR أي نص
                if not combined_text.strip():
                    combined_text = "No text detected in the image."

                # =========================
                # TEXT PREPROCESSING
                # =========================
                sequences = tokenizer.texts_to_sequences([combined_text])
                X_text = pad_sequences(sequences, maxlen=200, padding='post')

                # =========================
                # IMAGE PREPROCESSING
                # =========================
                img_resized = cv2.resize(img_rgb, (128, 128))
                X_img = np.array([img_resized]) / 255.0

                # =========================
                # MODEL PREDICTION
                # =========================
                # تنبيه: تأكد أن الترتيب هنا [الصورة، النص] هو نفس ترتيب تدريب الموديل
                prediction = model.predict([X_img, X_text])
                score = float(prediction[0][0])
                confidence = score * 100

                # ======================================
                # DISPLAY RESULTS (Inside Container)
                # ======================================
                with results_container:
                    st.divider()
                    st.subheader("System Analytics Output")

                    if score > 0.5:
                        st.error("Result: Fake / Fraudulent Job Posting")
                        st.metric(
                            label="Fraud Probability Score",
                            value=f"{confidence:.2f}%"
                        )
                    else:
                        st.success("Result: Safe & Real Job Posting")
                        # لعرض نسبة الأمان الحقيقية بدلاً من نسبة الاحتيال المنخفضة
                        safe_confidence = (1 - score) * 100
                        st.metric(
                            label="Safe Probability Score",
                            value=f"{safe_confidence:.2f}%"
                        )

                    # ======================================
                    # OCR DETAILS
                    # ======================================
                    st.write("") 
                    st.subheader("Extracted OCR Text")
                    st.text_area(
                        label="Detected Text",
                        value=combined_text,
                        height=150
                    )

            except Exception as e:
                st.error(f"Prediction Error:\n{e}")