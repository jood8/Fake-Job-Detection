# Fake-Job-Detection
---

# 🕵️ Fake Job Posting Detection System

A **multimodal deep learning web application** built with **Streamlit**, **TensorFlow/Keras**, and **EasyOCR** to detect fraudulent job advertisements using both image and text analysis.

The system leverages a dual-branch neural network that combines visual features from job post images with extracted textual information to classify whether a job posting is legitimate or fake.

---

## 🚀 Live Demo

Try the application here:
👉 [https://fake-job-detection-4vwsuxzedjnzdwdsvzxxze.streamlit.app/](https://fake-job-detection-4vwsuxzedjnzdwdsvzxxze.streamlit.app/)

---

## ✨ Key Features

* 🧠 **Multimodal Deep Learning Model**
  Combines image-based CNN features with NLP-based text features for improved accuracy.

* 👁️ **OCR Text Extraction**
  Uses `EasyOCR` to automatically extract text from uploaded job advertisement images.

* 📊 **Real-time Prediction**
  Outputs a fraud probability score with instant classification results.

* 🎨 **Interactive Streamlit UI**
  Simple and clean interface with split layout for image upload and results visualization.

* ⚡ **Fast Inference Pipeline**
  Optimized preprocessing and lightweight architecture for quick predictions.

---

## 🧠 Model Architecture

The system is based on a **late-fusion multimodal neural network**:

### 🖼️ Image Branch

* Pre-trained **MobileNetV2** (frozen base)
* Custom **Conv2D + MaxPooling layers**
* Feature extraction from visual advertisement patterns

### 📝 Text Branch

* **Embedding Layer**
* **GlobalAveragePooling1D**
* Processes OCR-extracted or structured text data

### 🔗 Fusion Layer

* Concatenation of image and text features
* Fully connected Dense layers
* Dropout for regularization
* Final output: **Sigmoid activation (fraud probability)**

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **Deep Learning:** TensorFlow / Keras
* **Computer Vision:** OpenCV, MobileNetV2
* **OCR Engine:** EasyOCR
* **Data Processing:** NumPy, Pandas, Scikit-learn

---

## 📦 Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/jood8/Fake-Job-Detection.git
cd Fake-Job-Detection
```

---

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

### 3️⃣ Run the application

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
Fake-Job-Detection/
│
├── app.py
├── model/
│   ├── cnn_model.h5
│   └── text_model.h5
├── utils/
│   ├── preprocessing.py
│   └── ocr.py
├── requirements.txt
└── README.md
```

---

## 📊 Example Output

* Input: Job advertisement image
* Output:

  * Fraud Probability: `0.87`
  * Prediction: ❌ Fake Job Posting

---

## ⚠️ Limitations

* Performance depends on OCR accuracy for low-quality images
* Model may struggle with unseen job posting formats
* Requires further tuning for production-level deployment

---

## 🚀 Future Improvements

* Improve OCR preprocessing for noisy images
* Replace MobileNetV2 with EfficientNet for higher accuracy
* Add multilingual text support
* Deploy with API backend (FastAPI / Flask)
* Add dataset expansion & retraining pipeline

---

## 👩‍💻 Author

**Jood Shatnawi**


---

إ*README viral (تجيب stars أكثر)**
