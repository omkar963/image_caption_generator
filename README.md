# image_caption_generator
# 🧠 Image Caption Generator

A deep learning project that automatically generates descriptive captions for images using **InceptionV3** for feature extraction and **LSTM** for sequence generation.

## 🚀 Features
- Uses **MS COCO** dataset for training
- Extracts visual features using pretrained **InceptionV3**
- Generates natural-language captions using **LSTM**
- Evaluated with **BLEU score**
- Built and trained in **Google Colab**

## 🧩 Architecture
1. **Data Preparation** – Image resizing, normalization, caption tokenization  
2. **Feature Extraction** – CNN (InceptionV3) generates image embeddings  
3. **Caption Generation** – LSTM network decodes image features into text  
4. **Evaluation** – BLEU score and qualitative visual analysis  

## ⚙️ Tech Stack
`Python`, `TensorFlow/Keras`, `NumPy`, `Matplotlib`, `MS COCO`, `Flask`

## 📊 Results
Example:  
**Input:** 🖼️ (Cat on a sofa)  
**Output:** “A cat is sitting on the sofa.”

## 👥 Team
Project by *[Team E Evoastra]* — developed collaboratively using **GitHub** and **Google Colab**.
