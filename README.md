🧠 VisionCaption AI — Image Caption Generator

Team Evoastra Ventures | Internship Project

VisionCaption AI is a deep learning project that generates short descriptive captions for images using a CNN-LSTM architecture. It combines computer vision (VGG16) and natural language processing (LSTM) to describe what it “sees” in an image.

⚠️ Note: This project is built for learning purposes. The model was trained on the Flickr8k dataset (≈8,000 images), so captions may not always be accurate or grammatically perfect.

🚀 Features

Upload any image through a clean web interface

AI automatically generates a descriptive caption

CNN (VGG16) extracts image features

LSTM decodes those features into text sequences

Built using TensorFlow, Flask, and HTML/CSS/JS

🧩 Tech Stack
Component	      Technology
Frontend	      HTML5, CSS3, JavaScript
Backend  	      Flask (Python)
Deep Learning	  TensorFlow / Keras
Feature Extractor	VGG16 (pre-trained on ImageNet)
Language Model	  LSTM
Dataset	          Flickr8k

🏗️ Project Structure
image_caption_generator/
│
├── app.py                            # Flask backend
├── tokenizer.pkl                      # Tokenizer for text preprocessing
├── caption_model_evoastra_fixed.keras # Trained model file
│
├── templates/
│   ├── index.html                    # Frontend UI
│
└── README.md

⚙️ How It Works

Image Feature Extraction (CNN)
Uploaded image is resized to 224×224 pixels.
The VGG16 model (pre-trained on ImageNet) extracts high-level visual features (a 4096-dimensional vector).

Sequence Generation (LSTM)
The extracted features are fed into an LSTM model trained on the Flickr8k dataset.
The model predicts one word at a time until it forms a complete sentence.

Output Caption
The predicted caption is cleaned and displayed on the frontend with a simulated confidence score.

💻 How to Run Locally
Step 1: Clone the Repository
git clone https://github.com/yourusername/VisionCaptionAI.git
cd VisionCaptionAI

Step 2: Install Dependencies

Make sure Python 3.9+ is installed, then run:

pip install -r requirements.txt


If you don’t have a requirements file, you can manually install:

pip install flask tensorflow pillow numpy tqdm

Step 3: Run the App
python app.py


Your app will be available at:
👉 http://127.0.0.1:5000/

Open it in your browser and upload any image.

🧠 Model Training Overview

Dataset: Flickr8k Dataset
Architecture: VGG16 + LSTM
Max Caption Length: 38
Vocabulary Size: ~5000 words
Epochs: 10
Framework: TensorFlow / Keras

The model was trained on a limited dataset (8k images) to keep training time reasonable, which limits caption accuracy.

📸 Example Outputs
Image	Predicted Caption
🐶 Dog playing in a park	“a dog running in the grass”
🏖️ People on the beach	“a group of people standing near water”
🚲 Child with a bike	“a young boy riding a bicycle”

(Captions vary slightly with each prediction)

⚠️ Limitations

Accuracy is limited by the small training dataset (8k images).

Captions may be short, repetitive, or partially incorrect.

Not suitable for production without retraining on larger datasets like MS-COCO or Flickr30k.

🔮 Future Improvements

Replace VGG16 with InceptionV3 or EfficientNet for better image understanding.

Train the model on larger datasets for better vocabulary and context.

Experiment with Transformer-based models (BLIP, ViT-GPT2, or InstructBLIP).

Add multilingual caption generation support.

## ⚙️ Model Setup

This repository does not include large binary files (`.keras`, `.pkl`) due to GitHub’s file size limits.

Before running the Flask app, make sure you:
1. Run the training notebook (`model.ipynb`) to:
   - Extract features from the Flickr8k dataset
   - Train the caption generation model
   - Save outputs as:
     - `caption_model_evoastra_fixed.keras`
     - `tokenizer.pkl`
     - `features.pkl`
2. Move both files to the project root (same folder as `app.py`).
3. Start the Flask app:
   ```bash
   python app.py


❤️ Acknowledgments

Flickr8k Dataset on Kaggle

TensorFlow

Keras

Team Evoastra Ventures for development and experimentation

📜 License

This project is open-source under the MIT License.
You are free to use, modify, and distribute it for educational purposes.

👨‍💻 Author

Omkar Mhamunkar
Full Stack Developer & AI Enthusiast
📧 omkarvmhamunkar@gmail.com

