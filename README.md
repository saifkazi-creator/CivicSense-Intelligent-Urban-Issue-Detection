🏙️ CivicSense: Intelligent Urban Issue Detection (ML Backend)
4

🚀 AI-powered system for detecting urban issues like potholes and garbage using computer vision.

🔗 Repository: https://github.com/saifkazi-creator/CivicSense-Intelligent-Urban-Issue-Detection

🚀 Overview

CivicSense is a computer vision-based system designed to detect urban issues such as potholes, garbage, and other infrastructure problems from images.

This project focuses on building the machine learning backend, which can later be integrated into a full smart city platform.

🎯 Problem Statement

Urban issues are typically reported manually, which is slow and inefficient.

👉 This project aims to:

Automate detection of civic issues using AI
Improve reporting efficiency
Enable scalable smart city solutions
🧠 Features
📸 Image classification using CNN
🤖 Detects:
Potholes
Garbage
Other urban elements
⚡ Training + validation pipeline
🧪 Model evaluation with metrics
🛠️ Tech Stack
Python
TensorFlow / Keras
OpenCV
NumPy
📂 Project Structure
CivicSense/
│── train.py
│── predict.py
│── requirements.txt
│── README.md
│── .gitignore
│── civicsense_model.h5
⚙️ Installation
git clone https://github.com/saifkazi-creator/CivicSense-Intelligent-Urban-Issue-Detection.git
cd CivicSense-Intelligent-Urban-Issue-Detection
pip install -r requirements.txt
▶️ Usage
🔹 Train the model
python train.py
🔹 Run prediction
python predict.py
🧠 Model Architecture

The model is a Convolutional Neural Network (CNN) consisting of:

3 Convolution layers with ReLU activation
MaxPooling layers for downsampling
Fully connected Dense layers
Softmax output layer (3 classes)
📊 Model Details
Image Size: 224 × 224
Batch Size: 32
Epochs: 10
Optimizer: Adam
Loss Function: Sparse Categorical Crossentropy
📈 Model Performance
🔹 Classification Report
Class        Precision    Recall    F1-Score    Support
------------------------------------------------------
Garbage        1.00        0.50       0.67         2
Potholes       0.67        1.00       0.80         2

Accuracy                               0.75         4
Macro Avg      0.83        0.75       0.73         4
Weighted Avg   0.83        0.75       0.73         4
🔹 Confusion Matrix
[[1 1]
 [0 2]]
🧠 Interpretation

The model achieves an overall accuracy of 75% on the test dataset.

The model performs well in detecting potholes (100% recall)
Garbage detection has lower recall (50%), indicating some misclassifications
The confusion matrix shows that 1 garbage image was misclassified

👉 This suggests:

Need for larger dataset
Better class balance
Potential improvement using transfer learning
📊 Dataset

The dataset includes images of:

Potholes
Garbage
Other categories

⚠️ Dataset is not included due to size limitations.

👉 Dataset link: (Add your Kaggle / Google Drive link here)

🔮 Future Scope
🌐 Web interface for real-time predictions
📱 Mobile application integration
☁️ Cloud deployment (AWS/GCP)
🔍 Upgrade to advanced models (MobileNet, ResNet)
🤝 Contributing

Contributions are welcome! Feel free to fork and improve the project.

📜 License

This project is licensed under the MIT License.
