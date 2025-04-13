# 🌾 Crop Disease Detection System

An AI-powered tool that utilizes the **YOLOv8 model** to detect and classify crop diseases. The system allows users to upload images of crops, analyze them using a pre-trained model, and obtain predictions along with confidence scores.

## 🚀 Features
- **Crop Disease Detection**: Upload an image to detect and classify plant diseases.
- **YOLOv8 Model Integration**: Uses a trained YOLOv8 model for high-accuracy detection.
- **Interactive UI**: Built with **Streamlit** for an intuitive user experience.
- **Performance Metrics**: Displays **F1 Score, Precision, Recall curves**, and **Confusion Matrices**.
- **Training & Validation Sample Images**: Shows model training insights.
- **Predictions Dataframe**: Preview results in a structured format.

---

## 🛠 Installation

### **1. Clone the Repository**
```sh
git clone https://github.com/krishan098/crop-disease-detection-project.git
cd crop-disease-detection-project
```

### **2. Set Up a Virtual Environment (Optional but Recommended)**
```sh
python -m venv env
source env/bin/activate  # On Windows use: env\Scripts\activate
```

### **3. Install Dependencies**
```sh
pip install -r requirements.txt
```

### **4. Install Additional System Dependencies (Linux Only)**
```sh
sudo apt update
sudo apt install -y libgl1-mesa-glx
```

### **5. Run the Application**
```sh
streamlit run app.py
```

---

## 📂 Project Structure
```
📦 crop-disease-detection
├── 📂 images
│   ├── train_batch0.jpg
│   ├── train_batch1.jpg
│   ├── train_batch2.jpg
│   ├── val_batch0_labels.jpg
│   ├── val_batch0_pred.jpg
│
├── 📂 performance curves
│   ├── F1_curve.png
│   ├── P_curve.png
│   ├── R_curve.png
│   ├── PR_curve.png
│
├── 📂 confusion matrices
│   ├── confusion_matrix.png
│   ├── confusion_matrix_normalized.png
│
├── best(1).pt  # Trained YOLOv8 model
├── predictions_df.csv  # Sample Predictions
├── app.py  # Main Streamlit application
├── requirements.txt  # Python dependencies
├── README.md  # Documentation
```

---

## 🎯 Usage Guide
1. **Upload an image**: Select a crop image (`.jpg`, `.png`, `.jpeg`).
2. **View predictions**: The system will display detected diseases, confidence scores, and bounding box locations.
3. **Analyze model performance**: Review F1 Score, Precision-Recall curves, and Confusion Matrices.
4. **Inspect sample training data**: Explore training and validation images used for model development.
5. **Check predictions data**: Preview model predictions in a tabular format.

---

## 🧪 Model Training Details
- The model was trained using **YOLOv8** on a dataset of labeled crop disease images.
- Performance metrics include **Precision, Recall, F1 Score, and Confusion Matrix analysis**.

---

## 🛠 Troubleshooting
### **1. `ImportError: libGL.so.1 not found` (Linux Only)**
```sh
sudo apt install -y libgl1-mesa-glx
```
### **2. PyTorch Class Instantiation Error**
If you encounter: `RuntimeError: Tried to instantiate class '__path__._path', but it does not exist!`, try reinstalling PyTorch:
```sh
pip uninstall torch torchvision torchaudio -y
pip cache purge
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### **3. Streamlit Crashing on Startup**
If using **Jupyter Notebook**, add:
```python
import asyncio
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
```

---

## 🧑‍💻 Authors
- **Krishan Mittal**
- **Himanshi Gupta**

---

## 📜 License
This project is licensed under the **MIT License**.

---

## ⭐ Acknowledgements
- **Ultralytics YOLOv8** for the object detection framework.
- **Streamlit** for the UI framework.
- Open-source contributors for dataset support.

If you find this project useful, please ⭐ star this repository!


![image](https://github.com/user-attachments/assets/4bd16578-9ae9-418c-9f0c-81d69c239326)
![image](https://github.com/user-attachments/assets/9ee8595e-59d7-4b65-acf9-d13f8fcdc529)
![image](https://github.com/user-attachments/assets/a1621e42-d0b9-4fea-a7ca-3f759da5cef2)
![image](https://github.com/user-attachments/assets/7dd4d0cb-6877-4abe-ba2c-a6fe4dd9efb4)

