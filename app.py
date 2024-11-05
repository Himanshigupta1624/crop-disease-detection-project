
import streamlit as st
from ultralytics import YOLO 
from PIL import Image
import numpy as np
import pandas as pd

# Page configuration
st.set_page_config(page_title="Crop Disease Detection System", page_icon="🌾", layout="centered")

# Sidebar
st.sidebar.title("🌾 Crop Disease Detection")
st.sidebar.write("An AI-powered tool to help detect crop diseases using the YOLOv8 model. Upload an image to get started.")

# Load your model
model_path = 'best(1).pt'  # Adjust this path as needed

# Load the model
@st.cache_resource  # Cache the model to avoid reloading on every run
def load_model():
    try:
        model = YOLO(model_path)
        st.sidebar.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.sidebar.error("Failed to load the model.")
        st.sidebar.write(e)
        return None

model = load_model()

# Main app content
st.title("🌾 Crop Disease Detection System")
st.write("Upload a crop image below, and the system will detect and classify any diseases present.")

# Image upload
uploaded_file = st.file_uploader("Upload a crop image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("### Analyzing...")

    # Run YOLOv8 model on the image
    results = model.predict(np.array(image))

    # # Display predictions
    st.write("### Detection Results")
    detected_image = results[0].plot()  # Get the image with annotations

    # Display annotated image
    st.image(detected_image, caption="Predicted Output", use_column_width=True)

    # Display detection details
    st.write("### Detection Details")
    boxes = results[0].boxes  # Get the bounding boxes

    if boxes:
        for box in boxes:
            xmin, ymin, xmax, ymax = box.xyxy[0].cpu().numpy()  # Coordinates
            confidence = box.conf[0].item()  # Confidence score
            class_id = int(box.cls[0].item())  # Class ID
            class_name = results[0].names[class_id]  # Class name

            st.write(f"**Detected:** {class_name} (Confidence: {confidence:.2f})")
            # st.write(f"**Confidence:** {confidence:.2f}")
            st.write(f"**Location:** x_min={xmin:.2f}, y_min={ymin:.2f}, x_max={xmax:.2f}, y_max={ymax:.2f}")
    else:
        st.write("No diseases detected in the image.")

else:
    st.write("Please upload an image to start detection.")
# Display sample images from training batches
st.write("---")
st.write("### Sample Training and Validation Images")
try:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("images/train_batch0.jpg", caption="Training Batch 0", use_column_width=True)
    with col2:
        st.image("images/train_batch1.jpg", caption="Training Batch 1", use_column_width=True)
    with col3:
        st.image("images/train_batch2.jpg", caption="Training Batch 2", use_column_width=True)

    val_col1, val_col2 = st.columns(2)
    with val_col1:
        st.image("images/val_batch0_labels.jpg", caption="Validation Batch 0 Labels", use_column_width=True)
    with val_col2:
        st.image("images/val_batch0_pred.jpg", caption="Validation Batch 0 Predictions", use_column_width=True)
except Exception as e:
    st.write("Error loading training and validation images.")
    st.write(e)

# Display CSV content (predictions and submissions)
st.write("---")
st.write("### CSV Data Preview")
try:
    predictions_df = pd.read_csv("predictions_df.csv")
    

    st.write("**Predictions Dataframe**")
    st.dataframe(predictions_df.head())

    
except Exception as e:
    st.write("Error loading CSV data.")
    st.write(e)    

# Model performance graphs
st.write("---")
st.write("### Model Performance")

# Paths to your performance metrics images
f1_curve_path = "performance curves/F1_curve.png"
precision_curve_path = "performance curves/P_curve.png"
recall_curve_path = "performance curves/R_curve.png"
pr_curve_path = "performance curves/PR_curve.png"
confusion_matrix_path = "confusion matrices/confusion_matrix.png"
confusion_matrix_normalized_path = "confusion matrices/confusion_matrix_normalized.png"

# Display the performance metrics in a compact grid
try:
    col1, col2 = st.columns(2)

    with col1:
        st.write("#### F1 Score Curve")
        st.image(f1_curve_path, caption="F1 Score Curve", use_column_width=True)
        st.write("#### Precision Curve")
        st.image(precision_curve_path, caption="Precision Curve", use_column_width=True)

    with col2:
        st.write("#### Recall Curve")
        st.image(recall_curve_path, caption="Recall Curve", use_column_width=True)
        st.write("#### Precision-Recall Curve")
        st.image(pr_curve_path, caption="Precision-Recall Curve", use_column_width=True)

    # Display confusion matrices in a row
    st.write("#### Confusion Matrices")
    cm_col1, cm_col2 = st.columns(2)
    with cm_col1:
        st.image(confusion_matrix_path, caption="Confusion Matrix", use_column_width=True)
    with cm_col2:
        st.image(confusion_matrix_normalized_path, caption="Normalized Confusion Matrix", use_column_width=True)

except Exception as e:
    st.write("Error loading performance metrics images.")
    st.write(e)

# Additional information section
st.write("---")
st.write("### About This App")
st.write("""
This application uses a YOLOv8 model trained on crop disease images to help farmers and agricultural professionals detect plant diseases early. 
The model classifies various diseases based on the visual characteristics of crops in the uploaded image. 
Early detection can assist in taking timely interventions for better crop health management.
""")

# Sidebar footer
st.sidebar.write("---")
st.sidebar.write("Developed by: **Krishan Mittal and Himanshi Gupta**")


# import streamlit as st
# from ultralytics import YOLO
# from PIL import Image
# import pandas as pd
# import numpy as np

# # Page configuration
# st.set_page_config(page_title="Crop Disease Detection System", page_icon="🌾", layout="wide")

# # Sidebar
# st.sidebar.title("🌾 Crop Disease Detection")
# st.sidebar.write("An AI-powered tool to detect crop diseases using the YOLOv8 model. Upload an image to get started.")

#  model
# model_path = 'yolov8n.pt'

# @st.cache_resource  # Cache the model to avoid reloading on every run
# def load_model():
#     try:
#         model = YOLO(model_path)
#         st.sidebar.success("Model loaded successfully!")
#         return model
#     except Exception as e:
#         st.sidebar.error("Failed to load the model.")
#         st.sidebar.write(e)
#         return None

# model = load_model()


# st.title("🌾 Crop Disease Detection System")
# st.write("Upload a crop image below, and the system will detect and classify any diseases present.")

# uploaded_file = st.file_uploader("Upload a crop image", type=["jpg", "png", "jpeg"])

# if uploaded_file:
#     # Display uploaded image
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Uploaded Image", use_column_width=True)
#     st.write("### Analyzing...")


#     results = model.predict(np.array(image))

#     
#     st.write("### Detection Results")
#     detected_image = results[0].plot()

#     
#     st.image(detected_image, caption="Predicted Output", use_column_width=True)


#     st.write("### Detection Details")
#     boxes = results[0].boxes

#     if boxes:
#         for box in boxes:
#             xmin, ymin, xmax, ymax = box.xyxy[0].cpu().numpy()
#             confidence = box.conf[0].item()
#             class_id = int(box.cls[0].item())
#             class_name = results[0].names[class_id]

#             st.write(f"**Detected:** {class_name} (Confidence: {confidence:.2f})")
#             st.write(f"**Location:** x_min={xmin:.2f}, y_min={ymin:.2f}, x_max={xmax:.2f}, y_max={ymax:.2f}")
#     else:
#         st.write("No diseases detected in the image.")
# else:
#     st.write("Please upload an image to start detection.")

# st.write("---")
# st.write("### Sample Training and Validation Images")
# try:
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         st.image("images/train_batch0.jpg", caption="Training Batch 0", use_column_width=True)
#     with col2:
#         st.image("images/train_batch1.jpg", caption="Training Batch 1", use_column_width=True)
#     with col3:
#         st.image("images/train_batch2.jpg", caption="Training Batch 2", use_column_width=True)

#     val_col1, val_col2 = st.columns(2)
#     with val_col1:
#         st.image("images/val_batch0_labels.jpg", caption="Validation Batch 0 Labels", use_column_width=True)
#     with val_col2:
#         st.image("images/val_batch0_pred.jpg", caption="Validation Batch 0 Predictions", use_column_width=True)
# except Exception as e:
#     st.write("Error loading training and validation images.")
#     st.write(e)


# st.write("---")
# st.write("### CSV Data Preview")
# try:
#     predictions_df = pd.read_csv("predictions_df.csv")
    

#     st.write("**Predictions Dataframe**")
#     st.dataframe(predictions_df.head())

    
# except Exception as e:
#     st.write("Error loading CSV data.")
#     st.write(e)

# # Model performance graphs
# st.write("---")
# st.write("### Model Performance")


# f1_curve_path = "performance curves/F1_curve.png"
# precision_curve_path = "performance curves/P_curve.png"
# recall_curve_path = "performance curves/R_curve.png"
# pr_curve_path = "performance curves/PR_curve.png"
# confusion_matrix_path = "confusion matrices/confusion_matrix.png"
# confusion_matrix_normalized_path = "confusion matrices/confusion_matrix_normalized.png"

# # Display performance metrics in a compact layout
# try:
#     col1, col2 = st.columns(2)

#     with col1:
#         st.write("#### F1 Score Curve")
#         st.image(f1_curve_path, caption="F1 Score Curve", use_column_width=True)
#         st.write("#### Precision Curve")
#         st.image(precision_curve_path, caption="Precision Curve", use_column_width=True)

#     with col2:
#         st.write("#### Recall Curve")
#         st.image(recall_curve_path, caption="Recall Curve", use_column_width=True)
#         st.write("#### Precision-Recall Curve")
#         st.image(pr_curve_path, caption="Precision-Recall Curve", use_column_width=True)

#     # Display confusion matrices
#     st.write("#### Confusion Matrices")
#     cm_col1, cm_col2 = st.columns(2)
#     with cm_col1:
#         st.image(confusion_matrix_path, caption="Confusion Matrix", use_column_width=True)
#     with cm_col2:
#         st.image(confusion_matrix_normalized_path, caption="Normalized Confusion Matrix", use_column_width=True)

# except Exception as e:
#     st.write("Error loading performance metrics images.")
#     st.write(e)


# st.write("---")
# st.write("### About This App")
# st.write("""
# This application uses a YOLOv8 model trained on crop disease images to help farmers and agricultural professionals detect plant diseases early. 
# The model classifies various diseases based on the visual characteristics of crops in the uploaded image. 
# Early detection can assist in taking timely interventions for better crop health management.
# """)


# st.sidebar.write("---")
# st.sidebar.write("Developed by: **Krishan Mittal and Himanshi Gupta**")
    