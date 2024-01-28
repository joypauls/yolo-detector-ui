import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import base64
import torch

# Create application title and file uploader widget.
st.title("Object Detection")
image_buffer = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Model
model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)

# Images
imgs = ["https://ultralytics.com/images/zidane.jpg"]  # batch of images

# Inference
results = model(imgs)

# Results
results.print()
