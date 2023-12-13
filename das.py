import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

# Function to perform semantic segmentation
def predict_semantic_segmentation(image, model):
    # Preprocess the input image
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    input_image = transform(image).unsqueeze(0)

    # Perform prediction
    with torch.no_grad():
        prediction = model(input_image)

    # Process the prediction to get segmentation mask
    segmentation_mask = torch.argmax(prediction, dim=1).squeeze().numpy()

    return segmentation_mask

# Streamlit app
def main():
    st.title('Aplikasi Segmentasi Semantik DAS')

    # Upload image through Streamlit
    uploaded_file = st.file_uploader("Pilih gambar DAS", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Citra DAS', use_column_width=True)

        # Make prediction on the uploaded image
        if st.button('Segmentasi'):
            # Load the pre-trained model
            das_model = torch.load('Unet-Mobilenet.pt', map_location=torch.device('cpu'))
            das_model.eval()

            # Perform semantic segmentation prediction
            segmentation_result = predict_semantic_segmentation(image, das_model)

            # Display the segmented image
            st.image(segmentation_result, caption='Hasil Segmentasi', use_column_width=True)

if __name__ == '__main__':
    main()
