import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
import joblib
import pandas as pd
from PIL import Image

def open_image(image_file):
    # Open the image using PIL
    image = Image.open(image_file)
    # Convert PIL image to numpy array
    image_np = np.array(image)
    # If image is not RGB (e.g., it's RGBA), convert it to RGB
    if image_np.shape[-1] == 4:
        image_np = image_np[:, :, :3]

    # resize the image to 25% of its original size
    image_np = cv2.resize(image_np, (0, 0), fx=0.50, fy=0.50, interpolation=cv2.INTER_AREA)
    return image_np


def create_mask(image, color, tolerance):
    lower_bound = color - tolerance
    upper_bound = color + tolerance
    mask = np.logical_and(image >= lower_bound, image <= upper_bound)
    return np.all(mask, axis=-1).astype(int)

def most_common_color(image):
    image = image.reshape(-1, 3)
    mask = ~((image == [255, 255, 255]).all(axis=1) | (image == [0, 0, 0]).all(axis=1))
    image = image[mask]

    hist, edges = np.histogramdd(image, bins=256, range=((0, 256), (0, 256), (0, 256)))

    weighted_mean = [np.average(edges[i][:-1], weights=np.sum(hist, axis=tuple(j for j in range(3) if j != i)))
                     for i in range(3)]

    return tuple(map(round, weighted_mean))

def rgb_to_luv(rgb):
    if rgb.ndim == 2:
        rgb = rgb.reshape(1, -1, 3)

    reference_white_A = np.array([1.09850, 1.00000, 0.35585])

    xyz = rgb_to_xyz_illuminant_a(rgb)
    luv = xyz_to_luv(xyz, reference_white_A)

    if rgb.shape[0] == 1:
        luv = luv.reshape(-1, 3)

    return luv

def linearize_rgb(rgb):
    rgb = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return rgb

def rgb_to_xyz_illuminant_a(rgb):
    rgb_linear = linearize_rgb(rgb)
    matrix = np.array([
        [0.4002, 0.7075, -0.0808],
        [-0.2263, 1.1653, 0.0457],
        [0.0000, 0.0000, 0.8253]
    ])
    return np.dot(rgb_linear, matrix.T)

def xyz_to_luv(xyz, reference_white):
    xyz = np.maximum(xyz, 1e-6)
    Xr, Yr, Zr = reference_white
    L = np.where(xyz[:,:,1]/Yr > 216/24389,
                 116 * (xyz[:,:,1]/Yr)**(1/3) - 16,
                 24389/27 * xyz[:,:,1]/Yr)

    d = xyz[:,:,0] + 15 * xyz[:,:,1] + 3 * xyz[:,:,2]
    ur_prime = 4 * Xr / (Xr + 15 * Yr + 3 * Zr)
    vr_prime = 9 * Yr / (Xr + 15 * Yr + 3 * Zr)

    u_prime = 4 * xyz[:,:,0] / d
    v_prime = 9 * xyz[:,:,1] / d

    u = 13 * L * (u_prime - ur_prime)
    v = 13 * L * (v_prime - vr_prime)

    return np.dstack((L, u, v))


def process_images(masked_image, original_image, model):
    # Ensure images are the same size
    if masked_image.shape != original_image.shape:
        original_image = cv2.resize(original_image, (masked_image.shape[1], masked_image.shape[0]))

    # Create mask
    maskArea = create_mask(masked_image, np.array([255, 255, 255]), 10)
    masked_area = original_image[maskArea == 1]
    non_masked_area = original_image[maskArea == 0]

    non_masked_area_color = np.array(most_common_color(non_masked_area))

    # Convert to Luv color space
    masked_area_luv = rgb_to_luv(masked_area / 255)
    non_masked_area_rgb = np.array(non_masked_area_color, dtype=np.float32) / 255.0
    non_masked_area_luv = rgb_to_luv(non_masked_area_rgb.reshape(1, 1, 3))[0]

    # Create Luv image
    luv_image = np.zeros_like(original_image, dtype=np.float32)
    luv_image[maskArea == 1] = masked_area_luv

    L_component = luv_image[..., 0]
    u_component = luv_image[..., 1]
    v_component = luv_image[..., 2]

    # Calculate differences
    L_diff = (L_component - non_masked_area_luv[0])
    maskL = L_diff <= 0
    L_diff[L_diff <= 0] = 0

    u_diff = u_component - non_masked_area_luv[1]
    v_diff = v_component - non_masked_area_luv[2]

    u_diff[maskL] = 0
    v_diff[maskL] = 0

    # Prepare data for prediction
    df_pred = pd.DataFrame({
        'delta_e_L': L_diff.flatten(),
        'delta_e_u': u_diff.flatten(),
        'delta_e_v': v_diff.flatten()
    })

    df_pred['delta_e_L_squared'] = df_pred['delta_e_L'] ** 2
    df_pred['delta_e_u_squared'] = df_pred['delta_e_u'] ** 2
    df_pred['delta_e_v_squared'] = df_pred['delta_e_v'] ** 2

    X_pred = df_pred[['delta_e_L', 'delta_e_L_squared', 'delta_e_u', 'delta_e_u_squared', 'delta_e_v', 'delta_e_v_squared']]

    # Make predictions
    y_pred = model.predict(X_pred)
    y_pred = y_pred.reshape(original_image.shape[0], original_image.shape[1])
    y_pred[maskL] = 0

    return y_pred

def main():
    st.set_page_config(page_title="2D Material Layer Detection", layout="wide")

    st.title('2D Material Layer Detection')
    st.write("Upload your images and model file to detect 2D material layers.")

    # Sidebar for file uploads and processing
    with st.sidebar:
        st.header("Input Files")
        masked_image_file = st.file_uploader("Upload Masked Image", type=['png', 'jpg', 'jpeg', 'bmp'])
        original_image_file = st.file_uploader("Upload Original Image", type=['png', 'jpg', 'jpeg', 'bmp'])
        model_file = st.file_uploader("Upload Model File", type=['joblib'])

        process_button = st.button('Process Images', disabled=not (masked_image_file and original_image_file and model_file))

    # Main content area
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Masked Image")
        if masked_image_file:
            masked_image = open_image(masked_image_file)
            st.image(masked_image, caption="Uploaded Masked Image", use_column_width=True)
        else:
            st.info("Please upload a masked image.")

    with col2:
        st.subheader("Original Image")
        if original_image_file:
            original_image = open_image(original_image_file)
            st.image(original_image, caption="Uploaded Original Image", use_column_width=True)
        else:
            st.info("Please upload an original image.")

    if model_file:
        st.sidebar.success("Model file uploaded successfully.")
    else:
        st.sidebar.info("Please upload a model file.")

    # Results area
    st.header("Results")
    if process_button:
        if masked_image_file and original_image_file and model_file:
            try:
                with st.spinner('Processing images...'):
                    masked_image = open_image(masked_image_file)
                    original_image = open_image(original_image_file)
                    model = joblib.load(model_file)
                    result = process_images(masked_image, original_image, model)

                fig, ax = plt.subplots(figsize=(12, 8))
                im = ax.imshow(result, cmap='turbo')
                ax.set_title('2D Material Layer Detection Result')
                ax.axis('off')
                cbar = fig.colorbar(im, ax=ax)
                cbar.set_label('Prediction Value')

                st.pyplot(fig)

                st.success("Image processing completed successfully!")
            except Exception as e:
                st.error(f"An error occurred during processing: {str(e)}")
        else:
            st.error("Please ensure all required files are uploaded before processing.")

    # Instructions or additional information
    with st.expander("How to use this app"):
        st.write("""
        1. Upload a masked image of your 2D material sample.
        2. Upload the corresponding original image.
        3. Upload your trained model file (.joblib format).
        4. Click 'Process Images' to start the detection.
        5. The result will be displayed as a color-coded image showing the detected layers.
        """)

if __name__ == '__main__':
    main()
