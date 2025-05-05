import streamlit as st
import numpy as np
from skimage import io as ios, transform
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pandas as pd
import io
import cv2 as cv
from imgaug import augmenters as iaa
from matplotlib.colors import Normalize
from matplotlib import rcParams

# Optional: Set a consistent and professional font style
rcParams.update({'font.size': 12, 'font.family': 'serif'})

# RGB to Luv conversion functions (unchanged)
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

def rgb_to_luv(rgb):
    reference_white_A = np.array([1.09850, 1.00000, 0.35585])
    xyz = rgb_to_xyz_illuminant_a(rgb)
    return xyz_to_luv(xyz, reference_white_A)

def open_image(image_file):
    image = ios.imread(image_file)
    
    # Handle grayscale images by converting them to RGB
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.shape[2] == 4:
        # Convert RGBA to RGB by discarding the alpha channel
        image = image[:, :, :3]
    elif image.shape[2] == 3:
        pass  # Image is already RGB
    else:
        raise ValueError(f"Unsupported number of channels: {image.shape[2]}")
    
    # Normalize image if not already in float format
    if image.dtype != np.float32 and image.dtype != np.float64:
        image = image.astype(np.float32) / 255.0
    return image

def create_mask(image, color, tolerance):
    lower_bound = np.clip(color - tolerance, 0, 255)
    upper_bound = np.clip(color + tolerance, 0, 255)
    mask = np.logical_and(image >= lower_bound, image <= upper_bound)
    return np.all(mask, axis=-1).astype(int)

def most_common_color(image):
    image = image.reshape(-1, 3)
    mask = ~((image == [255, 255, 255]).all(axis=1) | (image == [0, 0, 0]).all(axis=1))
    image = image[mask]

    if len(image) == 0:
        return (255, 255, 255)  # Default to white if no valid pixels

    hist, edges = np.histogramdd(image, bins=256, range=((0, 256), (0, 256), (0, 256)))

    weighted_mean = [np.average(edges[i][:-1], weights=np.sum(hist, axis=tuple(j for j in range(3) if j != i)))
                     for i in range(3)]

    return tuple(map(round, weighted_mean))

def distance(a, b):
    """
    Computes the element-wise signed distance between two NumPy arrays.
    If a < b, returns positive (b-a), otherwise returns negative (a-b).
    """
    return np.where(a < b, b - a, (a - b) * -1)

def pair_images(original_files, masked_files):
    paired = []
    min_len = min(len(original_files), len(masked_files))
    if len(original_files) != len(masked_files):
        st.warning(f"You have uploaded {len(original_files)} original images and {len(masked_files)} masked images. "
                   f"Only the first {min_len} pairs will be processed.")
    for i in range(min_len):
        paired.append((original_files[i], masked_files[i]))
    return paired

def augment_image(image):
    """
    Apply a series of augmentations to the image.
    """
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),  # horizontal flips
        iaa.Flipud(0.5),  # vertical flips
        iaa.Affine(rotate=(-100, 100)),  # random rotations
        iaa.MultiplyBrightness((0.8, 1.2))  # brightness adjustments
    ])
    return seq.augment_image(image)

def extract_features(masked_image, original_image):


    # make masked image size 50% of the real one 

    masked_image = cv.resize(masked_image, (0, 0), fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)

    

    # Resize images to original size if necessary
    if masked_image.shape[:2] != original_image.shape[:2]:
        original_image = transform.resize(original_image, masked_image.shape[:2],
                                          anti_aliasing=True,
                                          preserve_range=True).astype(masked_image.dtype)

    # Resize images as per existing logic
    
    # make 
    masked_image_resized = masked_image * 255
    original_image_resized = original_image * 255

    # Define mask colors
    monolayer_mask_color = np.array([0, 0, 254])
    bilayer_mask_color = np.array([254, 0, 0])
    trilayer_mask_color = np.array([0, 255, 3])
    fourlayer_mask_color = np.array([255, 156, 0])
    five_layer_mask_color = np.array([220, 1, 250])
    sixlayer_mask_color = np.array([255, 255, 0])
    not_segmented_color = np.array([255, 255, 255])

    tolerance = 10

    masks = {
        "1": create_mask(masked_image_resized, monolayer_mask_color, tolerance),
        "2": create_mask(masked_image_resized, bilayer_mask_color, tolerance),
        "3": create_mask(masked_image_resized, trilayer_mask_color, tolerance),
        "4": create_mask(masked_image_resized, fourlayer_mask_color, tolerance),
        "5": create_mask(masked_image_resized, five_layer_mask_color, tolerance),
        "6": create_mask(masked_image_resized, sixlayer_mask_color, tolerance),
        "7": create_mask(masked_image_resized, not_segmented_color, tolerance)
    }

    variables_b = [[], [], []]
    targets_b = []

    # Calculate substrate area
    substrate_mask = 1 - (masks["1"] + masks["2"] + masks["3"] + masks["4"] + masks["5"] + masks["6"] + masks["7"])
    substrate_area = original_image_resized * substrate_mask[:, :, np.newaxis]

    substrate_color = np.array(most_common_color(substrate_area))
    substrate_color_luv = rgb_to_luv(substrate_color[np.newaxis, np.newaxis, :] / 255)[0, 0]

    for key, mask in masks.items():
        if np.sum(mask) == 0 or key == "7":
            continue

        # Convert mask to uint8 to prevent dtype issues
        mask_uint8 = mask.astype(np.uint8)

        # Compute layer_area and ensure it's uint8
        layer_area = (original_image_resized * mask_uint8[:, :, np.newaxis]).astype(np.uint8)

        black_areas_mask = (layer_area[:, :, 0] == 0) & (layer_area[:, :, 1] == 0) & (layer_area[:, :, 2] == 0)
        layer_area[layer_area.sum(axis=2) == 0] = substrate_color
        layer_area_luv = rgb_to_luv(layer_area / 255)
        layer_area_xyz = rgb_to_xyz_illuminant_a(layer_area / 255)
        layer_area_luv = layer_area_luv[~black_areas_mask]
        layer_area_xyz = layer_area_xyz[~black_areas_mask]
        layer_area_xyz = layer_area_xyz.reshape(-1, 3)
        layer_area_luv = layer_area_luv.reshape(-1, 3)

        l_part = layer_area_luv[:, 0]
        u_part = layer_area_luv[:, 1]
        v_part = layer_area_luv[:, 2]

        delta_e_l = (l_part - substrate_color_luv[0])
        delta_e_u = distance(u_part, substrate_color_luv[1])
        delta_e_v = distance(v_part, substrate_color_luv[2])

        variables_b[0].extend(delta_e_l)
        variables_b[1].extend(delta_e_u)
        variables_b[2].extend(delta_e_v)

        # Assuming 'key' represents a categorical variable
        targets_b.extend([int(key)] * len(delta_e_l))

        # Data Augmentation for 5-layer data
        if key == "5":
            # Extract the layer area as an image and ensure it's uint8
            layer_image = layer_area.copy().astype(np.uint8)

            # Perform augmentation
            try:
                augmented_image = augment_image(layer_image)

                # Extract features from augmented image
                # Recompute mask on augmented image
                augmented_mask = create_mask(augmented_image, five_layer_mask_color, tolerance)
                if np.sum(augmented_mask) == 0:
                    continue  # Skip if no mask found after augmentation

                # Ensure augmented_layer_area is uint8
                augmented_layer_area = (augmented_image * augmented_mask[:, :, np.newaxis]).astype(np.uint8)
                augmented_layer_area_luv = rgb_to_luv(augmented_layer_area / 255)
                augmented_layer_area_luv = augmented_layer_area_luv.reshape(-1, 3)

                augmented_l_part = augmented_layer_area_luv[:, 0]
                augmented_u_part = augmented_layer_area_luv[:, 1]
                augmented_v_part = augmented_layer_area_luv[:, 2]

                augmented_delta_e_l = (augmented_l_part - substrate_color_luv[0])
                augmented_delta_e_u = distance(augmented_u_part, substrate_color_luv[1])
                augmented_delta_e_v = distance(augmented_v_part, substrate_color_luv[2])

                variables_b[0].extend(augmented_delta_e_l)
                variables_b[1].extend(augmented_delta_e_u)
                variables_b[2].extend(augmented_delta_e_v)

                targets_b.extend([int(key)] * len(augmented_delta_e_l))

            except Exception as e:
                st.warning(f"Augmentation failed for layer 5: {e}")
                continue

    variables_a = np.array(variables_b).T
    targets_a = np.array(targets_b)

    return variables_a, targets_a

def plot_confusion_matrix(conf_matrix, class_labels, accuracy, save_path=None):
    """
    Plots a normalized and annotated confusion matrix with counts and percentages.
    Dynamically adjusts text color for readability based on cell background.

    Args:
        conf_matrix (array): Confusion matrix.
        class_labels (list): List of class labels.
        accuracy (float): Overall accuracy.
        save_path (str, optional): Path to save the figure. Defaults to None.
    """
    # Normalize the confusion matrix by row (true labels)
    conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    conf_matrix_normalized = np.nan_to_num(conf_matrix_normalized)  # Replace NaNs with 0

    # Set up the matplotlib figure
    plt.figure(figsize=(10, 8))
    sns.set(style='whitegrid')

    # Create a heatmap without annotations
    ax = sns.heatmap(
        conf_matrix_normalized,
        annot=False,  # Disable default annotations
        cmap='Blues',
        cbar=True,
        linewidths=0.5,
        linecolor='gray',
        xticklabels=class_labels,
        yticklabels=class_labels
    )

    # Add custom annotations with count and percentage
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            count = conf_matrix[i, j]
            percentage = conf_matrix_normalized[i, j] * 100
            # Determine text color based on normalized value
            text_color = "white" if conf_matrix_normalized[i, j] > 0.5 else "black"
            annotation = f"{int(count)}\n{percentage:.1f}%"
            ax.text(j + 0.5, i + 0.5, annotation,
                    ha='center', va='center', color=text_color, fontsize=10)

    # Set labels, title and ticks
    ax.set_xlabel('Predicted Labels', fontsize=14, labelpad=10)
    ax.set_ylabel('True Labels', fontsize=14, labelpad=10)
    ax.set_title(f'Confusion Matrix (Accuracy: {accuracy:.2%})', fontsize=16, pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)

    # Improve layout
    plt.tight_layout()

    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return plt.gcf()

def main():
    st.title("Flake Analysis App - Classification")

    # User inputs
    material_name = st.text_input("Enter material name:")
    substrate_name = st.text_input("Enter substrate name:")

    st.markdown("""
    **Instructions:**
    - Upload original and masked images **in the same order**. For example:
      1. First original image
      2. First masked image
      3. Second original image
      4. Second masked image
      - and so on.
    """)

    # File uploader for masked images
    masked_image_files = st.file_uploader(
        "Upload masked images", type=["jpg", "png"], accept_multiple_files=True, key="masked"
    )

    # File uploader for original images
    original_image_files = st.file_uploader(
        "Upload original images", type=["jpg", "png"], accept_multiple_files=True, key="original"
    )

    # Add a button to trigger the processing
    if st.button("Process Images"):
        if masked_image_files and original_image_files and material_name and substrate_name:
            paired_images = pair_images(original_image_files, masked_image_files)
            
            if not paired_images:
                st.error("No image pairs found. Please upload at least one original and one masked image.")
                return

            all_variables = []
            all_targets = []

            with st.spinner("Processing images..."):
                for idx, (original_file, masked_file) in enumerate(paired_images):
                    st.write(f"Processing pair {idx + 1}...")
                    try:
                        # Read images
                        masked_image = open_image(masked_file)
                        original_image = open_image(original_file)

                        # Extract features
                        variables, targets = extract_features(masked_image, original_image)
                        all_variables.append(variables)
                        all_targets.append(targets)
                    except Exception as e:
                        st.error(f"Error processing pair {idx + 1}: {e}")
                        continue

            # Concatenate all features and targets
            if all_variables and all_targets:
                X_a = np.vstack(all_variables)
                y_a = np.hstack(all_targets)
            else:
                st.error("No features extracted from the uploaded images.")
                return

            # Create DataFrame
            df_a = pd.DataFrame({
                'delta_e_L': X_a[:,0],
                'delta_e_u': X_a[:,1],
                'delta_e_v': X_a[:,2],
                # Add more features if needed
            })

            # Define features and targets
            X = df_a[['delta_e_L', 'delta_e_u']]
            y = y_a

            st.success(f"Total samples collected: {len(y)}")
            st.write("Class distribution:")
            class_counts = pd.Series(y).value_counts().sort_index()
            st.bar_chart(class_counts)

            st.subheader("Model Training")

            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.20, random_state=42, shuffle=True, stratify=y
            )

            # Initialize the model
            rf_model = RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_split=4,
                min_samples_leaf=1,
                max_features="sqrt",
                bootstrap=True,
                random_state=42,
                n_jobs=-1
            )

            # Train the model
            rf_model.fit(X_train, y_train)

            # Save the model
            model_name = f"{material_name}-{substrate_name}.joblib"
            model_bytes = io.BytesIO()
            joblib.dump(rf_model, model_bytes)
            model_bytes.seek(0)

            # Predictions and metrics
            y_pred = rf_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            conf_matrix = confusion_matrix(y_test, y_pred)
            class_report = classification_report(y_test, y_pred, zero_division=0)
            class_labels = sorted(list(set(y)))

            report = {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1
            }

            # Display results
            st.subheader("Model Performance")

            # Download model button
            st.download_button(
                label="Download Model",
                data=model_bytes,
                file_name=model_name,
                mime="application/octet-stream"
            )

            # Display classification report
            st.write("Classification Report:")
            report_df = pd.DataFrame(list(report.items()), columns=['Metric', 'Value'])
            st.table(report_df)

            # Display enhanced confusion matrix
            st.write("Confusion Matrix:")
            fig_cm = plot_confusion_matrix(conf_matrix, class_labels, accuracy)
            st.pyplot(fig_cm)

            # Optionally, provide a download button for the confusion matrix
            buf = io.BytesIO()
            fig_cm.savefig(buf, format="png", dpi=300, bbox_inches='tight')
            buf.seek(0)
            st.download_button(
                label="Download Confusion Matrix",
                data=buf,
                file_name="confusion_matrix.png",
                mime="image/png"
            )
            plt.close(fig_cm)  # Close the figure to free memory

            # Display detailed classification report
            with st.expander("View Detailed Classification Report"):
                st.text(class_report)

            # Feature Importances
            st.subheader("Feature Importances")
            feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
            fig_fi, ax_fi = plt.subplots(figsize=(8, 6))
            sns.barplot(x=feature_importances, y=feature_importances.index, ax=ax_fi, palette='viridis')
            ax_fi.set_title('Feature Importances', fontsize=16)
            ax_fi.set_xlabel('Importance Score', fontsize=14)
            ax_fi.set_ylabel('Features', fontsize=14)
            plt.tight_layout()
            st.pyplot(fig_fi)
            plt.close(fig_fi)

        else:
            st.error("Please upload both sets of images and enter material and substrate names before processing.")

if __name__ == "__main__":
    main()
