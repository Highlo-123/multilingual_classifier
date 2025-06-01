import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure


def load_image(path):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Image not found at: {path}")
    return image


def resize_image(image, size=(64, 128)):
    return cv2.resize(image, size)


def to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def remove_noise(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    median = cv2.medianBlur(blurred, 3)
    return median


def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)


def adaptive_threshold(image):
    return cv2.adaptiveThreshold(
        image, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )


def morphological_ops(image):
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=1)
    dilated = cv2.dilate(opening, kernel, iterations=1)
    return dilated


def canny_edges(image):
    return cv2.Canny(image, threshold1=50, threshold2=150)


def deskew_image(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = image.shape
    center = (w // 2, h // 2)
    
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated


def extract_hog_features(image):
    features, hog_image = hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=True,
        transform_sqrt=True
    )
    hog_image = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    return features, hog_image


def plot_intermediate_steps(images, titles):
    plt.figure(figsize=(16, 10))
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(2, 4, i+1)
        cmap = 'gray' if len(img.shape) == 2 else None
        plt.imshow(img, cmap=cmap)
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def preprocess_image_pipeline(image_path):
    original = load_image(image_path)
    resized = resize_image(original)
    gray = to_grayscale(resized)
    denoised = remove_noise(gray)
    enhanced = apply_clahe(denoised)
    thresholded = adaptive_threshold(enhanced)
    morphed = morphological_ops(thresholded)
    edges = canny_edges(morphed)
    deskewed = deskew_image(morphed)
    hog_features, hog_vis = extract_hog_features(deskewed)

    # Optional: plot intermediate steps
    plot_intermediate_steps(
        [original, gray, denoised, enhanced, thresholded, morphed, deskewed, hog_vis],
        ['Original', 'Grayscale', 'Denoised', 'CLAHE', 'Thresholded', 'Morphed', 'Deskewed', 'HOG']
    )

    return hog_features, deskewed


if __name__ == "__main__":
    image_path = "sample_input.jpg"  # Change this path
    features, final_image = preprocess_image_pipeline(image_path)

    print(f"✅ HOG feature vector shape: {features.shape}")
