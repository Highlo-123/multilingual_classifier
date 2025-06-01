from sklearn import svm
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from skimage import exposure
import cv2
import numpy as np
import os

def classify_images_using_HOG_SVM(image_folder, label_map):
    """
    Classifies images using HOG features and an SVM classifier.

    Parameters:
    - image_folder (str): The path to the folder containing subfolders of images for each class.
    - label_map (dict): A dictionary that maps subfolder names to class labels.

    Returns:
    - clf (svm.SVC): The trained SVM classifier.
    - X_test (array): The feature vector for the test set.
    - y_test (array): The actual labels for the test set.
    - test_accuracy (float): The accuracy on the test set.
    """
    
    def extract_hog_features(image):
        """
        Extracts HOG features from a given image.

        Parameters:
        - image (numpy.ndarray): The input image to extract features from.

        Returns:
        - features (numpy.ndarray): The extracted HOG features.
        """
        # Resize image to a consistent size for feature extraction
        resized_img = cv2.resize(image, (64, 128))  # Resize to 64x128 for consistency
        
        # Convert to grayscale (required for HOG feature extraction)
        gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

        # Compute HOG features (using 8x8 pixels per cell and 2x2 cells per block)
        features, hog_image = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, multichannel=False)

        # Optionally, apply a contrast normalization technique to enhance features
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

        return features
    
    def load_data(image_folder, label_map):
        """
        Loads images from subfolders, extracts their HOG features and labels.

        Parameters:
        - image_folder (str): Path to the dataset folder containing subfolders of images.
        - label_map (dict): Mapping from folder names to class labels.

        Returns:
        - features (numpy.ndarray): Array containing the HOG feature vectors.
        - labels (numpy.ndarray): Array containing the corresponding labels.
        """
        features = []
        labels = []

        # Loop through the subfolders in the image folder
        for folder_name in os.listdir(image_folder):
            folder_path = os.path.join(image_folder, folder_name)

            # Only process directories (each representing a class)
            if os.path.isdir(folder_path):
                label = label_map[folder_name]  # Map folder name to class label
                
                # Loop through the images in the class folder
                for filename in os.listdir(folder_path):
                    image_path = os.path.join(folder_path, filename)
                    
                    if image_path.endswith(('.png', '.jpg', '.jpeg')):  # Process only image files
                        image = cv2.imread(image_path)  # Read the image
                        feature = extract_hog_features(image)  # Extract HOG features
                        features.append(feature)  # Append feature vector
                        labels.append(label)  # Append corresponding label
        
        return np.array(features), np.array(labels)

    # Load the dataset and extract features and labels
    X, y = load_data(image_folder, label_map)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train the SVM classifier (RBF kernel)
    clf = svm.SVC(kernel='rbf', C=1, gamma='scale')  # SVM with RBF kernel
    clf.fit(X_train, y_train)  # Fit the classifier on the training data

    
  

    return clf, X_test, y_test

