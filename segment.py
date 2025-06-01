import cv2
import numpy as np

def segment_words(image_path):
    # Load the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Threshold the image to binary
    _, binary_img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)

    # Perform noise removal using morphological operations
    kernel = np.ones((5, 5), np.uint8)
    morph_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)

    # Find contours of the words
    contours, _ = cv2.findContours(morph_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours from left to right based on the x-coordinate of the bounding box
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

    # Segment each word based on the contours
    segmented_words = []
    for ctr in contours:
        # Get bounding box of the contour
        x, y, w, h = cv2.boundingRect(ctr)

        # Skip small contours that might be noise
        if w < 10 or h < 10:
            continue
        
        # Extract the word from the image
        word_img = img[y:y+h, x:x+w]
        segmented_words.append(word_img)

        # Optionally, you can draw bounding boxes on the original image to visualize
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Show the image with bounding boxes
    cv2.imshow('Segmented Words', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return segmented_words


