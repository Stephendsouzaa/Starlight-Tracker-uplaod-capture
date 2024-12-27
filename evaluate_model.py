import argparse
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import accuracy_score

def load_test_data(test_data_path, img_size=(224, 224)):
    """
    Loads and preprocesses test data from the specified directory.

    Args:
        test_data_path (str): Path to the test data directory.
        img_size (tuple): Target size for image resizing.

    Returns:
        tuple: Arrays of images and corresponding labels.
    """
    images = []
    labels = []
    
    # Define the class labels
    class_labels = [
        "Andromeda", "Aquila", "Auriga", "CanisMajor", "Capricornus",
        "Cetus", "Columba", "Gemini", "Grus", "Leo",
        "Orion", "Pavo", "Pegasus", "Phoenix", "Pisces",
        "PiscisAustrinus", "Puppis", "UrsaMajor", "UrsaMinor", "Vela"
    ]

    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"The directory {test_data_path} does not exist.")

    for filename in os.listdir(test_data_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):  # Check for image files
            try:
                # Load and preprocess the image
                img_path = os.path.join(test_data_path, filename)
                img = load_img(img_path, target_size=img_size)
                img_array = img_to_array(img) / 255.0  # Normalize pixel values
                images.append(img_array)
                
                # Extract the label from the filename (assumes filename format matches class names)
                label_name = filename.split('.')[0]  # Remove file extension
                if label_name in class_labels:
                    labels.append(class_labels.index(label_name))
                else:
                    print(f"Warning: '{label_name}' is not in class labels. Skipping this image.")
            except Exception as e:
                print(f"Error processing file {filename}: {e}")

    if not images or not labels:
        raise ValueError("No valid images or labels found in the specified directory.")

    return np.array(images), np.array(labels)

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model on the test dataset.

    Args:
        model: Loaded Keras model.
        X_test (np.array): Test images.
        y_test (np.array): True labels for the test images.
    """
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)

    # Calculate and print accuracy
    accuracy = accuracy_score(y_test, predicted_classes)
    print(f'Accuracy: {accuracy:.2f}')
    print(f"Predicted classes: {predicted_classes}")
    print(f"True classes: {y_test}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained Keras model on test data.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model file')
    parser.add_argument('--test_data_path', type=str, required=True, help='Path to the test data directory')
    args = parser.parse_args()

    # Load the trained Keras model
    print(f"Loading model from: {args.model_path}")
    model = load_model(args.model_path)

    # Load test data
    print(f"Loading test data from: {args.test_data_path}")
    X_test, y_test = load_test_data(args.test_data_path)

    # Evaluate the model
    print("Evaluating the model...")
    evaluate_model(model, X_test, y_test)
