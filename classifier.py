import os
import sys
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from joblib import dump, load
import seaborn as sns
import matplotlib.pyplot as plt



def load_images_from_folder(folder, width, height):
    """
    Load images and corresponding labels from a folder structure where each subdirectory represents a class.

    Parameters:
        folder (str): Path to the root folder containing subdirectories for each class.
        width (int): Width of the images to be loaded.
        height (int): Height of the images to be loaded.
    Returns:
        numpy.ndarray: Array of images loaded from the folder.
        numpy.ndarray: Array of labels corresponding to the images.

    """
    images = []
    labels = []
    class_folders = sorted(os.listdir(folder))  # List all subdirectories (classes)
    for class_label, class_name in enumerate(class_folders):
        class_path = os.path.join(folder, class_name)
        if os.path.isdir(class_path):
            for filename in os.listdir(class_path):
                img_path = os.path.join(class_path, filename)
                if img_path.endswith(".jpg") or img_path.endswith(".jpeg"):
                    image = cv2.imread(img_path)
                    if image is not None:
                        # Resize or preprocess image if needed
                        resized_image = cv2.resize(image, (width, height))
                        images.append(resized_image)
                        labels.append(class_label)
    return np.array(images), np.array(labels)


def svm_classifier(X_train, X_test):
    """
    Use SVM classifier to predict labels for test data.
    
    Args:
        X_train (array): part of the dataset used for training
        X_test (array): part of the dataset used for testing

    Returns:
        y_pred (ndarray): predicted labels for the test data
    """
    # Flatten the images for SVM classifier
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    print('Starting training SVM classifier...')

    # Create and train the SVM classifier
    svm_classifier = SVC(kernel='linear', verbose=True)
    svm_classifier.fit(X_train_flat, y_train)
    print('Training completed.')
    
    # Save the trained model to a file
    print('Saving the trained model to a file...')
    dump(svm_classifier, 'svm_classifier_model.pkl')
    print('Model saved successfully.')
    
    # Predict labels for test data
    print('Predicting labels for test data...')
    y_pred = svm_classifier.predict(X_test_flat)
    print('Prediction completed.')
    return y_pred


def mlp_classifier(X_train, X_test):
    """
    Use MLP classifier to predict labels for test data.
    
    Args:
        X_train (array): part of the dataset used for training
        X_test (array): part of the dataset used for testing

    Returns:
        y_pred (ndarray): predicted labels for the test data
    """
    # Flatten the images for MLP classifier
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    # Create and train the MLP classifier
    print('Starting training MLP classifier...')
    mlp_classifier = MLPClassifier(activation='relu', solver='adam', learning_rate= 'adaptive', learning_rate_init=0.0003, hidden_layer_sizes=(512,512, 256, 128), tol=1e-6, max_iter=1000, verbose = True)
    mlp_classifier.fit(X_train_flat, y_train)
    print('Training completed.')
    
    # Save the trained model to a file
    print('Saving the trained model to a file...')
    dump(mlp_classifier, 'mlp_classifier_model'+'.pkl')
    print('Model saved successfully.')

    # Predict labels for test data
    print('Predicting labels for test data...')
    y_pred = mlp_classifier.predict(X_test_flat)
    print('Prediction completed.')
    return y_pred




if __name__ == '__main__':
    """
    Main function to load images, train classifiers, and evaluate the performance.
    """
    
    # Check if the correct number of arguments is provided
    sys.argv = sys.argv[1:]
    if len(sys.argv) != 3:
        print(len(sys.argv))
        print("Usage: python classifier.py <classifier> <image width> <image height>")
        print("classifier: mlp or svm")
        print("image width: width of the image")
        print("image height: height of the image")
        sys.exit(1)
    
    folder = os.getcwd()
    print(sys.argv)
    
    width = int(sys.argv[1])
    height = int(sys.argv[2])
    
    images_path = os.path.join(folder, "images"+ str(width)+ str(height)+ ".npy")
    labels_path = os.path.join(folder, "labels"+ str(width)+ str(height)+ ".npy")
    
    if not os.path.exists(images_path) and not os.path.exists(labels_path):
        # Create images and labels from the folder structure
        images, labels = load_images_from_folder(folder, width, height)
        np.save(images_path, images)
        np.save(labels_path, labels)
        print("Images and labels saved successfully.")
        print("Images shape:", images.shape)
        print("Labels shape:", labels.shape)
    
    
    
    # Get the classes of the problem, which are the subdirectories of the folder
    list_folder = sorted(os.listdir(folder))
    class_folders = []
    for class_label, class_name in enumerate(class_folders):
        class_path = os.path.join(folder, class_name)
        if os.path.isdir(class_path):
            class_folders.append(class_name)

    # Load images and labels from the saved files
    images = np.load(images_path)
    labels = np.load(labels_path)

    print("Images and labels loaded successfully.")
    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)


    if sys.argv[0] == 'mlp':
        y_pred = mlp_classifier(X_train, X_test)
    elif sys.argv[0] == 'svm':
        y_pred = svm_classifier(X_train, X_test)

    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_folders, yticklabels=class_folders)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix.png")


    # Create a figure to aggregate the predictions
    fig, axes = plt.subplots(nrows=5, ncols=10, figsize=(20, 10))

    # Initialize counter for subplot indices
    row_index = 0
    col_index = 0
    

    # Iterate through test set
    for i, (image, true_label, pred_label) in enumerate(zip(X_test, y_test, y_pred)):
        if true_label == pred_label:
            # Plot the corrected predicted images
            axes[row_index, col_index].imshow(image)
            axes[row_index, col_index].axis('off')
            axes[row_index, col_index].set_title(f"True: {list_folder[true_label]}\nPredicted: {list_folder[pred_label]}")
            
            # Update subplot indices
            col_index += 1
            if col_index == 10:
                row_index += 1
                col_index = 0
                
            # Break if all subplots are filled
            if row_index == 5:
                break

    # Remove empty subplots
    for i in range(row_index, 5):
        for j in range(col_index, 10):
            fig.delaxes(axes[i, j])
    
    # Save the aggregated correct predictions image
    plt.tight_layout()
    plt.savefig("correct_predictions.png")
    
    fig1, axes1 = plt.subplots(nrows=4, ncols=10, figsize=(20, 10))
    row_index1 = 0
    col_index1 = 0
    # Plot the misclassified images
    for i, (image, true_label, pred_label) in enumerate(zip(X_test, y_test, y_pred)):
        if true_label != pred_label:
            axes1[row_index1, col_index1].imshow(image)
            axes1[row_index1, col_index1].axis('off')
            axes1[row_index1, col_index1].set_title(f"True: {list_folder[true_label]}\nPredicted: {list_folder[pred_label]}")
            col_index1 += 1
            if col_index1 == 10:
                row_index1 += 1
                col_index1 = 0
            if row_index1 == 4:
                break

    # Remove empty subplots
    for i in range(row_index1, 4):
        for j in range(col_index1, 10):
            fig1.delaxes(axes1[i, j])
    
    plt.tight_layout()
    plt.savefig("misclassified_predictions.png")
