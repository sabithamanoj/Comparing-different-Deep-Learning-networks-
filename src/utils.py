import os
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, cohen_kappa_score, confusion_matrix, precision_score, recall_score, RocCurveDisplay
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import random
import logging
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def generate_data_paths_with_label(data_directory):
    filepaths = []
    labels = []

    folders = os.listdir(data_directory)
    for folder in folders:
        folder_path = os.path.join(data_directory, folder)
        filelist = os.listdir(folder_path)
        for file in filelist:
            if 'mask' not in file:
                fpath = os.path.join(folder_path, file)
                filepaths.append(fpath)
                labels.append(folder)

    # Concatenate data paths with labels into one dataframe
    Fseries = pd.Series(filepaths, name='filepaths')
    Lseries = pd.Series(labels, name='labels')
    df = pd.concat([Fseries, Lseries], axis=1)
    return df

# Reference : https://www.kaggle.com/code/aditimondal23/vgg19-breast
# Function to evaluate the model
def evaluation(model, x_train, y_train, x_val, y_val, x_test, y_test, history):
    train_loss, train_acc = model.evaluate(x_train, y_train.toarray())
    val_loss, val_acc = model.evaluate(x_val, y_val.toarray())
    test_loss_value, test_accuracy = model.evaluate(x_test, y_test.toarray())

    y_pred = model.predict(x_test)
    y_pred_label = np.argmax(y_pred, axis=1)
    y_true_label = np.argmax(y_test.toarray(), axis=1)

    f1_measure = f1_score(y_true_label, y_pred_label, average='weighted')
    roc_score = roc_auc_score(y_test.toarray(), y_pred)
    kappa_score = cohen_kappa_score(y_true_label, y_pred_label)
    precision = precision_score(y_true_label, y_pred_label, average='weighted')
    recall = recall_score(y_true_label, y_pred_label, average='weighted')
    cm = confusion_matrix(y_true_label, y_pred_label)

    logging.info("\n--- Model Evaluation Metrics ---")
    logging.info(f"Train accuracy: {train_acc:.4f}")
    logging.info(f"Validation accuracy: {val_acc:.4f}")
    logging.info(f"Test accuracy: {test_accuracy:.4f}")
    logging.info(f"F1 Score: {f1_measure:.4f}")
    logging.info(f"Kappa Score: {kappa_score:.4f}")
    logging.info(f"ROC AUC Score: {roc_score:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    logging.info(f"Confusion Matrix")
    logging.info(cm)

    return y_true_label, y_pred, cm

# Plotting function for model performance metrics with improved alignment and spacing
def Plotting(encoder, acc, val_acc, loss, val_loss, y_true, y_pred, cm):
    # Plot accuracy and loss
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle("Model's Metrics Visualization", fontsize=16)

    # Accuracy Plot
    ax1.plot(range(1, len(acc) + 1), acc, label='Training Accuracy', color='blue', linestyle='-')
    ax1.plot(range(1, len(val_acc) + 1), val_acc, label='Validation Accuracy', color='orange', linestyle='--')
    ax1.set_title('History of Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    # Loss Plot
    ax2.plot(range(1, len(loss) + 1), loss, label='Training Loss', color='red', linestyle='-')
    ax2.plot(range(1, len(val_loss) + 1), val_loss, label='Validation Loss', color='green', linestyle='--')
    ax2.set_title('History of Loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    plt.subplots_adjust(wspace=0.3, hspace=0.5)  # Adjust space between accuracy and loss plots
    plt.show()
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=encoder.categories_[0])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.grid(False)  # Remove grid for clarity
    plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)  # Adjust spacing for clarity
    plt.show()

    # Plot ROC Curve for each class
    plt.figure(figsize=(10, 8))
    for i in range(len(encoder.categories_[0])):
        fpr, tpr, _ = roc_curve(y_true, y_pred[:, i], pos_label=i)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'ROC curve for {encoder.categories_[0][i]} (area = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

    # Calculate precision, recall, and F1 score for each class
    precision = precision_score(y_true, y_pred.argmax(axis=1), average=None)
    recall = recall_score(y_true, y_pred.argmax(axis=1), average=None)
    f1 = f1_score(y_true, y_pred.argmax(axis=1), average=None)

    # Bar plot for precision, recall, and F1 score with improved spacing
    metrics = [precision, recall, f1]
    metrics_names = ['Precision', 'Recall', 'F1 Score']
    class_labels = encoder.categories_[0]

    x = np.arange(len(class_labels))  # Label locations
    width = 0.2  # Width of the bars
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, metric, width, label=metrics_names[i])

    ax.set_xlabel('Classes')
    ax.set_ylabel('Scores')
    ax.set_title('Precision, Recall, and F1 Score for Each Class')
    ax.set_xticks(x + width)
    ax.set_xticklabels(class_labels)
    ax.legend()
    plt.grid(axis='y')  # Add gridlines for readability
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)  # Adjust space around bar plots
    plt.show()

# Fit and evaluate the model, and visualize the performance
def fit_evaluate(encoder, model, x_train, y_train, x_test, y_test, bs, Epochs, patience):
    # Early stopping to prevent overfitting
    es = EarlyStopping(monitor='val_loss', mode='min', patience=patience, restore_best_weights=True, verbose=1)
    # Model checkpoint to save the best model based on validation accuracy
    mc = ModelCheckpoint('best_model.keras', monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)

    # Split training data further into train and validation sets
    x1_train, x_val, y1_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42,
                                                        stratify=y_train.toarray())
    # Fit the model
    history = model.fit(x1_train, y1_train.toarray(),
                        validation_data=(x_val, y_val.toarray()),
                        epochs=Epochs,
                        batch_size=bs,
                        callbacks=[es, mc])

    # Retrieve history for training and validation metrics
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    # Evaluate the model and collect true/predicted labels
    y_true, y_pred, cm = evaluation(model, x1_train, y1_train, x_val, y_val, x_test, y_test, history)

    # Plot training history, ROC curves, and class-specific precision, recall, and F1 score
    Plotting(encoder, acc, val_acc, loss, val_loss, y_true, y_pred, cm)

# Fit and evaluate the model, and visualize the performance
def fit_evaluate_explainability(encoder, model, x_train, y_train, x_test, y_test, bs, Epochs, patience):
    # Early stopping to prevent overfitting
    es = EarlyStopping(monitor='val_loss', mode='min', patience=patience, restore_best_weights=True, verbose=1)
    # Model checkpoint to save the best model based on validation accuracy
    mc = ModelCheckpoint('best_model.keras', monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)

    # Split training data further into train and validation sets
    x1_train, x_val, y1_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42,
                                                        stratify=y_train.toarray())
    # Fit the model
    history = model.fit(x1_train, y1_train.toarray(),
                        validation_data=(x_val, y_val.toarray()),
                        epochs=Epochs,
                        batch_size=bs,
                        callbacks=[es, mc])

    # Retrieve history for training and validation metrics
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    # Evaluate the model and collect true/predicted labels
    y_true, y_pred, cm = evaluation(model, x1_train, y1_train, x_val, y_val, x_test, y_test, history)

    # Plot training history, ROC curves, and class-specific precision, recall, and F1 score
    Plotting(encoder, acc, val_acc, loss, val_loss, y_true, y_pred, cm)
    return model

# Function to visualize multiple predictions from the dataset with added comments
def visualize_model_performance(model, x_test, y_test, class_labels=['benign', 'normal', 'malignant'], num_samples=12):
    # Select random samples from the test set
    indices = random.sample(range(len(x_test)), num_samples)
    test_images = x_test[indices]
    true_labels = y_test.toarray()[indices]  # Convert sparse matrix to array if needed
    true_labels = np.argmax(true_labels, axis=1)  # Get true class indices

    # Make predictions on the selected test samples
    predictions = model.predict(test_images)
    predicted_labels = np.argmax(predictions, axis=1)  # Get predicted class indices
    # Set up the plotting layout for 4 images per row
    plt.figure(figsize=(18, 12))
    rows = num_samples // 4 + int(num_samples % 4 != 0)  # Calculate the required number of rows

    for i, (img, true_label, pred_label, pred_prob) in enumerate(
            zip(test_images, true_labels, predicted_labels, predictions)):
        plt.subplot(rows, 4, i + 1)  # Arrange images in 4 columns
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"True: {class_labels[true_label]}\nPred: {class_labels[pred_label]} ({max(pred_prob) * 100:.2f}%)",
                  fontsize=12, color="blue" if true_label == pred_label else "red")  # Color title based on correctness

    plt.subplots_adjust(wspace=0.4, hspace=0.6)  # Adjust spacing between plots
    plt.suptitle("Model Predictions on Test Images", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# Fit and evaluate the model, and visualize the performance
def fit_evaluate(encoder, model, x_train, y_train, x_test, y_test, bs, Epochs, patience):
    # Early stopping to prevent overfitting
    es = EarlyStopping(monitor='val_loss', mode='min', patience=patience, restore_best_weights=True, verbose=1)
    # Model checkpoint to save the best model based on validation accuracy
    mc = ModelCheckpoint('best_model.keras', monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)

    # Split training data further into train and validation sets
    x1_train, x_val, y1_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42,
                                                        stratify=y_train.toarray())
    # Fit the model
    history = model.fit(x1_train, y1_train.toarray(),
                        validation_data=(x_val, y_val.toarray()),
                        epochs=Epochs,
                        batch_size=bs,
                        callbacks=[es, mc])

    # Retrieve history for training and validation metrics
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    # Evaluate the model and collect true/predicted labels
    y_true, y_pred, cm = evaluation(model, x1_train, y1_train, x_val, y_val, x_test, y_test, history)

    # Plot training history, ROC curves, and class-specific precision, recall, and F1 score
    Plotting(encoder, acc, val_acc, loss, val_loss, y_true, y_pred, cm)

def fit_evaluate_with_augmentation(encoder, model, x_train, y_train, x_test, y_test, bs, Epochs, patience):
    # Early stopping to prevent overfitting
    es = EarlyStopping(monitor='val_loss', mode='min', patience=patience, restore_best_weights=True, verbose=1)
    # Model checkpoint to save the best model based on validation accuracy
    mc = ModelCheckpoint('best_model.keras', monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)

    # Split training data further into train and validation sets
    x1_train, x_val, y1_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42,
                                                        stratify=y_train.toarray())
    # Data augmentation using datagen
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    # Create a generator for the training data
    train_generator = datagen.flow(x1_train, y1_train.toarray(), batch_size=bs)

    # Fit the model
    history = model.fit(train_generator,
                        validation_data=(x_val, y_val.toarray()),
                        epochs=Epochs,
                        batch_size=bs,
                        callbacks=[es, mc])

    # Retrieve history for training and validation metrics
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    # Evaluate the model and collect true/predicted labels
    y_true, y_pred, cm = evaluation(model, x1_train, y1_train, x_val, y_val, x_test, y_test, history)

    # Plot training history, ROC curves, and class-specific precision, recall, and F1 score
    Plotting(encoder, acc, val_acc, loss, val_loss, y_true, y_pred, cm)



# Function to visualize multiple predictions from the dataset with added comments
def visualize_model_performance(model, x_test, y_test, class_labels=['benign', 'normal', 'malignant'], num_samples=12):
    # Select random samples from the test set
    indices = random.sample(range(len(x_test)), num_samples)
    test_images = x_test[indices]
    true_labels = y_test.toarray()[indices]  # Convert sparse matrix to array if needed
    true_labels = np.argmax(true_labels, axis=1)  # Get true class indices

    # Make predictions on the selected test samples
    predictions = model.predict(test_images)
    predicted_labels = np.argmax(predictions, axis=1)  # Get predicted class indices
    # Set up the plotting layout for 4 images per row
    plt.figure(figsize=(18, 12))
    rows = num_samples // 4 + int(num_samples % 4 != 0)  # Calculate the required number of rows

    for i, (img, true_label, pred_label, pred_prob) in enumerate(
            zip(test_images, true_labels, predicted_labels, predictions)):
        plt.subplot(rows, 4, i + 1)  # Arrange images in 4 columns
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"True: {class_labels[true_label]}\nPred: {class_labels[pred_label]} ({max(pred_prob) * 100:.2f}%)",
                  fontsize=12, color="blue" if true_label == pred_label else "red")  # Color title based on correctness

    plt.subplots_adjust(wspace=0.4, hspace=0.6)  # Adjust spacing between plots
    plt.suptitle("Model Predictions on Test Images", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

##########################################################
# Explainability
########################################################
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model


def grad_cam(model, img_array, class_index, last_conv_layer_name):
    """
    Compute Grad-CAM for the given input image and model.

    Parameters:
    model (tf.keras.Model): Trained model.
    img_array (numpy.ndarray): Preprocessed input image (shape: (1, height, width, channels)).
    class_index (int): Index of the target class.
    last_conv_layer_name (str): Name of the last convolutional layer in the model.

    Returns:
    heatmap (numpy.ndarray): Grad-CAM heatmap (shape: (height, width)).
    """
    grad_model = Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1).numpy()

    # Normalize heatmap to 0-1
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1
    return heatmap


def overlay_heatmap(img, heatmap, alpha=0.4, colormap='viridis'):
    """
    Overlay the Grad-CAM heatmap on the input image.

    Parameters:
    img (numpy.ndarray): Original image.
    heatmap (numpy.ndarray): Grad-CAM heatmap.
    alpha (float): Heatmap transparency.
    colormap (str): Matplotlib colormap.

    Returns:
    overlayed_image (numpy.ndarray): Image with heatmap overlay.
    """
    heatmap = plt.cm.get_cmap(colormap)(heatmap)
    heatmap = np.uint8(255 * heatmap[:, :, :3])  # Remove alpha channel

    overlayed_image = img.copy()
    overlayed_image = np.uint8((1 - alpha) * overlayed_image + alpha * heatmap)
    return overlayed_image


def visualize_grad_cam(base_model, modified_model, img_array, class_index, class_labels, ground_truth, prediction):
    """
    Visualize Grad-CAM for a given image and target class.

    Parameters:
    base_model (tf.keras.Model): The pre-trained base VGG19 model.
    modified_model (tf.keras.Model): The overall model (base_model + custom layers).
    img_array (numpy.ndarray): Preprocessed image array.
    class_index (int): Index of the target class.
    class_labels (list): List of class names.
    """
    # Ensure input layer is properly connected
    input_layer = modified_model.input

    # Get the last conv layer from the base model (VGG19)
    last_conv_layer = base_model.get_layer('block5_conv4')
    #print('summary of base model')
    #print(base_model.summary())
    #print('summary of custom layers')
    #print(modified_model.summary())

    # Assuming the original model is 'modified_model' and it has already been trained
    # Define the input layer explicitly
    input_layer = tf.keras.layers.Input(shape=(128, 128, 3))

    # Use the VGG19 part from the trained modified model
    vgg19_output = modified_model.get_layer('vgg19')(input_layer)

    # Now connect the rest of the layers, ensuring you're using the same structure as in 'modified_model'
    flatten = tf.keras.layers.Flatten()(vgg19_output)
    batch_norm_1 = tf.keras.layers.BatchNormalization()(flatten)
    dense_1 = tf.keras.layers.Dense(512)(batch_norm_1)
    batch_norm_2 = tf.keras.layers.BatchNormalization()(dense_1)
    dense_2 = tf.keras.layers.Dense(256)(batch_norm_2)
    batch_norm_3 = tf.keras.layers.BatchNormalization()(dense_2)
    dropout_1 = tf.keras.layers.Dropout(0.5)(batch_norm_3)
    dense_3 = tf.keras.layers.Dense(128)(dropout_1)
    batch_norm_4 = tf.keras.layers.BatchNormalization()(dense_3)
    dropout_2 = tf.keras.layers.Dropout(0.5)(batch_norm_4)
    reduce_dimension = tf.keras.layers.Dense(64)(dropout_2)
    dense_4 = tf.keras.layers.Dense(3)(reduce_dimension)
    #print('layer names')
    #for layer in modified_model.layers:
    #    print(layer.name)

    from tensorflow.keras.layers import Input, Flatten, Dense
    # Get the input shape (assuming 128x128x3 input shape)
    input_layer = Input(shape=(128, 128, 3))

    # Use the VGG19 model up to block5_conv4
    vgg19 = tf.keras.applications.VGG19(include_top=False, weights='imagenet', input_tensor=input_layer)
    block5_conv4 = vgg19.get_layer('block5_conv4').output

    # Flatten the convolutional layer output
    flatten = Flatten()(block5_conv4)

    # Add a Dense layer to match the required input dimensions (64 units in this case)
    dense_64 = Dense(64, activation='relu')(flatten)  # Match the expected input size (64)

    # Then add the original dense_4 layer (which should have 64 input units)
    dense_4 = modified_model.get_layer('dense_4')(dense_64)

    # Create the new Grad-CAM model
    grad_model = Model(inputs=input_layer, outputs=[block5_conv4, dense_4])

    # Check the model summary to verify the structure
    #grad_model.summary()

    # Check if the number of layers in the new model matches the original model
    # If the model architectures are identical, you can set the weights
    try:
        grad_model.set_weights(modified_model.get_weights())
        print("Weights loaded successfully")
    except ValueError as e:
        print(f"Error loading weights: {e}")
        # If there's a mismatch, try selectively loading weights
        for layer in grad_model.layers:
            if layer.name in [l.name for l in modified_model.layers]:
                layer.set_weights(modified_model.get_layer(layer.name).get_weights())

    # After setting the weights, you can print the summary
    #print(grad_model.summary())

    # Compute gradients for the specified class
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_index]  # Get the loss for the target class

    # Compute gradients of the loss w.r.t. the convolutional outputs
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # Pool gradients

    # Generate the heatmap using the pooled gradients
    conv_outputs = conv_outputs[0]  # We are assuming a batch size of 1
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1).numpy()

    # Apply ReLU and normalize the heatmap
    import numpy as np
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1
    '''
    # Display the image with the Grad-CAM heatmap overlay
    import matplotlib.pyplot as plt
    plt.imshow(img_array[0])  # Show the original image
    plt.imshow(heatmap, cmap='viridis', alpha=0.6)  # Overlay the heatmap
    plt.title(f"Class: {class_labels[class_index]}")
    plt.axis('off')
    plt.show()
    
    # Normalize the heatmap to the range of [0, 1] for better blending
    heatmap = np.uint8(255 * heatmap)  # Scale the heatmap to the range [0, 255]

    # Apply the colormap to the heatmap
    heatmap = plt.cm.viridis(heatmap)  # Apply 'viridis' colormap

    # Create a figure and a 1x3 grid of subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot the original grayscale image in the first subplot
    axes[0].imshow(img_array[0], cmap='gray')
    axes[0].set_title("Original Image")
    axes[0].axis('off')  # Hide axes

    # Plot the heatmap in the second subplot
    axes[1].imshow(heatmap, cmap='hot')
    axes[1].set_title("Heatmap")
    axes[1].axis('off')  # Hide axes

    # Plot the superimposed image with the heatmap in the third subplot
    axes[2].imshow(img_array[0], cmap='gray')  # Display the grayscale image
    axes[2].imshow(heatmap,'hot', alpha=0.4)  # Overlay the heatmap with transparency
    axes[2].set_title("Superimposed Image and Heatmap")
    axes[2].axis('off')  # Hide axes

    # Add a title for the whole figure
    fig.suptitle(f"Class: {class_labels[class_index]}", fontsize=16)

    plt.tight_layout()
    plt.show()
    '''

    # Assuming img_array and heatmap are already defined
    # img_array: Original image (grayscale)
    # heatmap: Grad-CAM heatmap
    # sample_label: True label of the image
    # pred_label: Predicted label of the image

    # Resize the heatmap to match the size of the original image
    heatmap_resized = cv2.resize(heatmap, (img_array.shape[2], img_array.shape[1]))  # Resize to (width, height)
    heatmap_resized = heatmap_resized * 255  # Scale heatmap to [0, 255] range
    heatmap_resized = np.clip(heatmap_resized, 0, 255).astype(np.uint8)  # Clip to ensure it's within [0, 255]

    print(f"Heatmap min: {heatmap_resized.min()}, max: {heatmap_resized.max()}")

    # Apply a color map to the heatmap (using HOT colormap)
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_HOT)

    # Convert the original image to [0, 255] scale if necessary
    if img_array[0].max() <= 1.0:
        original_image = (img_array[0] * 255).astype("uint8")
    else:
        original_image = img_array[0].astype("uint8")

    # Ensure the image and heatmap have compatible shapes
    original_image_resized = cv2.resize(original_image, (heatmap_colored.shape[1], heatmap_colored.shape[0]))

    # Ensure dtype compatibility for blending
    original_image_resized = original_image_resized.astype("float32")
    heatmap_colored = heatmap_colored.astype("float32")

    # Blend images
    #super_imposed_image = cv2.addWeighted(original_image_resized, 0.8, heatmap_colored, 0.003, 0.0)
    super_imposed_image = cv2.addWeighted(original_image_resized.astype("float32"), 0.8,
                                          heatmap_colored.astype("float32"), 0.4, 0.0)

    # Visualize results
    f, ax = plt.subplots(1, 3, figsize=(15, 8))

    # Display the original image
    ax[0].imshow(original_image / 255.0)
    ax[0].set_title("Original Image")
    ax[0].set_title(f"True label: {ground_truth} \n Predicted label: {prediction}")

    ax[0].axis("off")

    # Display the heatmap
    ax[1].imshow(heatmap_colored / 255.0)
    ax[1].set_title("Class Activation Map")
    ax[1].axis("off")

    # Display the superimposed image
    ax[2].imshow(super_imposed_image / 255.0)
    ax[2].set_title("Superimposed Image")
    ax[2].axis("off")

    plt.tight_layout()
    plt.show()




def visualize_model_performance_with_grad_cam(
        base_model,
        modified_model,
        x_test,
        y_test,
        class_labels,
        last_conv_layer_name="block5_conv4",
        num_samples=1,
):
    """
    Visualize Grad-CAM for multiple samples in the test set.

    Parameters:
    base_model (tf.keras.Model): The pre-trained base VGG19 model.
    modified_model (tf.keras.Model): The overall model (base_model + custom layers).
    x_test (numpy.ndarray): Test images preprocessed for VGG19.
    y_test (numpy.ndarray): True labels for the test set in one-hot format.
    class_labels (list): List of class names.
    last_conv_layer_name (str): Name of the last convolutional layer to visualize.
    num_samples (int): Number of test samples to visualize.

    Returns:
    None
    """
    # Ensure the number of samples to visualize is within range
    num_samples = min(num_samples, len(x_test))

    # Randomly select test indices
    sample_indices = np.random.choice(range(len(x_test)), size=num_samples, replace=False)

    for idx in sample_indices:
        img_array = np.expand_dims(x_test[idx], axis=0)  # Expand dimension for batch size 1
        true_label = np.argmax(y_test[idx])  # Get the true label index
        pred_label = np.argmax(modified_model.predict(img_array))  # Predicted label index

        print(f"Sample Index: {idx}")
        print(f"True Label: {class_labels[true_label]}, Predicted Label: {class_labels[pred_label]}")
        ground_truth = class_labels[true_label]
        prediction = class_labels[pred_label]
        # Generate Grad-CAM visualization
        visualize_grad_cam(
            base_model=base_model,
            modified_model=modified_model,
            img_array=img_array,
            class_index=pred_label,
            class_labels=class_labels,
            ground_truth = ground_truth,
            prediction= prediction
        )




