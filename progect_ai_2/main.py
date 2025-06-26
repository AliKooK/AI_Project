"""
Comparative Study of Image Classification Using Decision Tree, Naive Bayes, and Feedforward Neural Networks

This project implements and compares three machine learning models for image classification:
1. Naive Bayes Classifier
2. Decision Tree Classifier  
3. Feedforward Neural Network (MLPClassifier)

Author: [Your Name]
Date: [Current Date]
Course: [Course Name]
"""

import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set high-quality matplotlib parameters
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 18
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                               f1_score, confusion_matrix, ConfusionMatrixDisplay,
                               classification_report)
from sklearn.preprocessing import StandardScaler
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib style for high-quality output
plt.style.use('default')
sns.set_palette("husl")

# --- 1. Data Loading and Preprocessing ---
def load_and_prepare_data(data_folder, image_size=(64, 64), verbose=True):
    """
    Loads, resizes, and flattens images from the specified folder.
    Handles the specific structure: Car-Bike-Dataset/{Car,Bike} and Truck/
    
    Args:
        data_folder (str): Path to the main dataset directory.
        image_size (tuple): The target dimensions for resizing images.
        verbose (bool): Whether to print progress information.
    
    Returns:
        tuple: A tuple of (flattened_images, labels, class_names).
    """
    images = []
    labels = []
    
    # Define the actual class paths based on the folder structure
    class_paths = {
        'Car': os.path.join(data_folder, 'Car-Bike-Dataset', 'Car'),
        'Bike': os.path.join(data_folder, 'Car-Bike-Dataset', 'Bike'),
        'Truck': os.path.join(data_folder, 'Truck')
    }
    
    # Filter to only existing paths
    existing_classes = {}
    for class_name, class_path in class_paths.items():
        if os.path.exists(class_path):
            existing_classes[class_name] = class_path
            
    class_names = sorted(existing_classes.keys())
    label_dict = {name: i for i, name in enumerate(class_names)}
    
    if verbose:
        print(f"Loading images from {data_folder}...")
        print(f"Found {len(class_names)} classes: {class_names}")
    
    class_counts = {}
    
    for class_name in class_names:
        class_path = existing_classes[class_name]
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        class_counts[class_name] = len(image_files)
        
        if verbose:
            print(f"Processing {class_name}: {len(image_files)} images from {class_path}")
        
        for image_name in image_files:
            image_path = os.path.join(class_path, image_name)
            # Read image in grayscale to simplify feature space
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # Resize and flatten the image
                img_resized = cv2.resize(img, image_size)
                # Normalize pixel values to [0, 1]
                img_normalized = img_resized.astype(np.float32) / 255.0
                images.append(img_normalized.flatten())
                labels.append(label_dict[class_name])
            elif verbose:
                print(f"Warning: Could not read image {image_path}")
    
    if verbose:
        print(f"Successfully loaded {len(images)} images.")
        print("Class distribution:")
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count} images")
    
    return np.array(images), np.array(labels), class_names, class_counts

def plot_sample_images(data_folder, class_names, image_size=(64, 64), samples_per_class=3):
    """
    Display sample images from each class.
    """
    # Define the actual class paths based on the folder structure
    class_paths = {
        'Car': os.path.join(data_folder, 'Car-Bike-Dataset', 'Car'),
        'Bike': os.path.join(data_folder, 'Car-Bike-Dataset', 'Bike'),
        'Truck': os.path.join(data_folder, 'Truck')
    }
    
    # Filter to only classes that exist in our class_names
    existing_class_paths = {name: class_paths[name] for name in class_names if name in class_paths and os.path.exists(class_paths[name])}
    
    # Calculate the maximum number of samples available
    max_samples = 0
    for class_name in class_names:
        if class_name in existing_class_paths:
            class_path = existing_class_paths[class_name]
            image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            max_samples = max(max_samples, len(image_files))
    
    actual_samples_per_class = min(samples_per_class, max_samples)
    if actual_samples_per_class == 0:
        print("No images found to display.")
        return
    
    fig, axes = plt.subplots(len(class_names), actual_samples_per_class, figsize=(8*actual_samples_per_class, 6*len(class_names)))
    
    # Handle single row or single column cases
    if len(class_names) == 1:
        axes = axes.reshape(1, -1) if actual_samples_per_class > 1 else [axes]
    elif actual_samples_per_class == 1:
        axes = axes.reshape(-1, 1)
    
    for i, class_name in enumerate(class_names):
        if class_name in existing_class_paths:
            class_path = existing_class_paths[class_name]
            image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            
            for j in range(actual_samples_per_class):
                if j < len(image_files):
                    img_path = os.path.join(class_path, image_files[j])
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img_resized = cv2.resize(img, image_size)
                        if len(class_names) == 1 and actual_samples_per_class == 1:
                            axes.imshow(img_resized, cmap='gray')
                            axes.set_title(f'{class_name} - Sample {j+1}')
                            axes.axis('off')
                        else:
                            axes[i, j].imshow(img_resized, cmap='gray')
                            axes[i, j].set_title(f'{class_name} - Sample {j+1}')
                            axes[i, j].axis('off')
                else:
                    # Fill empty slots
                    if len(class_names) == 1 and actual_samples_per_class == 1:
                        axes.axis('off')
                    else:
                        axes[i, j].axis('off')
                        axes[i, j].text(0.5, 0.5, 'No Image', ha='center', va='center', transform=axes[i, j].transAxes)
        else:
            # Handle missing class folder
            for j in range(actual_samples_per_class):
                if len(class_names) == 1 and actual_samples_per_class == 1:
                    axes.axis('off')
                    axes.text(0.5, 0.5, 'Folder Not Found', ha='center', va='center', transform=axes.transAxes)
                else:
                    axes[i, j].axis('off')
                    axes[i, j].text(0.5, 0.5, 'Folder Not Found', ha='center', va='center', transform=axes[i, j].transAxes)
    
    plt.tight_layout()
    plt.savefig('sample_images.png', dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.show()
    print("✅ Sample images saved as 'sample_images.png'")

# --- 2. Model Training and Evaluation ---
def evaluate_model(model, X_train, y_train, X_test, y_test, class_names, model_name=None):
    """
    Trains the model and returns comprehensive performance metrics.
    
    Args:
        model: The scikit-learn model instance.
        X_train, y_train: Training data and labels.
        X_test, y_test: Testing data and labels.
        class_names: List of class names.
        model_name: Optional custom model name.
    
    Returns:
        dict: A dictionary containing comprehensive performance metrics.
    """
    if model_name is None:
        model_name = model.__class__.__name__
    
    print(f"\n--- Training {model_name} ---")
    
    # Training
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Cross-validation scores
    cv_scores = cross_val_score(model, X_train, y_train, cv=min(5, len(X_train)), scoring='accuracy')
    
    # Calculate metrics
    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
        "Recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, average='weighted', zero_division=0),
        "Confusion Matrix": confusion_matrix(y_test, y_pred, labels=range(len(class_names))),
        "Training Time (s)": training_time,
        "CV Mean Accuracy": cv_scores.mean(),
        "CV Std Accuracy": cv_scores.std(),
        "Classification Report": classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
    }
    
    print(f"Training completed in {training_time:.2f} seconds.")
    print(f"Test Accuracy: {metrics['Accuracy']:.4f}")
    print(f"Cross-validation Accuracy: {metrics['CV Mean Accuracy']:.4f} (+/- {metrics['CV Std Accuracy']*2:.4f})")
    
    return metrics

def plot_results_comparison(results):
    """
    Create comprehensive visualization of model comparison.
    """
    # Prepare data for plotting
    models = [r['Model'] for r in results]
    accuracy = [r['Accuracy'] for r in results]
    precision = [r['Precision'] for r in results]
    recall = [r['Recall'] for r in results]
    f1_scores = [r['F1-Score'] for r in results]
    training_times = [r['Training Time (s)'] for r in results]
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    
    # Performance metrics comparison
    metrics_data = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1_scores
    }
    
    x = np.arange(len(models))
    width = 0.2
    
    for i, (metric, values) in enumerate(metrics_data.items()):
        axes[0, 0].bar(x + i*width, values, width, label=metric)
    
    axes[0, 0].set_xlabel('Models', fontsize=14)
    axes[0, 0].set_ylabel('Score', fontsize=14)
    axes[0, 0].set_title('Model Performance Comparison', fontsize=16, fontweight='bold')
    axes[0, 0].set_xticks(x + width * 1.5)
    axes[0, 0].set_xticklabels(models, fontsize=12)
    axes[0, 0].legend(fontsize=12)
    axes[0, 0].set_ylim(0, 1.1)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Training time comparison
    axes[0, 1].bar(models, training_times, color='orange', alpha=0.8)
    axes[0, 1].set_xlabel('Models', fontsize=14)
    axes[0, 1].set_ylabel('Training Time (seconds)', fontsize=14)
    axes[0, 1].set_title('Training Time Comparison', fontsize=16, fontweight='bold')
    axes[0, 1].tick_params(axis='x', rotation=45, labelsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Accuracy with error bars (CV)
    cv_means = [r['CV Mean Accuracy'] for r in results]
    cv_stds = [r['CV Std Accuracy'] for r in results]
    axes[0, 2].bar(models, cv_means, yerr=cv_stds, capsize=8, color='green', alpha=0.7)
    axes[0, 2].set_xlabel('Models', fontsize=14)
    axes[0, 2].set_ylabel('Cross-Validation Accuracy', fontsize=14)
    axes[0, 2].set_title('Cross-Validation Accuracy with Error Bars', fontsize=16, fontweight='bold')
    axes[0, 2].tick_params(axis='x', rotation=45, labelsize=12)
    axes[0, 2].set_ylim(0, 1.1)
    axes[0, 2].grid(True, alpha=0.3)
    
    # Individual confusion matrices
    for i, result in enumerate(results):
        if i < 3:  # Only plot first 3 models
            cm = result['Confusion Matrix']
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
            disp.plot(ax=axes[1, i], cmap='Blues', values_format='d')
            axes[1, i].set_title(f"Confusion Matrix: {result['Model']}", fontsize=14, fontweight='bold')
            
            # Add text annotations for better visibility in small datasets
            for j in range(len(class_names)):
                for k in range(len(class_names)):
                    axes[1, i].text(k, j, str(cm[j, k]), ha='center', va='center', 
                                  color='white' if cm[j, k] > cm.max()/2 else 'black', 
                                  fontsize=16, weight='bold')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.show()
    print("✅ Model comparison chart saved as 'model_comparison.png'")

def create_results_summary(results, class_names, class_counts):
    """
    Create a comprehensive summary report.
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE RESULTS SUMMARY")
    print("="*80)
    
    print(f"\nDataset Information:")
    print(f"  Total Classes: {len(class_names)}")
    print(f"  Classes: {', '.join(class_names)}")
    print(f"  Total Images: {sum(class_counts.values())}")
    print(f"  Class Distribution: {dict(class_counts)}")
    
    print(f"\nModel Performance Summary:")
    print("-" * 80)
    
    # Create a DataFrame for better formatting
    summary_data = []
    for result in results:
        summary_data.append({
            'Model': result['Model'],
            'Accuracy': f"{result['Accuracy']:.4f}",
            'Precision': f"{result['Precision']:.4f}",
            'Recall': f"{result['Recall']:.4f}",
            'F1-Score': f"{result['F1-Score']:.4f}",
            'CV Accuracy': f"{result['CV Mean Accuracy']:.4f} ± {result['CV Std Accuracy']:.4f}",
            'Training Time': f"{result['Training Time (s)']:.2f}s"
        })
    
    df = pd.DataFrame(summary_data)
    print(df.to_string(index=False))
    
    # Best model analysis
    best_accuracy_idx = np.argmax([r['Accuracy'] for r in results])
    fastest_model_idx = np.argmin([r['Training Time (s)'] for r in results])
    
    print(f"\nModel Analysis:")
    print(f"  Best Accuracy: {results[best_accuracy_idx]['Model']} ({results[best_accuracy_idx]['Accuracy']:.4f})")
    print(f"  Fastest Training: {results[fastest_model_idx]['Model']} ({results[fastest_model_idx]['Training Time (s)']:.2f}s)")
    
    # Detailed classification reports
    print(f"\nDetailed Classification Reports:")
    print("-" * 80)
    for result in results:
        print(f"\n{result['Model']}:")
        print(result['Classification Report'])

# --- 3. Dataset Preparation Helper ---
def setup_sample_dataset(data_path):
    """
    Creates additional sample classes if only one class exists.
    This is a helper function for demonstration purposes.
    """
    existing_classes = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    
    if len(existing_classes) < 3:
        print(f"\nWarning: Only {len(existing_classes)} class(es) found. For a proper classification task,")
        print("you need at least 3 classes with sufficient images in each class.")
        print("\nTo complete this project, please:")
        print("1. Add more image classes to your dataset")
        print("2. Ensure each class has at least 50-100 images")
        print("3. Aim for a total of 500+ images across all classes")
        print("\nExample dataset structure:")
        print("  data/")
        print("    ├── cars/")
        print("    ├── trucks/")
        print("    ├── motorcycles/")
        print("    └── bicycles/")
        
        return False
    return True

# --- 4. Main Execution ---
if __name__ == "__main__":
    # Configuration
    DATA_PATH = "image_classification_project/data"
    IMAGE_DIMENSIONS = (64, 64)  # Optimized resolution for faster processing
    TEST_SET_SIZE = 0.25
    SEED = 42
    
    print("="*80)
    print("COMPARATIVE STUDY OF IMAGE CLASSIFICATION")
    print("Decision Tree vs Naive Bayes vs Feedforward Neural Network")
    print("="*80)
    
    # Check if dataset meets requirements
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data path '{DATA_PATH}' does not exist.")
        print("Please ensure your dataset is properly organized.")
        exit(1)
    
    # Step 1: Load and process the data
    try:
        X, y, class_names, class_counts = load_and_prepare_data(DATA_PATH, IMAGE_DIMENSIONS)
        
        # Check dataset adequacy
        if not setup_sample_dataset(DATA_PATH):
            print("\nProceeding with available data for demonstration...")
        
        # Display sample images
        print("\nDisplaying sample images from each class...")
        plot_sample_images(DATA_PATH, class_names, IMAGE_DIMENSIONS)
        
    except Exception as e:
        print(f"Error loading data: {e}")
        exit(1)
    
    # Step 2: Split data into training and testing sets
    if len(np.unique(y)) < 2:
        print("Error: Need at least 2 classes for classification.")
        exit(1)
    
    # For small datasets, use a smaller test size and ensure minimum samples per class
    total_samples = len(X)
    if total_samples < 20:
        # For very small datasets, use a fixed split or cross-validation only
        print(f"Small dataset detected ({total_samples} samples). Using minimal test split.")
        test_size = max(1, int(total_samples * 0.2))  # At least 1 sample for testing
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=SEED
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SET_SIZE, random_state=SEED, stratify=y
        )
    
    print(f"\nDataset split: {len(X_train)} training samples, {len(X_test)} testing samples.")
    
    # Step 3: Feature scaling for Neural Network
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Step 4: Initialize models with optimal parameters
    models = [
        ('Naive Bayes', GaussianNB()),
        ('Decision Tree', DecisionTreeClassifier(
            random_state=SEED,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )),
        ('Neural Network (MLP)', MLPClassifier(
            hidden_layer_sizes=(100, 50),
            max_iter=500,
            random_state=SEED,
            learning_rate='adaptive',
            early_stopping=True,
            validation_fraction=0.1
        ))
    ]
    
    # Step 5: Train and evaluate each model
    results = []
    
    for model_name, model in models:
        if 'Neural Network' in model_name:
            # Use scaled features for neural network
            result = evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test, class_names, model_name)
        else:
            # Use original features for other models
            result = evaluate_model(model, X_train, y_train, X_test, y_test, class_names, model_name)
        results.append(result)
    
    # Step 6: Create comprehensive results visualization
    print("\nGenerating comprehensive results visualization...")
    plot_results_comparison(results)
    
    # Step 7: Display comprehensive summary
    create_results_summary(results, class_names, class_counts)
    
    # Step 8: Visualize Decision Tree (if applicable)
    if len(results) > 1:  # Ensure Decision Tree exists
        print("\nGenerating Decision Tree visualization...")
        dt_model = models[1][1]  # Get Decision Tree model
        
        plt.figure(figsize=(28, 16))
        plot_tree(dt_model, max_depth=3, class_names=class_names, 
                 filled=True, rounded=True, fontsize=14)
        plt.title("Decision Tree Visualization (Top 3 Levels)", fontsize=20, pad=20)
        plt.savefig('decision_tree.png', dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.show()
        print("✅ Decision tree visualization saved as 'decision_tree.png'")
    
    # Step 9: Save results to file
    print("\nSaving results to 'results_summary.txt'...")
    with open('results_summary.txt', 'w') as f:
        f.write("COMPARATIVE STUDY OF IMAGE CLASSIFICATION RESULTS\n")
        f.write("="*50 + "\n\n")
        f.write(f"Dataset: {len(class_names)} classes, {sum(class_counts.values())} total images\n")
        f.write(f"Classes: {', '.join(class_names)}\n")
        f.write(f"Image dimensions: {IMAGE_DIMENSIONS}\n\n")
        
        for result in results:
            f.write(f"{result['Model']}:\n")
            f.write(f"  Accuracy: {result['Accuracy']:.4f}\n")
            f.write(f"  Precision: {result['Precision']:.4f}\n")
            f.write(f"  Recall: {result['Recall']:.4f}\n")
            f.write(f"  F1-Score: {result['F1-Score']:.4f}\n")
            f.write(f"  Training Time: {result['Training Time (s)']:.2f}s\n\n")
    
    print("✅ Results summary saved as 'results_summary.txt'")
    print("\nProject completed successfully!")
    print("Generated files:")
    print("  - sample_images.png: Sample images from each class")
    print("  - model_comparison.png: Comprehensive model comparison")
    print("  - decision_tree.png: Decision tree visualization")
    print("  - results_summary.txt: Detailed results summary")