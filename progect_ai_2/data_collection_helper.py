"""
Data Collection Helper for Image Classification Project

"""

import os
import requests
import cv2
import numpy as np
from urllib.parse import urlparse
import time

def create_dataset_structure(base_path, class_names):
    """
    Create the proper directory structure for the dataset.
    
    Args:
        base_path (str): Base path for the dataset
        class_names (list): List of class names to create directories for
    """
    print(f"Creating dataset structure at {base_path}...")
    
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    
    for class_name in class_names:
        class_path = os.path.join(base_path, class_name)
        if not os.path.exists(class_path):
            os.makedirs(class_path)
            print(f"Created directory: {class_path}")

def download_sample_images(url_list, save_path, class_name):
    """
    Download images from a list of URLs.
    
    Args:
        url_list (list): List of image URLs
        save_path (str): Path to save images
        class_name (str): Name of the class for naming files
    """
    print(f"Downloading {len(url_list)} images for {class_name}...")
    
    for i, url in enumerate(url_list):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                filename = f"{class_name}_{i+1:03d}.jpg"
                filepath = os.path.join(save_path, filename)
                
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                # Verify the image can be read
                img = cv2.imread(filepath)
                if img is None:
                    os.remove(filepath)
                    print(f"Removed corrupted image: {filename}")
                else:
                    print(f"Downloaded: {filename}")
            
            time.sleep(0.5)  # Be respectful to servers
            
        except Exception as e:
            print(f"Failed to download {url}: {e}")

def validate_dataset(data_path, min_images_per_class=50):
    """
    Validate that the dataset meets project requirements.
    
    Args:
        data_path (str): Path to the dataset
        min_images_per_class (int): Minimum images required per class
    
    Returns:
        dict: Validation results
    """
    print("Validating dataset...")
    
    if not os.path.exists(data_path):
        return {"valid": False, "error": "Dataset path does not exist"}
    
    class_dirs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    
    if len(class_dirs) < 3:
        return {"valid": False, "error": f"Need at least 3 classes, found {len(class_dirs)}"}
    
    class_counts = {}
    total_images = 0
    
    for class_name in class_dirs:
        class_path = os.path.join(data_path, class_name)
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        class_counts[class_name] = len(image_files)
        total_images += len(image_files)
        
        if len(image_files) < min_images_per_class:
            return {
                "valid": False, 
                "error": f"Class '{class_name}' has only {len(image_files)} images, need at least {min_images_per_class}"
            }
    
    if total_images < 500:
        return {
            "valid": False,
            "error": f"Total images: {total_images}, need at least 500"
        }
    
    return {
        "valid": True,
        "total_images": total_images,
        "num_classes": len(class_dirs),
        "class_counts": class_counts
    }

def resize_existing_images(data_path, target_size=(64, 64)):
    """
    Resize all images in the dataset to a consistent size.
    
    Args:
        data_path (str): Path to the dataset
        target_size (tuple): Target size for images
    """
    print(f"Resizing images to {target_size}...")
    
    class_dirs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    
    for class_name in class_dirs:
        class_path = os.path.join(data_path, class_name)
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        print(f"Processing {len(image_files)} images in {class_name}...")
        
        for image_file in image_files:
            image_path = os.path.join(class_path, image_file)
            
            try:
                img = cv2.imread(image_path)
                if img is not None:
                    img_resized = cv2.resize(img, target_size)
                    cv2.imwrite(image_path, img_resized)
                else:
                    print(f"Could not read {image_file}, removing...")
                    os.remove(image_path)
            except Exception as e:
                print(f"Error processing {image_file}: {e}")

def get_dataset_recommendations():
    """
    Provide recommendations for dataset sources and themes.
    """
    print("\n" + "="*60)
    print("DATASET COLLECTION RECOMMENDATIONS")
    print("="*60)
    
    print("\n1. RECOMMENDED DATASET THEMES:")
    print("   • Vehicles: cars, trucks, motorcycles, bicycles")
    print("   • Animals: cats, dogs, birds, horses")
    print("   • Food: pizza, burgers, salad, pasta")
    print("   • Objects: chairs, tables, phones, bottles")
    print("   • Traffic Signs: stop, yield, warning, speed limit")
    
    print("\n2. DATA SOURCES:")
    print("   • Kaggle: kaggle.com/datasets (search 'image classification')")
    print("   • Google Images: Use bulk download tools")
    print("   • Unsplash: unsplash.com (high quality, free)")
    print("   • Pixabay: pixabay.com (free images)")
    print("   • Your own photos: Take consistent, clear photos")
    
    print("\n3. COLLECTION TIPS:")
    print("   • Aim for 100-200 images per class")
    print("   • Ensure consistent image quality")
    print("   • Vary lighting, angles, backgrounds")
    print("   • Remove duplicates and corrupted files")
    print("   • Balance class distributions")
    
    print("\n4. LEGAL CONSIDERATIONS:")
    print("   • Use royalty-free or CC-licensed images")
    print("   • Cite sources in your project report")
    print("   • Avoid copyrighted material")

def main():
    """
    Main function to demonstrate dataset preparation.
    """
    print("Image Classification Dataset Helper")
    print("=" * 40)
    
    # Configuration
    DATA_PATH = "image_classification_project/data"
    SAMPLE_CLASSES = ["car", "truck", "motorcycle"]
    
    # Create basic structure
    create_dataset_structure(DATA_PATH, SAMPLE_CLASSES)
    
    # Validate current dataset
    validation = validate_dataset(DATA_PATH, min_images_per_class=10)  # Lower threshold for demo
    
    if validation["valid"]:
        print(f"\n✅ Dataset is valid!")
        print(f"   Total images: {validation['total_images']}")
        print(f"   Classes: {validation['num_classes']}")
        print(f"   Distribution: {validation['class_counts']}")
    else:
        print(f"\n❌ Dataset validation failed: {validation['error']}")
    
    # Show recommendations
    get_dataset_recommendations()
    
    print(f"\n📝 Next steps:")
    print(f"   1. Populate each class directory with 100+ images")
    print(f"   2. Run 'python main.py' to train and evaluate models")
    print(f"   3. Analyze results and write your report")

if __name__ == "__main__":
    main()
