"""
Extract Sample Images for LaTeX Report
This script copies sample images from each class to use in the LaTeX document
"""
import os
import shutil
import cv2

def extract_sample_images():
    """Extract one sample image from each class for the LaTeX report"""
    
    data_folder = "image_classification_project/data"
    
    # Define the actual paths based on the folder structure
    class_paths = {
        'Car': os.path.join(data_folder, 'Car-Bike-Dataset', 'Car'),
        'Bike': os.path.join(data_folder, 'Car-Bike-Dataset', 'Bike'),
        'Truck': os.path.join(data_folder, 'Truck')
    }
    
    output_names = {
        'Car': 'car_sample.jpg',
        'Bike': 'bike_sample.jpg', 
        'Truck': 'truck_sample.jpg'
    }
    
    print("🖼️  Extracting sample images for LaTeX report...")
    
    for class_name, class_path in class_paths.items():
        if os.path.exists(class_path):
            # Get the first available image
            image_files = [f for f in os.listdir(class_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if image_files:
                # Copy the first image as a sample
                source_path = os.path.join(class_path, image_files[0])
                dest_path = output_names[class_name]
                
                try:
                    # Load and resize the image to a reasonable size for the report
                    img = cv2.imread(source_path)
                    if img is not None:
                        # Resize to 300x200 for good quality in LaTeX
                        img_resized = cv2.resize(img, (300, 200))
                        cv2.imwrite(dest_path, img_resized)
                        print(f"✅ Extracted {class_name} sample: {dest_path}")
                    else:
                        # Fallback to direct copy if cv2 fails
                        shutil.copy2(source_path, dest_path)
                        print(f"✅ Copied {class_name} sample: {dest_path}")
                        
                except Exception as e:
                    print(f"❌ Error extracting {class_name} sample: {e}")
            else:
                print(f"❌ No images found in {class_name} folder")
        else:
            print(f"❌ Path not found: {class_path}")
    
    print("\n📝 Update your LaTeX file:")
    print("Replace the tcolorbox placeholders with:")
    print("\\includegraphics[width=\\textwidth,height=3cm,keepaspectratio]{car_sample.jpg}")
    print("\\includegraphics[width=\\textwidth,height=3cm,keepaspectratio]{bike_sample.jpg}")
    print("\\includegraphics[width=\\textwidth,height=3cm,keepaspectratio]{truck_sample.jpg}")

if __name__ == "__main__":
    extract_sample_images() 