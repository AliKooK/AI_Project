"""
Terminal Image Gallery - Browse all your images from command line
Navigate through your images using simple text commands
"""
import os
import cv2
from PIL import Image
import time

class TerminalImageGallery:
    def __init__(self):
        self.data_folder = "image_classification_project/data"
        
        # Define the actual paths based on the folder structure
        self.class_paths = {
            'Car': os.path.join(self.data_folder, 'Car-Bike-Dataset', 'Car'),
            'Bike': os.path.join(self.data_folder, 'Car-Bike-Dataset', 'Bike'),
            'Truck': os.path.join(self.data_folder, 'Truck')
        }
        
        self.load_all_images()
        self.current_index = 0
        
    def load_all_images(self):
        """Load all image paths"""
        self.all_images = []  # List of tuples: (class_name, image_path, image_filename)
        
        for class_name, class_path in self.class_paths.items():
            if os.path.exists(class_path):
                image_files = [f for f in os.listdir(class_path) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                for img_file in image_files:
                    full_path = os.path.join(class_path, img_file)
                    self.all_images.append((class_name, full_path, img_file))
                
                print(f"✅ Loaded {len(image_files)} images from {class_name}")
        
        print(f"\n📊 Total: {len(self.all_images)} images across all classes")
        
    def show_current_image_info(self):
        """Display information about current image"""
        if not self.all_images:
            print("❌ No images found!")
            return
            
        class_name, img_path, img_filename = self.all_images[self.current_index]
        
        # Get image dimensions
        try:
            img = cv2.imread(img_path)
            if img is not None:
                height, width, channels = img.shape
                file_size = os.path.getsize(img_path)
                file_size_mb = file_size / (1024 * 1024)
            else:
                height, width, channels = "?", "?", "?"
                file_size_mb = 0
        except:
            height, width, channels = "?", "?", "?"
            file_size_mb = 0
        
        # Clear screen and show info
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("=" * 60)
        print(f"🖼️  TERMINAL IMAGE GALLERY")
        print("=" * 60)
        print(f"📂 Class: {class_name}")
        print(f"📄 File: {img_filename}")
        print(f"📏 Dimensions: {width} x {height} pixels")
        print(f"💾 Size: {file_size_mb:.2f} MB")
        print(f"📍 Image {self.current_index + 1} of {len(self.all_images)}")
        print(f"📂 Path: {img_path}")
        print("=" * 60)
        
    def show_class_summary(self):
        """Show summary of all classes"""
        class_counts = {}
        for class_name, _, _ in self.all_images:
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
        print("\n📊 CLASS SUMMARY:")
        print("-" * 30)
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count} images")
        print("-" * 30)
        print(f"  Total: {len(self.all_images)} images")
        
    def list_images_by_class(self, class_name=None):
        """List images, optionally filtered by class"""
        if class_name:
            filtered_images = [(i, info) for i, info in enumerate(self.all_images) 
                             if info[0].lower() == class_name.lower()]
            if not filtered_images:
                print(f"❌ No images found for class '{class_name}'")
                return
                
            print(f"\n📋 Images in {class_name.upper()} class:")
            print("-" * 50)
            for i, (orig_idx, (cls, path, filename)) in enumerate(filtered_images):
                marker = "👉" if orig_idx == self.current_index else "  "
                print(f"{marker} {i+1:3d}. {filename}")
        else:
            print(f"\n📋 ALL IMAGES:")
            print("-" * 50)
            for i, (cls, path, filename) in enumerate(self.all_images):
                marker = "👉" if i == self.current_index else "  "
                print(f"{marker} {i+1:3d}. [{cls}] {filename}")
    
    def jump_to_image(self, index):
        """Jump to specific image by index"""
        if 1 <= index <= len(self.all_images):
            self.current_index = index - 1
            self.show_current_image_info()
        else:
            print(f"❌ Invalid index! Please enter a number between 1 and {len(self.all_images)}")
    
    def jump_to_class(self, class_name):
        """Jump to first image of specified class"""
        for i, (cls, path, filename) in enumerate(self.all_images):
            if cls.lower() == class_name.lower():
                self.current_index = i
                self.show_current_image_info()
                return
        print(f"❌ Class '{class_name}' not found!")
    
    def search_images(self, search_term):
        """Search for images by filename"""
        matches = []
        for i, (cls, path, filename) in enumerate(self.all_images):
            if search_term.lower() in filename.lower():
                matches.append((i, cls, filename))
        
        if matches:
            print(f"\n🔍 Found {len(matches)} matches for '{search_term}':")
            print("-" * 50)
            for i, (orig_idx, cls, filename) in enumerate(matches):
                marker = "👉" if orig_idx == self.current_index else "  "
                print(f"{marker} {orig_idx+1:3d}. [{cls}] {filename}")
        else:
            print(f"❌ No images found matching '{search_term}'")
    
    def show_help(self):
        """Show available commands"""
        print("\n📖 AVAILABLE COMMANDS:")
        print("-" * 40)
        print("Navigation:")
        print("  n, next     - Next image")
        print("  p, prev     - Previous image")
        print("  f, first    - First image")
        print("  l, last     - Last image")
        print()
        print("Information:")
        print("  i, info     - Show current image info")
        print("  s, summary  - Show class summary")
        print("  list        - List all images")
        print("  list <class> - List images in specific class")
        print()
        print("Jump:")
        print("  go <number> - Jump to image number")
        print("  class <name> - Jump to first image of class")
        print("  search <term> - Search images by filename")
        print()
        print("Other:")
        print("  h, help     - Show this help")
        print("  q, quit     - Quit gallery")
        print("-" * 40)
    
    def run(self):
        """Main gallery loop"""
        if not self.all_images:
            print("❌ No images found! Please check your data folder.")
            return
            
        self.show_current_image_info()
        self.show_help()
        
        while True:
            try:
                command = input("\n💬 Enter command (h for help): ").strip().lower()
                
                if command in ['q', 'quit', 'exit']:
                    print("👋 Goodbye!")
                    break
                    
                elif command in ['n', 'next']:
                    self.current_index = (self.current_index + 1) % len(self.all_images)
                    self.show_current_image_info()
                    
                elif command in ['p', 'prev', 'previous']:
                    self.current_index = (self.current_index - 1) % len(self.all_images)
                    self.show_current_image_info()
                    
                elif command in ['f', 'first']:
                    self.current_index = 0
                    self.show_current_image_info()
                    
                elif command in ['l', 'last']:
                    self.current_index = len(self.all_images) - 1
                    self.show_current_image_info()
                    
                elif command in ['i', 'info']:
                    self.show_current_image_info()
                    
                elif command in ['s', 'summary']:
                    self.show_class_summary()
                    
                elif command == 'list':
                    self.list_images_by_class()
                    
                elif command.startswith('list '):
                    class_name = command[5:].strip()
                    self.list_images_by_class(class_name)
                    
                elif command.startswith('go '):
                    try:
                        index = int(command[3:].strip())
                        self.jump_to_image(index)
                    except ValueError:
                        print("❌ Invalid number! Use: go <number>")
                        
                elif command.startswith('class '):
                    class_name = command[6:].strip()
                    self.jump_to_class(class_name)
                    
                elif command.startswith('search '):
                    search_term = command[7:].strip()
                    if search_term:
                        self.search_images(search_term)
                    else:
                        print("❌ Please provide search term! Use: search <term>")
                        
                elif command in ['h', 'help']:
                    self.show_help()
                    
                elif command == '':
                    continue  # Empty command, just continue
                    
                else:
                    print(f"❌ Unknown command: '{command}'. Type 'h' for help.")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    """Main function"""
    print("🚀 Starting Terminal Image Gallery...")
    gallery = TerminalImageGallery()
    gallery.run()

if __name__ == "__main__":
    main() 