"""
Use Arrow Keys: Left/Right to navigate, Up/Down to switch classes, 'q' to quit
"""
import os
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np

# Set high-quality matplotlib parameters
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['font.size'] = 12

class ImageGallery:
    def __init__(self):
        self.data_folder = "image_classification_project/data"
        
        # Define the actual paths based on the folder structure
        self.class_paths = {
            'Car': os.path.join(self.data_folder, 'Car-Bike-Dataset', 'Car'),
            'Bike': os.path.join(self.data_folder, 'Car-Bike-Dataset', 'Bike'),
            'Truck': os.path.join(self.data_folder, 'Truck')
        }
        
        self.image_size = (400, 400)  # Larger size for better viewing
        self.load_all_images()
        
        self.current_class_idx = 0
        self.current_image_idx = 0
        
        # Create the figure and axis
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.fig.suptitle('Interactive Image Gallery', fontsize=16, fontweight='bold')
        
        # Connect keyboard events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # Add buttons for navigation
        self.add_navigation_buttons()
        
        # Display first image
        self.display_current_image()
        
    def load_all_images(self):
        """Load all image paths organized by class"""
        self.all_images = {}
        self.class_names = []
        
        for class_name, class_path in self.class_paths.items():
            if os.path.exists(class_path):
                image_files = [f for f in os.listdir(class_path) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if image_files:  # Only add classes that have images
                    self.all_images[class_name] = []
                    for img_file in image_files:
                        self.all_images[class_name].append(os.path.join(class_path, img_file))
                    self.class_names.append(class_name)
                    print(f"Loaded {len(image_files)} images from {class_name}")
        
        print(f"Total classes: {len(self.class_names)}")
        total_images = sum(len(imgs) for imgs in self.all_images.values())
        print(f"Total images: {total_images}")
    
    def add_navigation_buttons(self):
        """Add navigation buttons to the interface"""
        # Previous image button
        ax_prev = plt.axes([0.1, 0.02, 0.1, 0.05])
        self.btn_prev = Button(ax_prev, '◀ Prev')
        self.btn_prev.on_clicked(self.prev_image)
        
        # Next image button  
        ax_next = plt.axes([0.25, 0.02, 0.1, 0.05])
        self.btn_next = Button(ax_next, 'Next ▶')
        self.btn_next.on_clicked(self.next_image)
        
        # Previous class button
        ax_prev_class = plt.axes([0.4, 0.02, 0.1, 0.05])
        self.btn_prev_class = Button(ax_prev_class, '▲ Class')
        self.btn_prev_class.on_clicked(self.prev_class)
        
        # Next class button
        ax_next_class = plt.axes([0.55, 0.02, 0.1, 0.05])
        self.btn_next_class = Button(ax_next_class, 'Class ▼')
        self.btn_next_class.on_clicked(self.next_class)
        
        # Auto-scroll button
        ax_auto = plt.axes([0.7, 0.02, 0.1, 0.05])
        self.btn_auto = Button(ax_auto, 'Auto')
        self.btn_auto.on_clicked(self.auto_scroll)
        
        # Quit button
        ax_quit = plt.axes([0.85, 0.02, 0.1, 0.05])
        self.btn_quit = Button(ax_quit, 'Quit')
        self.btn_quit.on_clicked(self.quit_gallery)
    
    def display_current_image(self):
        """Display the current image with information"""
        if not self.class_names:
            self.ax.text(0.5, 0.5, 'No images found!', ha='center', va='center', 
                        transform=self.ax.transAxes, fontsize=20)
            return
            
        current_class = self.class_names[self.current_class_idx]
        current_images = self.all_images[current_class]
        
        if not current_images:
            self.ax.text(0.5, 0.5, 'No images in this class!', ha='center', va='center', 
                        transform=self.ax.transAxes, fontsize=20)
            return
        
        # Get current image path
        img_path = current_images[self.current_image_idx]
        
        # Load and display image
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            self.ax.clear()
            self.ax.imshow(img_rgb)
            self.ax.axis('off')
            
            # Create detailed title
            img_filename = os.path.basename(img_path)
            title = f"{current_class} - Image {self.current_image_idx + 1}/{len(current_images)}\n"
            title += f"File: {img_filename}\n"
            title += f"Class {self.current_class_idx + 1}/{len(self.class_names)}"
            
            self.ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            
            # Add navigation instructions
            instructions = "Navigation: ← → (images) | ↑ ↓ (classes) | 'q' (quit) | 'a' (auto-scroll)"
            self.fig.text(0.5, 0.92, instructions, ha='center', fontsize=10, style='italic')
            
        else:
            self.ax.clear()
            self.ax.text(0.5, 0.5, f'Error loading image:\n{img_filename}', 
                        ha='center', va='center', transform=self.ax.transAxes, fontsize=12)
        
        plt.draw()
    
    def next_image(self, event=None):
        """Go to next image"""
        if not self.class_names:
            return
            
        current_class = self.class_names[self.current_class_idx]
        max_images = len(self.all_images[current_class])
        
        self.current_image_idx += 1
        if self.current_image_idx >= max_images:
            # Move to next class
            self.current_image_idx = 0
            self.current_class_idx = (self.current_class_idx + 1) % len(self.class_names)
        
        self.display_current_image()
    
    def prev_image(self, event=None):
        """Go to previous image"""
        if not self.class_names:
            return
            
        self.current_image_idx -= 1
        if self.current_image_idx < 0:
            # Move to previous class
            self.current_class_idx = (self.current_class_idx - 1) % len(self.class_names)
            current_class = self.class_names[self.current_class_idx]
            self.current_image_idx = len(self.all_images[current_class]) - 1
        
        self.display_current_image()
    
    def next_class(self, event=None):
        """Go to next class"""
        if not self.class_names:
            return
            
        self.current_class_idx = (self.current_class_idx + 1) % len(self.class_names)
        self.current_image_idx = 0
        self.display_current_image()
    
    def prev_class(self, event=None):
        """Go to previous class"""
        if not self.class_names:
            return
            
        self.current_class_idx = (self.current_class_idx - 1) % len(self.class_names)
        self.current_image_idx = 0
        self.display_current_image()
    
    def auto_scroll(self, event=None):
        """Auto-scroll through all images"""
        print("Starting auto-scroll... Press any key to stop")
        if hasattr(self, 'auto_scrolling'):
            self.auto_scrolling = not self.auto_scrolling
        else:
            self.auto_scrolling = True
            
        if self.auto_scrolling:
            self.auto_advance()
    
    def auto_advance(self):
        """Auto advance to next image"""
        if hasattr(self, 'auto_scrolling') and self.auto_scrolling:
            self.next_image()
            # Use timer for next advance
            self.fig.canvas.draw()
            plt.pause(1.5)  # Pause for 1.5 seconds
            if hasattr(self, 'auto_scrolling') and self.auto_scrolling:
                self.auto_advance()
    
    def quit_gallery(self, event=None):
        """Quit the gallery"""
        plt.close('all')
    
    def on_key_press(self, event):
        """Handle keyboard navigation"""
        if hasattr(self, 'auto_scrolling'):
            self.auto_scrolling = False  # Stop auto-scroll on any key
            
        if event.key == 'right':
            self.next_image()
        elif event.key == 'left':
            self.prev_image()
        elif event.key == 'up':
            self.prev_class()
        elif event.key == 'down':
            self.next_class()
        elif event.key == 'q':
            self.quit_gallery()
        elif event.key == 'a':
            self.auto_scroll()
        elif event.key == 's':
            self.save_current_image_info()
    
    def save_current_image_info(self):
        """Save information about current image"""
        if not self.class_names:
            return
            
        current_class = self.class_names[self.current_class_idx]
        current_images = self.all_images[current_class]
        img_path = current_images[self.current_image_idx]
        
        info = f"Current Image Info:\n"
        info += f"Class: {current_class}\n"
        info += f"File: {os.path.basename(img_path)}\n"
        info += f"Full Path: {img_path}\n"
        info += f"Image {self.current_image_idx + 1} of {len(current_images)} in this class\n"
        
        with open('current_image_info.txt', 'w') as f:
            f.write(info)
        print("💾 Image info saved to 'current_image_info.txt'")
    
    def show(self):
        """Start the gallery"""
        print("\n🖼️  Starting Interactive Image Gallery...")
        print("📖 Controls:")
        print("   ← → : Navigate between images")
        print("   ↑ ↓ : Navigate between classes") 
        print("   'a' : Auto-scroll through all images")
        print("   's' : Save current image info")
        print("   'q' : Quit gallery")
        print("   Or use the buttons at the bottom")
        print("\nEnjoy browsing your images! 🎨")
        
        plt.show()

if __name__ == "__main__":
    gallery = ImageGallery()
    gallery.show() 
