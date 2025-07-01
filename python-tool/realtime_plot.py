from PIL import Image
import os
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.animation import FuncAnimation

# show image, update image when there's a new image detected in the plots folder

def get_newest_image_folder(plots_folder="/Users/ayaan/coding/ASU-Summer-Internship-LiDAR-Prioject/lidar_analysis_results_new/"):
    """Get the newest image folder based on the timestamp."""
    folders = [f for f in os.listdir(plots_folder) if os.path.isdir(os.path.join(plots_folder, f))]
    if not folders:
        return None
    newest_folder = max(folders, key=lambda x: int(x.split('_')[-1]))
    return os.path.join(plots_folder, newest_folder)

def update_image(last_image_path, file_name, im, ax):
        newest_folder = get_newest_image_folder()
        if newest_folder:
            image_path = os.path.join(newest_folder, file_name)
            if os.path.exists(image_path) and image_path != last_image_path:
                try:
                    # Load and display the new image
                    img = mpimg.imread(image_path)
                    im.set_array(img)
                    im.set_extent([0, img.shape[1], img.shape[0], 0])
                    ax.set_xlim([0, img.shape[1]])
                    ax.set_ylim([img.shape[0], 0])
                    plt.draw()
                    last_image_path = image_path
                    print(f"Updated image: {image_path}")
                except Exception as e:
                    print(f"Error loading image: {e}")

def show_main_image():
    """display + update main LiDAR image realtime"""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title("Real-time Main Perspective LiDAR Visualization")
    ax.axis('off')
    
    # Initialize with empty plot
    im = ax.imshow([[0]], cmap='gray')
    plt.tight_layout()
    
    last_image_path = None
    
    # Create animation that updates every 3 seconds
    ani = FuncAnimation(fig, lambda frame: update_image(last_image_path, "lidar_3d_main.png", im, ax), interval=3000, cache_frame_data=False)

    # Show the plot window
    plt.show()
    
    return ani  # Return animation object to keep it alive

def show_density_heatmap_image():
    """display + update heatmap LiDAR image realtime"""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title("Real-time Density Heatmap Perspective LiDAR Visualization")
    ax.axis('off')
    
    # Initialize with empty plot
    im = ax.imshow([[0]], cmap='hot')
    plt.tight_layout()
    
    last_image_path = None
    
    # Create animation that updates every 3 seconds
    ani = FuncAnimation(fig, lambda frame: update_image(last_image_path, "density_heatmap.png", im, ax), interval=3000, cache_frame_data=False)
    
    # Show the plot window
    plt.show()

    return ani  # Return animation object to keep it alive

if __name__ == "__main__":
    print("Starting real-time LiDAR image viewer...")
    print("Close the window to stop the viewer.")
    ani = show_main_image()