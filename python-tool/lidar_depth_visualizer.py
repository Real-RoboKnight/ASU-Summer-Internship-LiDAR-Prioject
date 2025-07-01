import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from unity_lidar_wrapper import LiDARWrapper


class LiDARDepthVisualizer:
    def __init__(self, wrapper):
        """Initialize LiDARWrapper instance"""
        self.wrapper = wrapper
        self.scan_data = None
        self.cartesian_points = None

    def load_data(self):
        if not self.wrapper.load_data():
            return False
        self.scan_data = self.wrapper.scan_data.copy()
        print(f"Loaded {len(self.scan_data)} points using LiDARWrapper")

        # use wrapper's coordinate conversion
        self.cartesian_points = self.wrapper.to_cartesian()
        if self.cartesian_points is not None:
            # add distance column for color coding
            distances = self.scan_data['distance'].values
            valid_mask = distances > 0
            self.cartesian_points = np.column_stack((
                self.cartesian_points,
                distances
            ))

        return True

    def print_stats(self):
        self.wrapper.get_stats()

    def save_point_cloud(self, filename="point_cloud.xyz"):
        return self.wrapper.save_point_cloud(filename)

    def create_2d_depth_plot(self, projection='xy', figsize=(12, 8), save_path=None):
        """
        Create a 2D depth plot with color-coded distances
        projection: 'xy', 'xz', or 'yz'
        """
        if self.cartesian_points is None:
            print("No data available. Load data first.")
            return None

        fig, ax = plt.subplots(figsize=figsize)

        x, y, z, distances = self.cartesian_points.T

        # Filter out zero distances for better visualization
        valid_mask = distances > 0
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        z_valid = z[valid_mask]
        distances_valid = distances[valid_mask]

        # Select projection
        if projection == 'xy':
            plot_x, plot_y = x_valid, y_valid
            xlabel, ylabel = 'X (meters)', 'Y (meters)'
            title = 'LiDAR Depth Plot - Top View (XY)'
        elif projection == 'xz':
            plot_x, plot_y = x_valid, z_valid
            xlabel, ylabel = 'X (meters)', 'Z (meters)'
            title = 'LiDAR Depth Plot - Side View (XZ)'
        elif projection == 'yz':
            plot_x, plot_y = y_valid, z_valid
            xlabel, ylabel = 'Y (meters)', 'Z (meters)'
            title = 'LiDAR Depth Plot - Front View (YZ)'
        else:
            raise ValueError("Projection must be 'xy', 'xz', or 'yz'")

        # Create scatter plot with color-coded depth
        scatter = ax.scatter(plot_x, plot_y, c=distances_valid,
                             cmap='viridis', s=1, alpha=0.6)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Distance (meters)', rotation=270, labelpad=20)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"2D depth plot saved to {save_path}")

        return fig

    def create_3d_depth_plot(self, figsize=(12, 9), save_path=None):
        """Create a 3D depth plot with color-coded distances"""
        if self.cartesian_points is None:
            print("No data available. Load data first.")
            return None

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        x, y, z, distances = self.cartesian_points.T

        # Filter out zero distances
        valid_mask = distances > 0
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        z_valid = z[valid_mask]
        distances_valid = distances[valid_mask]

        # Create 3D scatter plot
        scatter = ax.scatter(x_valid, y_valid, z_valid,
                             c=distances_valid, cmap='plasma',
                             s=1, alpha=0.6)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.6)
        cbar.set_label('Distance (meters)', rotation=270, labelpad=20)

        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_zlabel('Z (meters)')
        ax.set_title('LiDAR 3D Point Cloud - Color Coded by Distance')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"3D depth plot saved to {save_path}")

        return fig

    def create_depth_heatmap(self, bins=50, figsize=(10, 8), save_path=None):
        """Create a 2D heatmap showing depth distribution"""
        if self.cartesian_points is None:
            print("No data available. Load data first.")
            return None

        x, y, z, distances = self.cartesian_points.T

        # Filter out zero distances
        valid_mask = distances > 0
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        distances_valid = distances[valid_mask]

        fig, ax = plt.subplots(figsize=figsize)

        # Create 2D histogram
        hist, xedges, yedges = np.histogram2d(x_valid, y_valid,
                                              bins=bins,
                                              weights=distances_valid)
        counts, _, _ = np.histogram2d(x_valid, y_valid, bins=bins)

        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            avg_distances = np.divide(hist, counts,
                                      out=np.zeros_like(hist),
                                      where=counts != 0)

        # Create heatmap
        im = ax.imshow(avg_distances.T, origin='lower',
                       extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                       cmap='coolwarm', aspect='equal')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Average Distance (meters)', rotation=270, labelpad=20)

        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_title('LiDAR Depth Heatmap - Average Distance per Bin')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Depth heatmap saved to {save_path}")

        return fig

    def create_distance_histogram(self, bins=50, figsize=(10, 6), save_path=None):
        """Create histogram of distance measurements"""
        if self.scan_data is None:
            print("No data available. Load data first.")
            return None

        distances = self.scan_data['distance'].values
        valid_distances = distances[distances > 0]

        fig, ax = plt.subplots(figsize=figsize)

        ax.hist(valid_distances, bins=bins, alpha=0.7,
                color='skyblue', edgecolor='black')
        ax.set_xlabel('Distance (meters)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of LiDAR Distance Measurements')
        ax.grid(True, alpha=0.3)

        # Add statistics
        mean_dist = np.mean(valid_distances)
        median_dist = np.median(valid_distances)
        ax.axvline(mean_dist, color='red', linestyle='--',
                   label=f'Mean: {mean_dist:.2f}m')
        ax.axvline(median_dist, color='orange', linestyle='--',
                   label=f'Median: {median_dist:.2f}m')
        ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Distance histogram saved to {save_path}")

        return fig

    def create_all_plots(self, output_dir="lidar_plots"):
        """Create all visualization plots and save them"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print("Creating all LiDAR depth visualizations...")

        # Show stats first
        self.print_stats()

        # Save point cloud using wrapper method
        self.save_point_cloud(f"{output_dir}/point_cloud.xyz")

        # 2D plots for different projections
        self.create_2d_depth_plot(
            'xy', save_path=f"{output_dir}/depth_plot_xy.png")
        plt.show()

        self.create_2d_depth_plot(
            'xz', save_path=f"{output_dir}/depth_plot_xz.png")
        plt.show()

        self.create_2d_depth_plot(
            'yz', save_path=f"{output_dir}/depth_plot_yz.png")
        plt.show()

        # 3D plot
        self.create_3d_depth_plot(save_path=f"{output_dir}/depth_plot_3d.png")
        plt.show()

        # Heatmap
        self.create_depth_heatmap(save_path=f"{output_dir}/depth_heatmap.png")
        plt.show()

        # Histogram
        self.create_distance_histogram(
            save_path=f"{output_dir}/distance_histogram.png")
        plt.show()

        print(f"All plots saved to {output_dir}/")


def run_scan_and_visualize(unity_exe_path, timeout=300, output_dir="lidar_plots"):
    """
    Complete pipeline: Run Unity scan and create visualizations
    """
    if LiDARWrapper is None:
        print("Error: LiDARWrapper not available")
        return False

    print("Starting Complete LiDAR Pipeline")

    # Create wrapper and run scan
    wrapper = LiDARWrapper(unity_exe_path)

    # Run the full pipeline from your wrapper
    if not wrapper.run_full_pipeline(timeout):
        print("LiDAR scan failed")
        return False

    # Create visualizations
    print("\nCreating Visualizations")
    visualizer = LiDARDepthVisualizer(wrapper=wrapper)

    if not visualizer.load_data():
        print("Failed to load data for visualization")
        return False

    # Create all plots
    visualizer.create_all_plots(output_dir)

    print("Pipeline Complete")
    return True


def main():
    """main function w/ terminal use"""
    print("LiDAR Depth Visualizer")
    print("Usage: python lidar_depth_visualizer.py <unity_exe_path>")

    if len(sys.argv) < 2:
        sys.exit(1)

    unity_exe = sys.argv[1]
    success = run_scan_and_visualize(unity_exe)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()


# usage examples
"""
# complete pipeline w/ unity scan
python lidar_depth_visualizer.py "pathtounity"

# in program
from basic_lidar_wrapper import LiDARWrapper
from lidar_depth_visualizer import LiDARDepthVisualizer, run_scan_and_visualize

# A: complete pipeline
run_scan_and_visualize(unity_exe_path)

# B: use existing wrapper
wrapper = LiDARWrapper(unity_exe_path)
wrapper.run_full_pipeline()

visualizer = LiDARDepthVisualizer(wrapper=wrapper)
visualizer.load_data()
visualizer.create_2d_depth_plot('xy')
plt.show()
"""
