import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import argparse
from unity_lidar_wrapper import LiDARWrapper


class AdvancedLiDARVisualizer:
    
    def __init__(self, wrapper=None, csv_file=None):
        """Initialize with either a LiDAR wrapper or CSV file"""
        self.wrapper = wrapper
        self.csv_file = csv_file
        self.scan_data = None
        self.cartesian_points = None
        self.processed_points = None
        self.depth_colors = None
        
        self.color_schemes = {
            'depth_green_blue': 'viridis',
            'thermal': 'plasma',
            'cool_warm': 'coolwarm',
            'jet': 'jet',
            'custom_depth': self._create_custom_colormap()
        }
        
        self.render_config = {
            'point_size': 1.0,
            'alpha': 0.8,
            'background_color': 'black',
            'dpi': 300,
            'remove_outliers': True,
            'smooth_surfaces': True,
            'density_filter': True
        }
    
    def _create_custom_colormap(self):
        """Create custom colormap from green to blue"""
        from matplotlib.colors import LinearSegmentedColormap
        
        colors = ['#00ff00', '#80ff00', '#ffff00', '#ff8000', '#ff0000', '#8000ff', '#0080ff', '#0000ff']
        n_bins = 256
        return LinearSegmentedColormap.from_list('custom_depth', colors, N=n_bins)
    
    def load_data(self):
        """Load LiDAR data from wrapper or CSV file"""
        print("Loading LiDAR data...")
        
        if self.wrapper:
            if not self.wrapper.load_data():
                print("Failed to load data from Unity wrapper")
                return False
            self.scan_data = self.wrapper.scan_data.copy()
            print(f"Loaded {len(self.scan_data)} points from Unity wrapper")
            
        elif self.csv_file:
            try:
                self.scan_data = pd.read_csv(self.csv_file)
                print(f"Loaded {len(self.scan_data)} points from CSV file")
            except Exception as e:
                print(f"Error loading CSV file: {e}")
                return False
        else:
            print("No data source provided")
            return False
        
        required_cols = ['distance', 'theta', 'phi']
        if not all(col in self.scan_data.columns for col in required_cols):
            print(f"Missing required columns. Found: {list(self.scan_data.columns)}")
            return False
        
        return True
    
    def preprocess_data(self):
        """Preprocess and convert to Cartesian coordinates"""
        if self.scan_data is None:
            print("No data loaded. Call load_data() first.")
            return False
        
        print("Preprocessing data...")
        
        df = self.scan_data.copy()
        initial_count = len(df)
        
        df = df.dropna()
        df = df[(df['distance'] > 0) & (df['distance'] < 100)]
        df = df[(df['theta'] >= 0) & (df['theta'] <= 2*np.pi)]
        df = df[(df['phi'] >= 0) & (df['phi'] <= np.pi)]
        
        print(f"Filtered data: {initial_count} -> {len(df)} points ({len(df)/initial_count*100:.1f}% retained)")
        
        theta = df['theta'].values
        phi = df['phi'].values
        distance = df['distance'].values
        
        x = distance * np.cos(theta) * np.sin(phi)
        y = distance * np.cos(phi)
        z = distance * np.sin(theta) * np.sin(phi)
        
        self.cartesian_points = np.column_stack((x, y, z, distance))
        self.processed_points = df
        
        if self.render_config['remove_outliers']:
            self._remove_outliers()
        
        print(f"Cartesian conversion complete: {len(self.cartesian_points)} points")
        return True
    
    def _remove_outliers(self):
        """Remove statistical outliers from point cloud"""
        if self.cartesian_points is None:
            return
        
        print("Removing outliers...")
        xyz = self.cartesian_points[:, :3]
        
        scaler = StandardScaler()
        xyz_scaled = scaler.fit_transform(xyz)
        
        n_points = len(xyz_scaled)
        eps = 0.1 if n_points > 10000 else 0.2
        
        clustering = DBSCAN(eps=eps, min_samples=10).fit(xyz_scaled)
        labels = clustering.labels_
        
        main_cluster_mask = labels != -1
        self.cartesian_points = self.cartesian_points[main_cluster_mask]
        
        print(f"Outlier removal: {np.sum(~main_cluster_mask)} outliers removed")
    
    def create_3d_visualization(self, style='professional', save_path=None, interactive=True):
        """Create 3D visualization"""
        if self.cartesian_points is None:
            print("No processed data available. Run preprocess_data() first.")
            return None
        
        print("Creating 3D visualization...")
        
        x, y, z, distances = self.cartesian_points.T
        
        if interactive:
            return self._create_plotly_3d(x, y, z, distances, style, save_path)
        else:
            return self._create_matplotlib_3d(x, y, z, distances, style, save_path)
    
    def _create_plotly_3d(self, x, y, z, distances, style, save_path):
        """Create interactive 3D visualization with Plotly"""
        print("Generating interactive 3D plot...")
        
        fig = go.Figure(data=[go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=2,
                color=distances,
                colorscale='Viridis',
                opacity=0.8,
                colorbar=dict(
                    title="Distance (m)",
                    titleside="right",
                    tickmode="linear",
                    tick0=0,
                    dtick=1
                )
            ),
            text=[f'Distance: {d:.2f}m' for d in distances],
            hovertemplate='<b>Position</b><br>' +
                         'X: %{x:.2f}m<br>' +
                         'Y: %{y:.2f}m<br>' +
                         'Z: %{z:.2f}m<br>' +
                         '%{text}<extra></extra>'
        )])
        
        fig.update_layout(
            title={
                'text': 'LiDAR 3D Point Cloud - Indoor Environment',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': 'white'}
            },
            scene=dict(
                xaxis=dict(
                    title='X (meters)',
                    gridcolor='rgb(50,50,50)',
                    zerolinecolor='rgb(50,50,50)',
                    color='white'
                ),
                yaxis=dict(
                    title='Y (meters)',
                    gridcolor='rgb(50,50,50)',
                    zerolinecolor='rgb(50,50,50)',
                    color='white'
                ),
                zaxis=dict(
                    title='Z (meters)',
                    gridcolor='rgb(50,50,50)',
                    zerolinecolor='rgb(50,50,50)',
                    color='white'
                ),
                bgcolor='black',
                aspectmode='data'
            ),
            paper_bgcolor='black',
            plot_bgcolor='black',
            font=dict(color='white'),
            width=1200,
            height=800
        )
        
        if save_path:
            html_path = save_path.replace('.png', '.html') if save_path.endswith('.png') else save_path + '.html'
            fig.write_html(html_path)
            print(f"Interactive 3D plot saved to {html_path}")
            
            if save_path.endswith('.png') or save_path.endswith('.jpg'):
                fig.write_image(save_path, width=1200, height=800)
                print(f"Static 3D plot saved to {save_path}")
        
        return fig
    
    def _create_matplotlib_3d(self, x, y, z, distances, style, save_path):
        """Create static 3D visualization with Matplotlib"""
        print("Generating static 3D plot...")
        
        fig = plt.figure(figsize=(15, 12), facecolor='black')
        ax = fig.add_subplot(111, projection='3d', facecolor='black')
        
        scatter = ax.scatter(x, y, z, 
                           c=distances, 
                           cmap='viridis',
                           s=self.render_config['point_size'],
                           alpha=self.render_config['alpha'],
                           edgecolors='none')
        
        ax.set_facecolor('black')
        fig.patch.set_facecolor('black')
        
        ax.set_xlabel('X (meters)', color='white', fontsize=12)
        ax.set_ylabel('Y (meters)', color='white', fontsize=12)
        ax.set_zlabel('Z (meters)', color='white', fontsize=12)
        ax.set_title('LiDAR 3D Point Cloud - Indoor Environment', 
                    color='white', fontsize=16, pad=20)
        
        ax.grid(True, alpha=0.3)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('gray')
        ax.yaxis.pane.set_edgecolor('gray')
        ax.zaxis.pane.set_edgecolor('gray')
        ax.xaxis.pane.set_alpha(0.1)
        ax.yaxis.pane.set_alpha(0.1)
        ax.zaxis.pane.set_alpha(0.1)
        
        ax.tick_params(colors='white', labelsize=10)
        
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, aspect=20)
        cbar.set_label('Distance (meters)', rotation=270, labelpad=15, color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        cbar.ax.yaxis.label.set_color('white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        
        ax.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, 
                       dpi=self.render_config['dpi'], 
                       bbox_inches='tight',
                       facecolor='black',
                       edgecolor='none')
            print(f"3D visualization saved to {save_path}")
        
        return fig
    
    def create_top_down_view(self, save_path=None):
        """Create top-down 2D view"""
        if self.cartesian_points is None:
            print("No processed data available.")
            return None
        
        print("Creating top-down view...")
        
        x, y, z, distances = self.cartesian_points.T
        
        fig, ax = plt.subplots(figsize=(12, 12), facecolor='black')
        ax.set_facecolor('black')
        
        scatter = ax.scatter(x, y, 
                           c=distances, 
                           cmap='viridis',
                           s=1, 
                           alpha=0.8,
                           edgecolors='none')
        
        ax.set_xlabel('X (meters)', color='white', fontsize=12)
        ax.set_ylabel('Y (meters)', color='white', fontsize=12)
        ax.set_title('LiDAR Top-Down View', color='white', fontsize=16)
        ax.grid(True, alpha=0.3, color='gray')
        ax.tick_params(colors='white')
        ax.set_aspect('equal')
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Distance (meters)', rotation=270, labelpad=15, color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        cbar.ax.yaxis.label.set_color('white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, 
                       dpi=self.render_config['dpi'], 
                       bbox_inches='tight',
                       facecolor='black')
            print(f"Top-down view saved to {save_path}")
        
        return fig
    
    def create_analysis_dashboard(self, output_dir="lidar_visualization_output"):
        """Create comprehensive analysis dashboard"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print(f"Creating comprehensive analysis dashboard in {output_dir}...")
        
        print("1. Creating main 3D visualization...")
        self.create_3d_visualization(
            style='professional',
            save_path=f"{output_dir}/lidar_3d_main.png",
            interactive=False
        )
        plt.show()
        
        fig_interactive = self.create_3d_visualization(
            style='professional',
            save_path=f"{output_dir}/lidar_3d_interactive",
            interactive=True
        )
        
        print("2. Creating top-down view...")
        self.create_top_down_view(f"{output_dir}/lidar_topdown.png")
        plt.show()
        
        print("3. Creating perspective views...")
        self._create_multiple_perspectives(output_dir)
        
        print("4. Creating distance analysis...")
        self._create_distance_analysis(output_dir)
        
        print("5. Creating density heatmap...")
        self._create_density_heatmap(output_dir)
        
        print("6. Generating statistics...")
        self._generate_statistics_report(output_dir)
        
        print(f"Complete analysis dashboard created in {output_dir}/")
        return True
    
    def _create_multiple_perspectives(self, output_dir):
        """Create multiple viewing perspectives"""
        if self.cartesian_points is None:
            return
        
        x, y, z, distances = self.cartesian_points.T
        
        perspectives = [
            ('front', 0, 0),
            ('side', 0, 90),
            ('diagonal', 30, 45),
            ('overhead', 90, 0)
        ]
        
        for name, elev, azim in perspectives:
            fig = plt.figure(figsize=(10, 8), facecolor='black')
            ax = fig.add_subplot(111, projection='3d', facecolor='black')
            
            scatter = ax.scatter(x, y, z, c=distances, cmap='viridis', s=1, alpha=0.8)
            
            ax.set_xlabel('X (meters)', color='white')
            ax.set_ylabel('Y (meters)', color='white')
            ax.set_zlabel('Z (meters)', color='white')
            ax.set_title(f'LiDAR {name.title()} View', color='white', fontsize=14)
            ax.tick_params(colors='white')
            ax.view_init(elev=elev, azim=azim)
            
            ax.grid(True, alpha=0.3)
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/lidar_perspective_{name}.png", 
                       dpi=300, bbox_inches='tight', facecolor='black')
            plt.close()
    
    def _create_distance_analysis(self, output_dir):
        """Create distance distribution analysis"""
        if self.cartesian_points is None:
            return
        
        distances = self.cartesian_points[:, 3]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), facecolor='black')
        
        ax1.hist(distances, bins=50, alpha=0.7, color='cyan', edgecolor='white')
        ax1.set_xlabel('Distance (meters)', color='white')
        ax1.set_ylabel('Frequency', color='white')
        ax1.set_title('Distance Distribution', color='white')
        ax1.grid(True, alpha=0.3)
        ax1.set_facecolor('black')
        ax1.tick_params(colors='white')
        
        y_coords = self.cartesian_points[:, 1]
        ax2.scatter(distances, y_coords, c=distances, cmap='viridis', s=1, alpha=0.6)
        ax2.set_xlabel('Distance (meters)', color='white')
        ax2.set_ylabel('Height (meters)', color='white')
        ax2.set_title('Distance vs Height', color='white')
        ax2.grid(True, alpha=0.3)
        ax2.set_facecolor('black')
        ax2.tick_params(colors='white')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/distance_analysis.png", 
                   dpi=300, bbox_inches='tight', facecolor='black')
        plt.close()
    
    def _create_density_heatmap(self, output_dir):
        """Create point density heatmap"""
        if self.cartesian_points is None:
            return
        
        x, y = self.cartesian_points[:, 0], self.cartesian_points[:, 2]
        
        fig, ax = plt.subplots(figsize=(10, 8), facecolor='black')
        ax.set_facecolor('black')
        
        hist, xedges, yedges = np.histogram2d(x, y, bins=50)
        
        im = ax.imshow(hist.T, origin='lower', 
                      extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                      cmap='hot', alpha=0.8)
        
        ax.set_xlabel('X (meters)', color='white')
        ax.set_ylabel('Z (meters)', color='white')
        ax.set_title('Point Density Heatmap (X-Z Plane)', color='white')
        ax.tick_params(colors='white')
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Point Density', rotation=270, labelpad=15, color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        cbar.ax.yaxis.label.set_color('white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/density_heatmap.png", 
                   dpi=300, bbox_inches='tight', facecolor='black')
        plt.close()
    
    def _generate_statistics_report(self, output_dir):
        """Generate comprehensive statistics report"""
        if self.cartesian_points is None:
            return
        
        x, y, z, distances = self.cartesian_points.T
        
        stats = {
            'total_points': len(self.cartesian_points),
            'distance_min': np.min(distances),
            'distance_max': np.max(distances),
            'distance_mean': np.mean(distances),
            'distance_std': np.std(distances),
            'spatial_extent_x': np.max(x) - np.min(x),
            'spatial_extent_y': np.max(y) - np.min(y),
            'spatial_extent_z': np.max(z) - np.min(z),
            'point_density': len(self.cartesian_points) / ((np.max(x) - np.min(x)) * (np.max(z) - np.min(z)))
        }
        
        with open(f"{output_dir}/scan_statistics.txt", 'w') as f:
            f.write("LiDAR Scan Statistics Report\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Total Points: {stats['total_points']:,}\n")
            f.write(f"Distance Range: {stats['distance_min']:.2f} - {stats['distance_max']:.2f} meters\n")
            f.write(f"Average Distance: {stats['distance_mean']:.2f} ± {stats['distance_std']:.2f} meters\n")
            f.write(f"Spatial Extent (X): {stats['spatial_extent_x']:.2f} meters\n")
            f.write(f"Spatial Extent (Y): {stats['spatial_extent_y']:.2f} meters\n")
            f.write(f"Spatial Extent (Z): {stats['spatial_extent_z']:.2f} meters\n")
            f.write(f"Point Density: {stats['point_density']:.1f} points/m²\n")
        
        print(f"Statistics report saved to {output_dir}/scan_statistics.txt")


def run_complete_pipeline(unity_exe_path=None, csv_file=None, output_dir="lidar_visualization_output"):
    """Run complete LiDAR visualization pipeline"""
    print("Starting Advanced LiDAR Visualization Pipeline")
    print("=" * 60)
    
    wrapper = None
    if unity_exe_path:
        print(f"Initializing Unity LiDAR wrapper with: {unity_exe_path}")
        wrapper = LiDARWrapper(unity_exe_path)
        if not wrapper.run_full_pipeline():
            print("Unity scan failed")
            return False
    
    visualizer = AdvancedLiDARVisualizer(wrapper=wrapper, csv_file=csv_file)
    
    if not visualizer.load_data():
        print("Failed to load data")
        return False
    
    if not visualizer.preprocess_data():
        print("Failed to preprocess data")
        return False
    
    visualizer.create_analysis_dashboard(output_dir)
    
    print("Pipeline completed successfully!")
    print(f"Results saved to: {os.path.abspath(output_dir)}")
    return True


def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(
        description="Advanced LiDAR 3D Visualization Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use with Unity executable
  python lidar_visualize.py --unity-exe "/path/to/Unity.app/Contents/MacOS/Unity"
  
  # Use with CSV file
  python lidar_visualize.py --csv-file "scan_data.csv"
  
  # Specify output directory
  python lidar_visualize.py --csv-file "data.csv" --output-dir "my_results"
        """
    )
    
    parser.add_argument(
        '--unity-exe', '--unity', '-u',
        type=str,
        help='Path to Unity executable for LiDAR simulation'
    )
    
    parser.add_argument(
        '--csv-file', '--csv', '-c',
        type=str,
        help='Path to CSV file containing LiDAR data (columns: distance, theta, phi)'
    )
    
    parser.add_argument(
        '--output-dir', '--output', '-o',
        type=str,
        default='lidar_visualization_output',
        help='Output directory for visualizations (default: lidar_visualization_output)'
    )
    
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Create interactive plots (requires plotly)'
    )
    
    args = parser.parse_args()
    
    if not args.unity_exe and not args.csv_file:
        print("Error: Must specify either --unity-exe or --csv-file")
        parser.print_help()
        sys.exit(1)
    
    success = run_complete_pipeline(
        unity_exe_path=args.unity_exe,
        csv_file=args.csv_file,
        output_dir=args.output_dir
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
