import os
import sys
import psutil
import gc
import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial, optimize, signal
from scipy.stats import gaussian_kde
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional
import json
from datetime import datetime
import argparse
import warnings
warnings.filterwarnings('ignore')
from unity_lidar_wrapper import LiDARWrapper
from sklearn.cluster import AgglomerativeClustering

@dataclass
class LiDARMetrics:
    point_density: float
    spatial_resolution: float
    coverage_uniformity: float
    noise_level: float
    dynamic_range: float
    angular_resolution_h: float
    angular_resolution_v: float
    range_accuracy: float
    beam_divergence: float
    
    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}


class AdvancedLiDARAnalyzer:
    """
    features:
    spatial analysis, clustering, segmentation, noise characterization, performance benchmarking, statistical modeling, comparative analysis
    """
    
    def __init__(self, wrapper: Optional[LiDARWrapper] = None):
        self.wrapper = wrapper
        self.raw_data = None
        self.processed_data = None
        self.cartesian_points = None
        self.metrics = None
        self.clusters = None
        self.noise_model = None
        
        # Memory monitoring
        self.memory_log = []
        self.process = psutil.Process()
        self._log_memory("Initialization", detailed=True)
        
        # analysis parameters
        self.analysis_config = {
            'clustering': {
                'dbscan_eps': 0.5,
                'dbscan_min_samples': 10,
                'kmeans_n_clusters': 8
            },
            'filtering': {
                'statistical_outlier_std': 2.0,
                'radius_outlier_neighbors': 16,
                'radius_outlier_radius': 1.0
            },
            'segmentation': {
                'plane_threshold': 0.1,
                'cylinder_threshold': 0.2,
                'sphere_threshold': 0.15
            }
        }
    
    def _log_memory(self, stage: str, detailed: bool = False):
        """Log current memory usage"""
        try:
            # Get process memory info
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)  # RSS in MB
            
            # Get system memory info
            system_memory = psutil.virtual_memory()
            system_used_mb = (system_memory.total - system_memory.available) / (1024 * 1024)
            
            log_entry = {
                'stage': stage,
                'process_memory_mb': round(memory_mb, 2),
                'system_memory_used_mb': round(system_used_mb, 2),
                'system_memory_percent': system_memory.percent
            }
            
            if detailed:
                # Add detailed object memory info
                object_memory = {}
                
                if self.raw_data is not None:
                    object_memory['raw_data_mb'] = round(sys.getsizeof(self.raw_data) / (1024 * 1024), 2)
                
                if self.processed_data is not None:
                    object_memory['processed_data_mb'] = round(sys.getsizeof(self.processed_data) / (1024 * 1024), 2)
                
                if self.cartesian_points is not None:
                    object_memory['cartesian_points_mb'] = round(self.cartesian_points.nbytes / (1024 * 1024), 2)
                
                if hasattr(self, '_cached_points_3d') and self._cached_points_3d is not None:
                    object_memory['cached_points_3d_mb'] = round(self._cached_points_3d.nbytes / (1024 * 1024), 2)
                
                if hasattr(self, '_cached_scaled_points') and self._cached_scaled_points is not None:
                    object_memory['cached_scaled_points_mb'] = round(self._cached_scaled_points.nbytes / (1024 * 1024), 2)
                
                log_entry['object_memory'] = object_memory
                log_entry['total_object_memory_mb'] = round(sum(object_memory.values()), 2)
            
            self.memory_log.append(log_entry)
            
            # Print memory usage
            print(f"🔍 Memory [{stage}]: Process={memory_mb:.1f}MB, Objects={log_entry.get('total_object_memory_mb', 0):.1f}MB")
            
            if detailed and 'object_memory' in log_entry:
                for obj, size in log_entry['object_memory'].items():
                    print(f"   📊 {obj}: {size}MB")
            
        except Exception as e:
            print(f"Warning: Memory logging failed for {stage}: {e}")
    
    def print_memory_summary(self):
        """Print a summary of memory usage throughout the process"""
        if not self.memory_log:
            print("No memory data logged")
            return
        
        print("\n" + "="*60)
        print("📈 MEMORY USAGE SUMMARY")
        print("="*60)
        
        for entry in self.memory_log:
            stage = entry['stage']
            process_mb = entry['process_memory_mb']
            obj_mb = entry.get('total_object_memory_mb', 0)
            
            print(f"{stage:25} | Process: {process_mb:8.1f}MB | Objects: {obj_mb:8.1f}MB")
        
        # Find peak usage
        peak_entry = max(self.memory_log, key=lambda x: x['process_memory_mb'])
        print(f"\n🔺 Peak Memory Usage: {peak_entry['process_memory_mb']:.1f}MB at stage '{peak_entry['stage']}'")
        
        # Calculate memory growth
        if len(self.memory_log) > 1:
            initial = self.memory_log[0]['process_memory_mb']
            final = self.memory_log[-1]['process_memory_mb']
            growth = final - initial
            print(f"📈 Total Memory Growth: {growth:+.1f}MB ({growth/initial*100:+.1f}%)")
        
        print("="*60)
    
    def load_data(self, csv_path: str = None) -> bool:
        self._log_memory("Before data loading")
        
        if self.wrapper and csv_path is None:
            if not self.wrapper.load_data():
                return False
            self.raw_data = self.wrapper.scan_data.copy()
            self._log_memory("After Unity data loading", detailed=True)
        elif csv_path:
            try:
                self.raw_data = pd.read_csv(csv_path)
                self._log_memory("After CSV loading", detailed=True)
            except Exception as e:
                print(f"Error loading CSV: {e}")
                return False
        else:
            print("No data source available")
            return False
        
        self._preprocess_data()
        self._log_memory("After preprocessing", detailed=True)
        
        self._compute_cartesian_coordinates()
        self._log_memory("After cartesian computation", detailed=True)
        
        self._compute_advanced_metrics()
        self._log_memory("After metrics computation", detailed=True)
        
        print(f"Loaded and preprocessed {len(self.processed_data)} points")
        return True
    
    def _preprocess_data(self):
        self._log_memory("Start preprocessing")
        
        df = self.raw_data.copy()
        self._log_memory("After raw data copy")
        
        # data validation
        print("Performing data validation...")
        initial_count = len(df)
        
        if initial_count == 0:
            print("Error: No data to process")
            self.processed_data = pd.DataFrame()
            return
        
        # remove invalid measurements
        df = df.dropna()
        self._log_memory("After dropna")
        
        # ensure required columns exist
        required_columns = ['distance', 'theta', 'phi']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            self.processed_data = pd.DataFrame()
            return
        print(df[df["distance"] > 0])
        # validate data ranges and remove outliers
        df = df[(df['distance'] >= 0) & (df['distance'] <= 1000)]  # can adjust range limit as needed
        df = df[(df['theta'] >= 0) & (df['theta'] <= 2*np.pi)]
        df = df[(df['phi'] >= 0) & (df['phi'] <= np.pi)]
        self._log_memory("After range validation")
        print(df[df["distance"] > 0])
        # remove infinite and NaN values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=numeric_columns)
        self._log_memory("After inf/nan removal")
        
        if len(df) == 0:
            print("Warning: All data filtered out during validation")
            self.processed_data = pd.DataFrame()
            return
        
        try:
            # add derived features with error handling
            df['range_rate'] = df.groupby(['theta', 'phi'])['distance'].diff().fillna(0)
            self._log_memory("After range_rate computation")
            
            df['angular_velocity'] = np.sqrt(df.groupby('distance')['theta'].diff().fillna(0)**2 + 
                                           df.groupby('distance')['phi'].diff().fillna(0)**2)
            self._log_memory("After angular_velocity computation")
            
            # Intensity normalization (if available)
            if 'intensity' in df.columns:
                intensity_range = df['intensity'].max() - df['intensity'].min()
                if intensity_range > 0:
                    df['intensity_normalized'] = (df['intensity'] - df['intensity'].min()) / intensity_range
                else:
                    df['intensity_normalized'] = np.zeros(len(df))
                self._log_memory("After intensity normalization")
                    
        except Exception as e:
            print(f"Warning: Error computing derived features ({e}), continuing with basic data")
        
        self.processed_data = df
        self._log_memory("End preprocessing", detailed=True)
        print(f"Preprocessing complete: {initial_count} -> {len(df)} points ({len(df)/initial_count*100:.1f}% retained)")
    
    def _compute_cartesian_coordinates(self):
        self._log_memory("Start cartesian computation")
        
        # enhanced coordinate transformation with error propagation
        if self.processed_data is None:
            return
        
        theta = self.processed_data['theta'].values
        phi = self.processed_data['phi'].values
        distance = self.processed_data['distance'].values
        print(theta, phi, distance)
        print(distance[distance != 0])
        self._log_memory("After extracting coordinate arrays")
        
        # standard conversion
        x = distance * np.cos(theta) * np.sin(phi)
        y = distance * np.cos(phi)
        z = distance * np.sin(theta) * np.sin(phi)
        print(x, y, z)
        self._log_memory("After coordinate transformation")
        
        # add measurement uncertainty estimates
        # assume 1% distance error and 0.1 degree angular error
        dist_error = distance * 0.01
        angular_error = np.deg2rad(0.1)
        self._log_memory("After error computation setup")
        
        # error propagation for Cartesian coordinates
        dx = np.sqrt((np.cos(theta) * np.sin(phi) * dist_error)**2 + 
                    (distance * (-np.sin(theta) * np.sin(phi) * angular_error))**2 +
                    (distance * np.cos(theta) * np.cos(phi) * angular_error)**2)
        
        dy = np.sqrt((np.cos(phi) * dist_error)**2 + 
                    (distance * (-np.sin(phi) * angular_error))**2)
        
        dz = np.sqrt((np.sin(theta) * np.sin(phi) * dist_error)**2 + 
                    (distance * (np.cos(theta) * np.sin(phi) * angular_error))**2 +
                    (distance * np.sin(theta) * np.cos(phi) * angular_error)**2)
        
        print(dx, dy, dz)
        self._log_memory("After error propagation")
        
        self.cartesian_points = np.column_stack((x, y, z, distance, dx, dy, dz))
        print(self.cartesian_points)
        self._log_memory("After cartesian points creation", detailed=True)
    
    def _compute_advanced_metrics(self):
        # compute comprehensive LiDAR performance metrics
        if self.cartesian_points is None:
            return
        
        x, y, z, distances = self.cartesian_points[:, 0], self.cartesian_points[:, 1], \
                           self.cartesian_points[:, 2], self.cartesian_points[:, 3]
        
        # point density analysis with error handling
        try:
            # check for minimum points and remove duplicates
            points_3d = np.column_stack((x, y, z))
            unique_points = np.unique(points_3d, axis=0)
            
            if len(unique_points) < 4:
                print("warning: not enough unique points for convex hull, using bounding box volume")
                # use bounding box volume as fallback
                x_range = np.max(x) - np.min(x)
                y_range = np.max(y) - np.min(y)
                z_range = np.max(z) - np.min(z)
                volume = x_range * y_range * z_range
            else:
                # add small random noise to prevent coplanar points
                noise_scale = np.std(unique_points, axis=0) * 1e-10
                noisy_points = unique_points + np.random.normal(0, noise_scale, unique_points.shape)
                hull = spatial.ConvexHull(noisy_points)
                volume = hull.volume
            
            point_density = len(distances) / volume if volume > 0 else 0
        except (spatial.qhull.QhullError, ValueError) as e:
            print(f"warning: Convex hull failed ({e}), using bounding box approximation")
            x_range = np.max(x) - np.min(x)
            y_range = np.max(y) - np.min(y)
            z_range = np.max(z) - np.min(z)
            volume = x_range * y_range * z_range
            point_density = len(distances) / volume if volume > 0 else 0
        
        # spatial resolution with error handling
        try:
            points_3d = np.column_stack((x, y, z))
            # remove duplicate points that could cause issues
            unique_points, unique_indices = np.unique(points_3d, axis=0, return_index=True)
            
            if len(unique_points) < 2:
                spatial_resolution = 0.1  # default value
            else:
                tree = spatial.KDTree(unique_points)
                distances_to_neighbors = tree.query(unique_points, k=min(2, len(unique_points)))[0]
                
                if distances_to_neighbors.shape[1] > 1:
                    spatial_resolution = np.median(distances_to_neighbors[:, 1])
                else:
                    spatial_resolution = 0.1  # default value
                    
        except ValueError:
            print(f"Warning: Spatial resolution computation failed ({e}), using default value")
            spatial_resolution = 0.1
        
        # using Voroni diagram w/ error handling
        try:
            # use 2d projection to avoid issues w/ 3d Voronoi
            points_2d = np.column_stack((x, y))
            unique_2d = np.unique(points_2d, axis=0)
            
            if len(unique_2d) < 3:
                print("warning: insufficient points for Voronoi diagram")
                coverage_uniformity = 0.5
            else:
                # add small noise to prevent duplicate points
                noise_scale = np.std(unique_2d, axis=0) * 1e-8
                noisy_2d = unique_2d + np.random.normal(0, noise_scale, unique_2d.shape)
                
                vor = spatial.Voronoi(noisy_2d)
                finite_regions = []
                
                for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
                    simplex = np.asarray(simplex)
                    if np.all(simplex >= 0):
                        finite_regions.append(vor.regions[vor.point_region[pointidx[0]]])
                
                areas = []
                for region in finite_regions:
                    if len(region) > 2 and -1 not in region:
                        polygon = vor.vertices[region]
                        if len(polygon) >= 3:
                            # use shoelace formula for area
                            area = 0.5 * abs(sum(polygon[i][0] * (polygon[(i+1) % len(polygon)][1] - polygon[i-1][1]) 
                                               for i in range(len(polygon))))
                            if area > 0 and area < 1e6:  # filter out unreasonably large areas
                                areas.append(area)
                
                if len(areas) > 1:
                    coverage_uniformity = 1.0 / (np.std(areas) / np.mean(areas))
                else:
                    coverage_uniformity = 0.5
                    
        except (spatial.qhull.QhullError, ValueError, ZeroDivisionError) as e:
            print(f"warning: Voronoi diagram failed ({e}), using default uniformity")
            coverage_uniformity = 0.5  # default value if Voronoi fails
        
        # noise characterization with improved filtering
        try:
            if len(distances) > 51:
                window_length = min(51, len(distances)//10*2+1)
                if window_length % 2 == 0:  # ensure odd window length
                    window_length += 1
                smoothed_distances = signal.savgol_filter(distances, window_length=window_length, polyorder=3)
            else:
                # for small datasets, use simple moving average
                smoothed_distances = np.convolve(distances, np.ones(min(5, len(distances)))/min(5, len(distances)), mode='same')
            
            noise_level = np.std(distances - smoothed_distances)
        except (ValueError, IndexError) as e:
            print(f"Warning: Noise filtering failed ({e}), using simple standard deviation")
            noise_level = np.std(distances) * 0.1  # Approximate noise level
        
        # dynamic range
        dynamic_range = np.max(distances) / np.min(distances[distances > 0]) if np.any(distances > 0) else 0
        
        # angular resolution with validation
        try:
            theta_vals = self.processed_data['theta'].values
            phi_vals = self.processed_data['phi'].values
            
            unique_theta = np.sort(np.unique(theta_vals))
            unique_phi = np.sort(np.unique(phi_vals))
            
            if len(unique_theta) > 1:
                angular_resolution_h = np.median(np.diff(unique_theta))
            else:
                angular_resolution_h = 0.01  # Default 0.01 radians
                
            if len(unique_phi) > 1:
                angular_resolution_v = np.median(np.diff(unique_phi))
            else:
                angular_resolution_v = 0.01  # Default 0.01 radians
                
        except (ValueError, KeyError) as e:
            print(f"warning: angular resolution failed ({e}), using default values")
            angular_resolution_h = 0.01
            angular_resolution_v = 0.01
        
        # range accuracy and beam divergence
        try:
            if 'distances_to_neighbors' in locals() and len(distances_to_neighbors) > 0:
                if distances_to_neighbors.shape[1] > 1:
                    range_accuracy = np.std(distances_to_neighbors[:, 1])
                else:
                    range_accuracy = np.std(distances) * 0.01  # approx
            else:
                range_accuracy = np.std(distances) * 0.01  # approx
                
            # beam divergence
            mean_distance = np.mean(distances)
            if mean_distance > 0 and spatial_resolution > 0:
                beam_divergence = spatial_resolution / mean_distance
            else:
                beam_divergence = 0.001  # default small value
                
        except (ValueError, ZeroDivisionError) as e:
            print(f"warning: range accuracy failed ({e}), using approximations")
            range_accuracy = np.std(distances) * 0.01
            beam_divergence = 0.001
        
        self.metrics = LiDARMetrics(
            point_density=point_density,
            spatial_resolution=spatial_resolution,
            coverage_uniformity=coverage_uniformity,
            noise_level=noise_level,
            dynamic_range=dynamic_range,
            angular_resolution_h=angular_resolution_h,
            angular_resolution_v=angular_resolution_v,
            range_accuracy=range_accuracy,
            beam_divergence=beam_divergence
        )
    
    def perform_advanced_clustering(self) -> Dict:
        """clustering analysis w/ multiple algs."""
        self._log_memory("Start clustering analysis")
        
        if self.cartesian_points is None:
            return {}
        
        # validate data quality first
        if not self._validate_data_quality():
            print("warning: data quality issues, clustering may be unreliable")
        
        points_3d = self.cartesian_points[:, :3]
        self._log_memory("After extracting 3D points")
        
        # remove any non-finite points
        finite_mask = np.isfinite(points_3d).all(axis=1)
        points_3d_clean = points_3d[finite_mask]
        self._log_memory("After cleaning finite points")
        
        if len(points_3d_clean) < 10:
            print("error: not enough valid points for clustering")
            return {}
        
        results = {}
        
        # DBSCAN Clustering
        print("Performing DBSCAN clustering")
        self._log_memory("Before DBSCAN")
        try:
            dbscan = DBSCAN(eps=self.analysis_config['clustering']['dbscan_eps'],
                           min_samples=self.analysis_config['clustering']['dbscan_min_samples'])
            dbscan_labels = dbscan.fit_predict(points_3d_clean)
            self._log_memory("After DBSCAN")
        except Exception as e:
            print(f"DBSCAN clustering failed: {e}")
            dbscan_labels = np.zeros(len(points_3d_clean))
        
        # K-Means Clustering
        print("Performing K-Means clustering")
        self._log_memory("Before K-Means")
        try:
            scaler = StandardScaler()
            points_scaled = scaler.fit_transform(points_3d_clean)
            self._log_memory("After scaling for K-Means")
            
            kmeans = KMeans(n_clusters=min(self.analysis_config['clustering']['kmeans_n_clusters'], len(points_3d_clean)), 
                           random_state=42, n_init=10)
            kmeans_labels = kmeans.fit_predict(points_scaled)
            self._log_memory("After K-Means")
        except Exception as e:
            print(f"K-Means clustering failed: {e}")
            kmeans_labels = np.zeros(len(points_3d_clean))
            kmeans = None
        
        # Gaussian Mixture Model clustering
        print("Performing Gaussian Mixture clustering")
        self._log_memory("Before GMM")
        try:
            from sklearn.mixture import GaussianMixture
            scaler = StandardScaler()
            points_scaled = scaler.fit_transform(points_3d_clean)
            self._log_memory("After scaling for GMM")
            
            n_components = min(6, len(points_3d_clean)//10)  # adaptive # of components
            gmm = GaussianMixture(n_components=max(1, n_components), random_state=42)
            gmm_labels = gmm.fit_predict(points_scaled)
            self._log_memory("After GMM")
        except Exception as e:
            print(f"GMM clustering failed: {e}")
            gmm_labels = np.zeros(len(points_3d_clean))
            gmm = None
        
        # hierarchical clustering
        print("Performing Hierarchical clustering")
        self._log_memory("Before Hierarchical")
        try:
            n_clusters = min(8, len(points_3d_clean)//5)  # adaptive # of clusters
            hierarchical = AgglomerativeClustering(n_clusters=max(1, n_clusters))
            hierarchical_labels = hierarchical.fit_predict(points_3d_clean)
            self._log_memory("After Hierarchical")
        except Exception as e:
            print(f"Hierarchical clustering failed: {e}")
            hierarchical_labels = np.zeros(len(points_3d_clean))
        
        results = {
            'dbscan': {
                'labels': dbscan_labels,
                'n_clusters': len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0),
                'noise_points': sum(dbscan_labels == -1)
            },
            'kmeans': {
                'labels': kmeans_labels,
                'n_clusters': self.analysis_config['clustering']['kmeans_n_clusters'],
                'centroids': kmeans.cluster_centers_,
                'inertia': kmeans.inertia_
            },
            'gmm': {
                'labels': gmm_labels,
                'n_clusters': 6,
                'bic': gmm.bic(points_scaled),
                'aic': gmm.aic(points_scaled)
            },
            'hierarchical': {
                'labels': hierarchical_labels,
                'n_clusters': 8
            }
        }
        
        self.clusters = results
        return results
    
    def detect_geometric_primitives(self) -> Dict:
        """detect planes, cylinders, and spheres in point cloud"""
        self._log_memory("Start geometric primitives detection")
        
        if self.cartesian_points is None:
            return {}
        
        points_3d = self.cartesian_points[:, :3]
        results = {}
        
        # Plane detection using RANSAC
        print("Detecting planes")
        self._log_memory("Before plane detection")
        best_plane = None
        best_inliers = 0
        
        for _ in range(1000):  # RANSAC iterations
            # Random sample
            sample_idx = np.random.choice(len(points_3d), 3, replace=False)
            sample_points = points_3d[sample_idx]
            
            # Fit plane
            v1 = sample_points[1] - sample_points[0]
            v2 = sample_points[2] - sample_points[0]
            normal = np.cross(v1, v2)
            if np.linalg.norm(normal) < 1e-6:
                continue
            normal = normal / np.linalg.norm(normal)
            
            # Count inliers
            distances = np.abs(np.dot(points_3d - sample_points[0], normal))
            inliers = distances < self.analysis_config['segmentation']['plane_threshold']
            
            if np.sum(inliers) > best_inliers:
                best_inliers = np.sum(inliers)
                best_plane = {
                    'normal': normal,
                    'point': sample_points[0],
                    'inliers': inliers,
                    'n_inliers': best_inliers
                }
        
        results['planes'] = [best_plane] if best_plane else []
        self._log_memory("After plane detection")
        
        # Sphere detection
        print("Detecting spheres")
        self._log_memory("Before sphere detection")
        def sphere_residuals(params, points):
            cx, cy, cz, r = params
            return np.sqrt((points[:, 0] - cx)**2 + (points[:, 1] - cy)**2 + 
                          (points[:, 2] - cz)**2) - r
        
        # Try multiple initializations
        best_sphere = None
        best_sphere_score = float('inf')
        
        for _ in range(10):
            # Initial guess
            center_guess = np.mean(points_3d, axis=0) + np.random.normal(0, 1, 3)
            radius_guess = np.std(np.linalg.norm(points_3d - center_guess, axis=1))
            initial_guess = [*center_guess, radius_guess]
            
            try:
                result = optimize.least_squares(sphere_residuals, initial_guess, args=(points_3d,))
                residuals = sphere_residuals(result.x, points_3d)
                inliers = np.abs(residuals) < self.analysis_config['segmentation']['sphere_threshold']
                
                if np.sum(inliers) > 50 and result.cost < best_sphere_score:
                    best_sphere_score = result.cost
                    best_sphere = {
                        'center': result.x[:3],
                        'radius': result.x[3],
                        'inliers': inliers,
                        'n_inliers': np.sum(inliers),
                        'cost': result.cost
                    }
            except:
                continue
        
        results['spheres'] = [best_sphere] if best_sphere else []
        self._log_memory("After sphere detection", detailed=True)
        
        return results
    
    def analyze_noise_characteristics(self) -> Dict:
        self._log_memory("Start noise analysis")
        
        if self.cartesian_points is None:
            return {}
        
        distances = self.cartesian_points[:, 3]
        x, y, z = self.cartesian_points[:, 0], self.cartesian_points[:, 1], self.cartesian_points[:, 2]
        self._log_memory("After extracting coordinate arrays")
        
        # Distance-dependent noise
        print("analyzing distance-dependent noise")
        distance_bins = np.linspace(distances.min(), distances.max(), 20)
        noise_vs_distance = []
        
        for i in range(len(distance_bins) - 1):
            mask = (distances >= distance_bins[i]) & (distances < distance_bins[i+1])
            if np.sum(mask) > 10:
                local_distances = distances[mask]
                local_noise = np.std(local_distances - np.mean(local_distances))
                noise_vs_distance.append((distance_bins[i], local_noise))
        self._log_memory("After distance-dependent noise analysis")
        
        # # Spatial noise correlation
        # print("analyzing spatial noise correlation")
        # tree = spatial.KDTree(np.column_stack((x, y, z)))
        # self._log_memory("After KDTree creation")
        
        # # compute local noise for each point
        # local_noise = []
        # for i, point in enumerate(np.column_stack((x, y, z))):
        #     neighbors = tree.query_ball_point(point, r=1.0)
        #     if len(neighbors) > 5:
        #         neighbor_distances = distances[neighbors]
        #         noise = np.std(neighbor_distances)
        #         local_noise.append(noise)
        #     else:
        #         local_noise.append(0)
        
        # local_noise = np.array(local_noise)
        # self._log_memory("After spatial noise correlation")
        
        # spectral analysis
        print("performing spectral noise analysis")
        sorted_distances = np.sort(distances)
        freqs = np.fft.fftfreq(len(sorted_distances))
        fft = np.fft.fft(sorted_distances - np.mean(sorted_distances))
        power_spectrum = np.abs(fft)**2
        self._log_memory("After spectral analysis")
        
        # anomaly detection
        print("detecting anomalous measurements")
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomaly_labels = iso_forest.fit_predict(self.cartesian_points[:, :4])
        self._log_memory("After anomaly detection")
        
        self.noise_model = {
            'distance_dependent': noise_vs_distance,
            # 'spatial_correlation': local_noise,
            'spectral': {
                'frequencies': freqs[:len(freqs)//2],
                'power_spectrum': power_spectrum[:len(power_spectrum)//2]
            },
            'anomalies': {
                'labels': anomaly_labels,
                'n_anomalies': np.sum(anomaly_labels == -1)
            }
        }
        
        self._log_memory("End noise analysis", detailed=True)
        return self.noise_model
    
    def create_research_report(self, output_dir: str = "lidar_research_output"):
        self._log_memory("Start research report generation")
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print("generating research report")
        
        # Perform all analyses
        
        geometric_results = self.detect_geometric_primitives()
        self._log_memory("After geometric primitives detection")
        
        noise_results = self.analyze_noise_characteristics()
        self._log_memory("After noise analysis")
        
        # Create visualizations
        self._create_advanced_visualizations(output_dir)
        self._log_memory("After visualization creation")
        
        # Generate report
        report = {
            'timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'total_points': len(self.processed_data),
                'valid_points': len(self.cartesian_points),
                'data_completeness': len(self.cartesian_points) / len(self.raw_data)
            },
            'performance_metrics': self.metrics.to_dict() if self.metrics else {},
            # 'clustering_analysis': clustering_results,
            'geometric_primitives': geometric_results,
            'noise_characteristics': {
                'overall_noise_level': self.metrics.noise_level if self.metrics else 0,
                'range_accuracy': self.metrics.range_accuracy if self.metrics else 0,
                'n_anomalies': noise_results.get('anomalies', {}).get('n_anomalies', 0)
            }
        }
        
        # Save report
        with open(f"{output_dir}/research_report.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self._log_memory("End research report generation", detailed=True)
        print(f"Research report generated in {output_dir}/")
        
        # Print memory summary at the end
        self.print_memory_summary()
        
        return report
    
    def _create_advanced_visualizations(self, output_dir: str):
        """Create visualizations"""
        self._log_memory("Start advanced visualizations")
        
        plt.style.use('seaborn-v0_8')
        
        # 1. Multi-panel overview
        fig = plt.figure(figsize=(20, 16))
        self._log_memory("After creating main figure")
        
        # Panel 1: 3D point cloud with clustering
        ax1 = fig.add_subplot(3, 3, 1, projection='3d')
        if self.clusters and 'dbscan' in self.clusters:
            labels = self.clusters['dbscan']['labels']
            unique_labels = set(labels)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
            
            for label, color in zip(unique_labels, colors):
                if label == -1:
                    color = 'black'
                mask = labels == label
                ax1.scatter(self.cartesian_points[mask, 0], 
                           self.cartesian_points[mask, 1], 
                           self.cartesian_points[mask, 2], 
                           c=[color], s=1, alpha=0.6)
        
        ax1.set_title('DBSCAN Clustering Results')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        self._log_memory("After panel 1")
        
        # Panel 2: Distance distribution
        ax2 = fig.add_subplot(3, 3, 2)
        distances = self.cartesian_points[:, 3]
        ax2.hist(distances, bins=50, alpha=0.7, density=True)
        
        # Fit and plot kernel density estimate
        print(distances)
        kde = gaussian_kde(distances)
        x_range = np.linspace(distances.min(), distances.max(), 100)
        ax2.plot(x_range, kde(x_range), 'r-', lw=2, label='KDE')
        ax2.set_xlabel('Distance (m)')
        ax2.set_ylabel('Density')
        ax2.set_title('Distance Distribution with KDE')
        ax2.legend()
        self._log_memory("After panel 2")
        
        # Panel 3: Noise characteristics
        ax3 = fig.add_subplot(3, 3, 3)
        if self.noise_model and 'distance_dependent' in self.noise_model:
            noise_data = self.noise_model['distance_dependent']
            if noise_data:
                distances_noise, noise_levels = zip(*noise_data)
                ax3.plot(distances_noise, noise_levels, 'bo-')
                ax3.set_xlabel('Distance (m)')
                ax3.set_ylabel('Noise Level (m)')
                ax3.set_title('Distance-Dependent Noise')
        
        # Panel 4: Angular resolution analysis
        ax4 = fig.add_subplot(3, 3, 4)
        theta_vals = self.processed_data['theta'].values
        phi_vals = self.processed_data['phi'].values
        
        ax4.hist2d(theta_vals, phi_vals, bins=50, cmap='viridis')
        ax4.set_xlabel('Theta (rad)')
        ax4.set_ylabel('Phi (rad)')
        ax4.set_title('Angular Sampling Pattern')
        
        # Panel 5: Spatial coverage
        ax5 = fig.add_subplot(3, 3, 5)
        x, y = self.cartesian_points[:, 0], self.cartesian_points[:, 1]
        
        try:
            # Create hexagonal binning with error handling
            # Remove any infinite or NaN values
            valid_mask = np.isfinite(x) & np.isfinite(y)
            x_clean = x[valid_mask]
            y_clean = y[valid_mask]
            
            if len(x_clean) > 0 and len(y_clean) > 0:
                hb = ax5.hexbin(x_clean, y_clean, gridsize=30, cmap='YlOrRd')
                plt.colorbar(hb, ax=ax5)
            else:
                ax5.text(0.5, 0.5, 'No valid spatial data', ha='center', va='center', transform=ax5.transAxes)
                
        except Exception as e:
            print(f"Warning: Hexbin plot failed ({e}), creating scatter plot instead")
            ax5.scatter(x, y, s=1, alpha=0.5)
            
        ax5.set_xlabel('X (m)')
        ax5.set_ylabel('Y (m)')
        ax5.set_title('Spatial Point Density')
        
        # Panel 6: PCA analysis
        ax6 = fig.add_subplot(3, 3, 6)
        pca = PCA()
        pca.fit(self.cartesian_points[:, :3])
        
        ax6.bar(range(1, 4), pca.explained_variance_ratio_)
        ax6.set_xlabel('Principal Component')
        ax6.set_ylabel('Explained Variance Ratio')
        ax6.set_title('PCA Analysis')
        ax6.set_xticks(range(1, 4))
        
        # Panel 7: Cluster validation metrics
        ax7 = fig.add_subplot(3, 3, 7)
        if self.clusters:
            methods = list(self.clusters.keys())
            n_clusters = [self.clusters[method]['n_clusters'] for method in methods]
            
            ax7.bar(methods, n_clusters)
            ax7.set_ylabel('Number of Clusters')
            ax7.set_title('Clustering Comparison')
            plt.xticks(rotation=45)
        
        # Panel 8: Range accuracy analysis
        ax8 = fig.add_subplot(3, 3, 8)
        try:
            if hasattr(self, 'cartesian_points') and len(self.cartesian_points) > 1:
                points_3d = self.cartesian_points[:, :3]
                # Remove duplicates and invalid points
                valid_mask = np.isfinite(points_3d).all(axis=1)
                clean_points = points_3d[valid_mask]
                
                if len(clean_points) > 1:
                    unique_points = np.unique(clean_points, axis=0)
                    
                    if len(unique_points) > 1:
                        tree = spatial.KDTree(unique_points)
                        distances_to_neighbors = tree.query(unique_points, k=min(2, len(unique_points)))[0]
                        
                        if distances_to_neighbors.shape[1] > 1:
                            neighbor_dists = distances_to_neighbors[:, 1]
                            ax8.hist(neighbor_dists, bins=min(50, len(neighbor_dists)//2), alpha=0.7)
                            median_dist = np.median(neighbor_dists)
                            ax8.axvline(median_dist, color='red', linestyle='--', 
                                       label=f'Median: {median_dist:.3f}m')
                            ax8.legend()
                        else:
                            ax8.text(0.5, 0.5, 'not enough data for analysis', 
                                    ha='center', va='center', transform=ax8.transAxes)
                    else:
                        ax8.text(0.5, 0.5, 'all points identical', 
                                ha='center', va='center', transform=ax8.transAxes)
                else:
                    ax8.text(0.5, 0.5, 'no valid points', 
                            ha='center', va='center', transform=ax8.transAxes)
            else:
                ax8.text(0.5, 0.5, 'no cartesian data available', 
                        ha='center', va='center', transform=ax8.transAxes)
                        
        except Exception as e:
            print(f"warning: range accuracy analysis failed ({e})")
            ax8.text(0.5, 0.5, f'Analysis failed: {str(e)[:50]}...', 
                    ha='center', va='center', transform=ax8.transAxes)
            
        ax8.set_xlabel('distance to Nearest Neighbor (m)')
        ax8.set_ylabel('frequency')
        ax8.set_title('spatial resolution analysis')
        
        # Panel 9: Performance metrics radar
        ax9 = fig.add_subplot(3, 3, 9, projection='polar')
        if self.metrics:
            metrics_dict = self.metrics.to_dict()
            # Normalize metrics for radar plot
            normalized_metrics = {}
            for key, value in metrics_dict.items():
                if value > 0:
                    normalized_metrics[key] = min(value / np.mean(list(metrics_dict.values())), 2)
                else:
                    normalized_metrics[key] = 0
            
            angles = np.linspace(0, 2*np.pi, len(normalized_metrics), endpoint=False)
            values = list(normalized_metrics.values())
            
            ax9.plot(angles, values, 'o-', linewidth=2)
            ax9.fill(angles, values, alpha=0.25)
            ax9.set_xticks(angles)
            ax9.set_xticklabels([key.replace('_', ' ').title() for key in normalized_metrics.keys()], 
                              rotation=45, ha='center')
            ax9.set_title('performance metrics overview')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/full_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        self._log_memory("After saving main visualization")
        
        # Create additional specialized plots
        self._create_clustering_comparison(output_dir)
        self._log_memory("After clustering comparison plots")
        
        self._create_noise_analysis_plots(output_dir)
        self._log_memory("After noise analysis plots")
        
        self._create_geometric_analysis_plots(output_dir)
        self._log_memory("End advanced visualizations")
    
    def _create_clustering_comparison(self, output_dir: str):
        # create detailed clustering comparison plots
        if not self.clusters:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Clustering Analysis Comparison', fontsize=16)
        
        methods = ['dbscan', 'kmeans', 'gmm', 'hierarchical']
        
        for i, method in enumerate(methods):
            if method not in self.clusters:
                continue
                
            ax = axes[i//2, i%2]
            labels = self.clusters[method]['labels']
            
            # 2D projection for visualization
            points_2d = self.cartesian_points[:, [0, 2]]  # X-Z plane
            
            scatter = ax.scatter(points_2d[:, 0], points_2d[:, 1], 
                               c=labels, cmap='tab10', s=1, alpha=0.6)
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Z (m)')
            ax.set_title(f'{method.upper()} Clustering\n({self.clusters[method]["n_clusters"]} clusters)')
            
            # Add cluster statistics
            if method == 'dbscan':
                noise_points = self.clusters[method]['noise_points']
                ax.text(0.02, 0.98, f'Noise points: {noise_points}', 
                       transform=ax.transAxes, va='top', 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/clustering_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_noise_analysis_plots(self, output_dir: str):
        """Create detailed noise analysis plots"""
        if not self.noise_model:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Comprehensive Noise Characteristics Analysis', fontsize=16)
        
        # Distance-dependent noise
        ax1 = axes[0, 0]
        if 'distance_dependent' in self.noise_model:
            noise_data = self.noise_model['distance_dependent']
            if noise_data:
                distances_noise, noise_levels = zip(*noise_data)
                ax1.plot(distances_noise, noise_levels, 'bo-', markersize=4)
                
                # Fit polynomial trend
                if len(distances_noise) > 3:
                    z = np.polyfit(distances_noise, noise_levels, 2)
                    p = np.poly1d(z)
                    x_smooth = np.linspace(min(distances_noise), max(distances_noise), 100)
                    ax1.plot(x_smooth, p(x_smooth), 'r--', alpha=0.7, label='Polynomial fit')
                    ax1.legend()
                
        ax1.set_xlabel('Distance (m)')
        ax1.set_ylabel('Noise Level (m)')
        ax1.set_title('Distance-Dependent Noise Profile')
        ax1.grid(True, alpha=0.3)
        
        # Spatial noise correlation
        ax2 = axes[0, 1]
        if 'spatial_correlation' in self.noise_model:
            local_noise = self.noise_model['spatial_correlation']
            x, y = self.cartesian_points[:, 0], self.cartesian_points[:, 1]
            
            scatter = ax2.scatter(x, y, c=local_noise, cmap='plasma', s=2, alpha=0.6)
            plt.colorbar(scatter, ax=ax2, label='Local Noise Level (m)')
            ax2.set_xlabel('X (m)')
            ax2.set_ylabel('Y (m)')
            ax2.set_title('Spatial Noise Distribution')
        
        # Spectral analysis
        ax3 = axes[1, 0]
        if 'spectral' in self.noise_model:
            freqs = self.noise_model['spectral']['frequencies']
            power = self.noise_model['spectral']['power_spectrum']
            
            ax3.loglog(freqs[1:], power[1:])  # Skip DC component
            ax3.set_xlabel('Frequency')
            ax3.set_ylabel('Power Spectral Density')
            ax3.set_title('Noise Power Spectrum')
            ax3.grid(True, alpha=0.3)
        
        # Anomaly detection results
        ax4 = axes[1, 1]
        if 'anomalies' in self.noise_model:
            labels = self.noise_model['anomalies']['labels']
            points_2d = self.cartesian_points[:, [0, 2]]
            
            normal_points = points_2d[labels == 1]
            anomaly_points = points_2d[labels == -1]
            
            ax4.scatter(normal_points[:, 0], normal_points[:, 1], 
                       c='blue', s=1, alpha=0.5, label='Normal')
            ax4.scatter(anomaly_points[:, 0], anomaly_points[:, 1], 
                       c='red', s=3, alpha=0.8, label='Anomalies')
            
            ax4.set_xlabel('X (m)')
            ax4.set_ylabel('Z (m)')
            ax4.set_title(f'Anomaly Detection\n({len(anomaly_points)} anomalies detected)')
            ax4.legend()
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/noise_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_geometric_analysis_plots(self, output_dir: str):
        """Create geometric primitive detection plots"""
        geometric_results = self.detect_geometric_primitives()
        
        fig = plt.figure(figsize=(16, 10))
        
        # 3D plot with detected primitives
        ax1 = fig.add_subplot(2, 2, (1, 3), projection='3d')
        
        # Plot all points
        x, y, z = self.cartesian_points[:, 0], self.cartesian_points[:, 1], self.cartesian_points[:, 2]
        ax1.scatter(x, y, z, c='lightgray', s=0.5, alpha=0.3)
        
        # Highlight detected planes
        if 'planes' in geometric_results:
            for i, plane in enumerate(geometric_results['planes']):
                if plane and 'inliers' in plane:
                    plane_points = self.cartesian_points[plane['inliers'], :3]
                    ax1.scatter(plane_points[:, 0], plane_points[:, 1], plane_points[:, 2], 
                               s=2, alpha=0.8, label=f'Plane {i+1}')
        
        # Highlight detected spheres
        if 'spheres' in geometric_results:
            for i, sphere in enumerate(geometric_results['spheres']):
                if sphere and 'inliers' in sphere:
                    sphere_points = self.cartesian_points[sphere['inliers'], :3]
                    ax1.scatter(sphere_points[:, 0], sphere_points[:, 1], sphere_points[:, 2], 
                               s=2, alpha=0.8, label=f'Sphere {i+1}')
                    
                    # Draw sphere wireframe
                    center = sphere['center']
                    radius = sphere['radius']
                    u = np.linspace(0, 2 * np.pi, 20)
                    v = np.linspace(0, np.pi, 20)
                    sphere_x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
                    sphere_y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
                    sphere_z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
                    ax1.plot_wireframe(sphere_x, sphere_y, sphere_z, alpha=0.3)
        
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('Detected Geometric Primitives')
        ax1.legend()
        
        # plane fitting quality
        ax2 = fig.add_subplot(2, 2, 2)
        if 'planes' in geometric_results and geometric_results['planes']:
            plane = geometric_results['planes'][0]
            if plane:
                inlier_distances = []
                outlier_distances = []
                
                normal = plane['normal']
                point_on_plane = plane['point']
                
                for i, point in enumerate(self.cartesian_points[:, :3]):
                    distance = abs(np.dot(point - point_on_plane, normal))
                    if plane['inliers'][i]:
                        inlier_distances.append(distance)
                    else:
                        outlier_distances.append(distance)
                
                ax2.hist(inlier_distances, bins=30, alpha=0.7, label='Inliers', color='green')
                ax2.hist(outlier_distances, bins=30, alpha=0.7, label='Outliers', color='red')
                ax2.axvline(self.analysis_config['segmentation']['plane_threshold'], 
                           color='black', linestyle='--', label='Threshold')
                ax2.set_xlabel('Distance to Plane (m)')
                ax2.set_ylabel('Frequency')
                ax2.set_title('Plane Fitting Quality')
                ax2.legend()
        
        # sphere fitting quality
        ax3 = fig.add_subplot(2, 2, 4)
        if 'spheres' in geometric_results and geometric_results['spheres']:
            sphere = geometric_results['spheres'][0]
            if sphere:
                center = sphere['center']
                radius = sphere['radius']
                
                residuals = []
                for point in self.cartesian_points[:, :3]:
                    distance_to_center = np.linalg.norm(point - center)
                    residual = abs(distance_to_center - radius)
                    residuals.append(residual)
                
                ax3.hist(residuals, bins=50, alpha=0.7, color='purple')
                ax3.axvline(self.analysis_config['segmentation']['sphere_threshold'], 
                           color='black', linestyle='--', label='Threshold')
                ax3.set_xlabel('Residual (m)')
                ax3.set_ylabel('Frequency')
                ax3.set_title('Sphere Fitting Quality')
                ax3.legend()
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/geometric_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def benchmark_performance(self, reference_data: Optional[np.ndarray] = None) -> Dict:
        # benchmark LiDAR performance against reference standards
        if self.cartesian_points is None:
            return {}
        
        print("Performing comprehensive performance benchmark...")
        
        benchmarks = {}
        
        # Temporal consistency (if multiple scans available)
        benchmarks['temporal_stability'] = self._analyze_temporal_stability()
        
        # Measurement precision
        benchmarks['precision_analysis'] = self._analyze_measurement_precision()
        
        # Coverage analysis
        benchmarks['coverage_analysis'] = self._analyze_coverage_completeness()
        
        # Comparative analysis (if reference provided)
        if reference_data is not None:
            benchmarks['comparative_analysis'] = self._compare_with_reference(reference_data)
        
        return benchmarks
    
    def _analyze_temporal_stability(self) -> Dict:
        """Analyze temporal stability of measurements"""
        # Simulate temporal analysis by analyzing local consistency
        tree = spatial.KDTree(self.cartesian_points[:, :3])
        
        stability_metrics = []
        for point in self.cartesian_points[:100]:  # Sample subset for efficiency
            neighbors = tree.query_ball_point(point[:3], r=0.5)
            if len(neighbors) > 5:
                neighbor_distances = self.cartesian_points[neighbors, 3]
                stability = np.std(neighbor_distances) / np.mean(neighbor_distances)
                stability_metrics.append(stability)
        
        return {
            'mean_stability': np.mean(stability_metrics),
            'stability_std': np.std(stability_metrics),
            'stable_points_ratio': np.sum(np.array(stability_metrics) < 0.1) / len(stability_metrics)
        }
    
    def _analyze_measurement_precision(self) -> Dict:
        """Analyze measurement precision using statistical methods"""
        distances = self.cartesian_points[:, 3]
        
        # Allan variance for long-term stability
        def allan_variance(data, tau_range):
            variances = []
            for tau in tau_range:
                if tau >= len(data) // 3:
                    break
                
                # Compute Allan variance for given tau
                differences = []
                for i in range(0, len(data) - 2*tau, tau):
                    diff = np.mean(data[i+tau:i+2*tau]) - np.mean(data[i:i+tau])
                    differences.append(diff**2)
                
                if differences:
                    variances.append(np.mean(differences) / 2)
                else:
                    variances.append(0)
            
            return np.array(variances)
        
        tau_range = range(1, min(100, len(distances)//10))
        allan_var = allan_variance(distances, tau_range)
        
        return {
            'allan_variance': allan_var.tolist(),
            'tau_range': list(tau_range[:len(allan_var)]),
            'short_term_precision': np.std(distances[:1000]) if len(distances) > 1000 else np.std(distances),
            'long_term_drift': np.abs(np.mean(distances[:len(distances)//2]) - 
                                     np.mean(distances[len(distances)//2:]))
        }
    
    def _analyze_coverage_completeness(self) -> Dict:
        theta_vals = self.processed_data['theta'].values
        phi_vals = self.processed_data['phi'].values
        
        # Create angular grid
        theta_bins = np.linspace(-np.pi, np.pi, 36)  # 10-degree bins
        phi_bins = np.linspace(0, np.pi, 18)  # 10-degree bins
        
        coverage_grid = np.zeros((len(theta_bins)-1, len(phi_bins)-1))
        
        for i in range(len(theta_bins)-1):
            for j in range(len(phi_bins)-1):
                mask = ((theta_vals >= theta_bins[i]) & (theta_vals < theta_bins[i+1]) &
                       (phi_vals >= phi_bins[j]) & (phi_vals < phi_bins[j+1]))
                coverage_grid[i, j] = np.sum(mask)
        
        # Calculate coverage metrics
        total_bins = coverage_grid.size
        covered_bins = np.sum(coverage_grid > 0)
        coverage_ratio = covered_bins / total_bins
        
        # Coverage uniformity
        non_zero_coverage = coverage_grid[coverage_grid > 0]
        uniformity = 1.0 / (np.std(non_zero_coverage) / np.mean(non_zero_coverage)) if len(non_zero_coverage) > 0 else 0
        
        return {
            'coverage_ratio': coverage_ratio,
            'coverage_uniformity': uniformity,
            'total_angular_bins': total_bins,
            'covered_bins': int(covered_bins),
            'coverage_grid': coverage_grid.tolist()
        }
    
    def _compare_with_reference(self, reference_data: np.ndarray) -> Dict:
        # Compare current data with reference dataset
        # Assume reference_data has same structure [x, y, z, distance]
        
        # Spatial alignment using ICP-like approach
        from sklearn.neighbors import NearestNeighbors
        
        current_points = self.cartesian_points[:, :3]
        ref_points = reference_data[:, :3]
        
        # Find correspondences
        nbrs = NearestNeighbors(n_neighbors=1).fit(ref_points)
        distances, indices = nbrs.kneighbors(current_points)
        
        # Calculate comparison metrics
        mean_spatial_error = np.mean(distances)
        rmse_spatial = np.sqrt(np.mean(distances**2))
        
        # Distance comparison for corresponding points
        current_distances = self.cartesian_points[:, 3]
        ref_distances = reference_data[indices.flatten(), 3]
        
        distance_errors = current_distances - ref_distances
        mean_distance_error = np.mean(distance_errors)
        rmse_distance = np.sqrt(np.mean(distance_errors**2))
        
        return {
            'mean_spatial_error': mean_spatial_error,
            'rmse_spatial': rmse_spatial,
            'mean_distance_error': mean_distance_error,
            'rmse_distance': rmse_distance,
            'correlation_coefficient': np.corrcoef(current_distances, ref_distances)[0, 1]
        }
    
    def _validate_data_quality(self) -> bool:
        # validate data quality to prevent geometric computation errors
        if self.cartesian_points is None or len(self.cartesian_points) == 0:
            print("Error: No cartesian points available")
            return False
        
        points_3d = self.cartesian_points[:, :3]
        
        # Check for minimum number of points
        if len(points_3d) < 4:
            print("Warning: Insufficient points for robust geometric analysis")
            return False
        
        # Check for duplicate points
        unique_points = np.unique(points_3d, axis=0)
        duplicate_ratio = 1 - len(unique_points) / len(points_3d)
        if duplicate_ratio > 0.9:
            print(f"Warning: High duplicate point ratio ({duplicate_ratio:.2%})")
        
        # Check for coplanar points (all points in same plane)
        if len(unique_points) >= 4:
            # Use PCA to check dimensionality
            try:
                pca = PCA()
                pca.fit(unique_points)
                explained_var = pca.explained_variance_ratio_
                
                # If first two components explain >99.9% of variance, points are essentially 2D
                if np.sum(explained_var[:2]) > 0.999:
                    print("Warning: Points appear to be coplanar (2D), some 3D analyses may fail")
                    
                # If first component explains >99.9%, points are essentially 1D
                if explained_var[0] > 0.999:
                    print("Warning: Points appear to be collinear (1D), most geometric analyses will fail")
                    return False
                    
            except Exception as e:
                print(f"Warning: Could not validate point dimensionality ({e})")
        
        # Check for reasonable coordinate ranges
        coord_ranges = np.ptp(unique_points, axis=0)  # peak-to-peak range
        if np.any(coord_ranges < 1e-10):
            print("Warning: Very small coordinate range detected, may cause numerical issues")
        
        # Check for infinite or NaN values
        if not np.all(np.isfinite(points_3d)):
            print("Warning: Non-finite values detected in coordinates")
            return False
        
        return True


def main():
    """
    Main function to demonstrate the Advanced LiDAR Analyzer
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Advanced LiDAR Analysis Suite for Academic Research",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python advanced_lidar_processing.py
  python advanced_lidar_processing.py --csv-file data/scan_data.csv
  python advanced_lidar_processing.py --unity-exe /path/to/unity/executable
  python advanced_lidar_processing.py --csv-file data.csv --output-dir results/
        """
    )
    
    parser.add_argument(
        '--csv-file', '--csv', '-c',
        type=str,
        help='Path to CSV file containing LiDAR data (columns: distance, theta, phi)'
    )
    
    parser.add_argument(
        '--unity-exe', '--unity', '-u',
        type=str,
        help='Path to Unity executable for LiDAR simulation'
    )
    
    parser.add_argument(
        '--output-dir', '--output', '-o',
        type=str,
        default='lidar_analysis_results',
        help='Output directory for analysis results (default: lidar_analysis_results)'
    )
    
    parser.add_argument(
        '--synthetic', '-s',
        action='store_true',
        help='Force generation of synthetic data for testing'
    )
    
    parser.add_argument(
        '--points', '-p',
        type=int,
        default=5000,
        help='Number of synthetic points to generate (default: 5000)'
    )
    
    args = parser.parse_args()

    print("Initializing analyzer...")
    
    # Initialize the analyzer
    analyzer = AdvancedLiDARAnalyzer()
    
    # Try to load data from different sources
    data_loaded = False
    
    # First priority: Use specified CSV file
    if args.csv_file and not args.synthetic:
        print(f"Attempting to load data from specified CSV file: {args.csv_file}")
        if os.path.exists(args.csv_file):
            try:
                if analyzer.load_data(args.csv_file):
                    print("Successfully loaded data from CSV file")
                    data_loaded = True
            except Exception as e:
                print(f"Failed to load from CSV: {e}")
        else:
            print(f"CSV file not found: {args.csv_file}")
    
    # Second priority: Use Unity executable if specified
    if not data_loaded and args.unity_exe and not args.synthetic:
        print(f"Attempting to use Unity executable: {args.unity_exe}")
        if os.path.exists(args.unity_exe):
            try:
                # TODO: implement Unity executable integration
                # This would require modifying the LiDARWrapper to accept executable path
                print("Unity executable integration not yet implemented")
                print("Falling back to other data sources")
            except Exception as e:
                print(f"Failed to use Unity executable: {e}")
        else:
            print(f"Unity executable not found: {args.unity_exe}")
    
    # Third priority: Try Unity wrapper if available (default behavior)
    if not data_loaded and LiDARWrapper and not args.synthetic:
        print("Attempting to load data from Unity LiDAR wrapper")
        try:
            wrapper = LiDARWrapper()
            analyzer = AdvancedLiDARAnalyzer(wrapper)
            if analyzer.load_data():
                print("successfully loaded data from Unity wrapper")
                data_loaded = True
        except Exception as e:
            print(f"failed to load from Unity wrapper: {e}")
    
    if not data_loaded:
        print("no data could be loaded or generated. Exiting.")
        return
    
    # Display basic information
    print(f"\nDataset Summary:")
    print(f"  Raw points: {len(analyzer.raw_data)}")
    print(f"  Processed points: {len(analyzer.processed_data)}")
    print(f"  Cartesian points: {len(analyzer.cartesian_points)}")
    
    # Display performance metrics
    if analyzer.metrics:
        print(f"\nPerformance Metrics:")
        metrics_dict = analyzer.metrics.to_dict()
        for key, value in metrics_dict.items():
            print(f"  {key.replace('_', ' ').title()}: {value:.4f}")

    print(f"\nrunning analysis")
    
    try:
        # Use specified output directory
        output_dir = args.output_dir
        report = analyzer.create_research_report(output_dir)
        
        print(f"\n✓ Analysis complete")
        print(f"  Results saved to: {output_dir}/")
        
        # display findings
        print(f"\nKey Findings")
        if 'clustering_analysis' in report:
            clustering = report['clustering_analysis']
            print(f"Clustering Results:")
            for method, results in clustering.items():
                print(f"  {method.upper()}: {results['n_clusters']} clusters")
        
        if 'noise_characteristics' in report:
            noise = report['noise_characteristics']
            print(f"Noise Analysis:")
            print(f"  Overall noise level: {noise.get('overall_noise_level', 0):.4f} m")
            print(f"  Anomalies detected: {noise.get('n_anomalies', 0)}")
        
    except Exception as e:
        print(f"✗ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\nAnalysis Complete")
    print(f"Results saved to: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()