from flask import Flask, render_template, request, jsonify, send_file
import os
import sys
import subprocess
import uuid
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def find_project_root():
    """find the project root by locating the Main Camera.prefab file"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    prefab_relative_path = os.path.join("..", "..", "Unity Project", "Virtual Camera", "Virtual Sensor", "Assets", "Main Camera.prefab")
    prefab_path = os.path.join(current_dir, prefab_relative_path)
    
    if os.path.exists(prefab_path):
        assets_dir = os.path.dirname(prefab_path)
        virtual_sensor_dir = os.path.dirname(assets_dir)
        virtual_camera_dir = os.path.dirname(virtual_sensor_dir)
        unity_project_dir = os.path.dirname(virtual_camera_dir)
        project_root = os.path.dirname(unity_project_dir)
        return os.path.abspath(project_root)
    
    search_dir = current_dir
    for _ in range(10):
        prefab_check = os.path.join(search_dir, "Unity Project", "Virtual Camera", "Virtual Sensor", "Assets", "Main Camera.prefab")
        if os.path.exists(prefab_check):
            return os.path.abspath(search_dir)
        
        parent_dir = os.path.dirname(search_dir)
        if parent_dir == search_dir:
            break
        search_dir = parent_dir

    return os.path.dirname(os.path.dirname(current_dir))

def get_dynamic_paths():
    """Get all dynamic paths based on project root"""
    project_root = find_project_root()
    
    return {
        'project_root': project_root,
        'lidar_results_folder': os.path.join(project_root, 'lidar_analysis_results'),
        'python_script_path': os.path.join(project_root, 'python-tool', 'lidar_visualize_new.py'),
        'unity_output_csv': os.path.join(project_root, 'Unity Project', 'Virtual Camera', 'Virtual Sensor', 'outputFile.csv')
    }

app = Flask(__name__)

PATHS = get_dynamic_paths()
LIDAR_RESULTS_FOLDER = PATHS['lidar_results_folder']
PYTHON_SCRIPT_PATH = PATHS['python_script_path']
UPLOAD_FOLDER = 'uploads'  # For manual uploads if needed
ALLOWED_EXTENSIONS = {'csv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['LIDAR_RESULTS_FOLDER'] = LIDAR_RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(LIDAR_RESULTS_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def find_analysis_script():
    """Find the analysis script - prioritize the dynamically determined path"""
    if os.path.exists(PYTHON_SCRIPT_PATH):
        return PYTHON_SCRIPT_PATH
    
    project_root = PATHS['project_root']
    possible_paths = [
        os.path.join(project_root, "python-tool", "lidar_visualize_new.py"),
        os.path.join(project_root, "lidar_visualize_new.py"),
        "lidar_visualize_new.py",
        os.path.join("python-tool", "lidar_visualize_new.py"),
        os.path.join("..", "lidar_visualize_new.py"),
        os.path.join(os.path.dirname(__file__), "lidar_visualize_new.py")
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return os.path.abspath(path)
    
    return None

def get_available_datasets():
    """Get all available datasets from the Unity LiDAR results folder"""
    datasets = []
    
    if not os.path.exists(LIDAR_RESULTS_FOLDER):
        return datasets
    
    for item in os.listdir(LIDAR_RESULTS_FOLDER):
        item_path = os.path.join(LIDAR_RESULTS_FOLDER, item)
        if os.path.isdir(item_path):
            has_results = any(
                f.endswith(('.png', '.jpg', '.html', '.txt'))
                for f in os.listdir(item_path)
                if os.path.isfile(os.path.join(item_path, f))
            )
            
            if has_results:
                mod_time = os.path.getmtime(item_path)
                datasets.append({
                    'id': item,
                    'name': f"Dataset {item}",
                    'path': item_path,
                    'modified': datetime.fromtimestamp(mod_time).isoformat(),
                    'has_results': True
                })
    
    # sort by modification time (newest first)
    datasets.sort(key=lambda x: x['modified'], reverse=True)
    return datasets

@app.route('/')
def index():
    datasets = get_available_datasets()
    return render_template('index.html', datasets=datasets)

@app.route('/api/datasets')
def list_datasets():
    """API endpoint to get available datasets"""
    datasets = get_available_datasets()
    return jsonify({'datasets': datasets})

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"lidar_data_{timestamp}_{file.filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'file_path': file_path,
            'message': 'File uploaded successfully'
        })
    
    return jsonify({'error': 'Invalid file type. Please upload a CSV file.'}), 400

@app.route('/view_dataset/<dataset_id>')
def view_dataset(dataset_id):
    """View results for an existing dataset"""
    dataset_path = os.path.join(LIDAR_RESULTS_FOLDER, dataset_id)
    
    if not os.path.exists(dataset_path):
        return jsonify({'error': 'Dataset not found'}), 404
    
    result_files = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(('.png', '.jpg', '.html', '.txt')):
                rel_path = os.path.relpath(os.path.join(root, file), dataset_path)
                result_files.append(rel_path)
    
    job_data = {
        'status': 'completed',
        'result_files': result_files,
        'filename': f'Unity Dataset {dataset_id}',
        'start_time': datetime.fromtimestamp(os.path.getctime(dataset_path)).isoformat(),
        'end_time': datetime.fromtimestamp(os.path.getmtime(dataset_path)).isoformat()
    }
    
    return render_template('results.html', job_id=dataset_id, job=job_data)

@app.route('/analyze', methods=['POST'])
def start_analysis():
    data = request.get_json()
    
    # Check if this is for an existing dataset or uploaded file
    if 'dataset_id' in data:
        # Analyzing existing Unity dataset
        dataset_id = data['dataset_id']
        dataset_path = os.path.join(LIDAR_RESULTS_FOLDER, dataset_id)
        
        if not os.path.exists(dataset_path):
            return jsonify({'error': 'Dataset not found'}), 404
        
        # Find CSV file in the dataset directory
        csv_file = None
        for file in os.listdir(dataset_path):
            if file.endswith('.csv'):
                csv_file = os.path.join(dataset_path, file)
                break
        
        if not csv_file:
            return jsonify({'error': 'No CSV file found in dataset'}), 404
        
        output_dir = dataset_path
        job_id = dataset_id
        
    else:
        if not data or 'filename' not in data:
            return jsonify({'error': 'No filename provided'}), 400
        
        filename = data['filename']
        csv_file = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if not os.path.exists(csv_file):
            return jsonify({'error': 'File not found'}), 404

        job_id = str(uuid.uuid4())
        output_dir = os.path.join(LIDAR_RESULTS_FOLDER, job_id)
        os.makedirs(output_dir, exist_ok=True)
    
    # Find analysis script
    script_path = find_analysis_script()
    if not script_path:
        return jsonify({'error': 'Analysis script not found'}), 500
    
    # Get analysis options
    options = data.get('options', {})
    interactive = options.get('interactive', True)
    
    try:
        # Prepare command
        cmd = [
            sys.executable,
            script_path,
            "--csv-file", csv_file,
            "--output-dir", output_dir
        ]
        
        if interactive:
            cmd.append("--interactive")
        
        # Run analysis synchronously
        process = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 minute timeout
        
        if process.returncode == 0:
            # Scan for generated files
            result_files = []
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    if file.endswith(('.png', '.jpg', '.html', '.txt')):
                        rel_path = os.path.relpath(os.path.join(root, file), output_dir)
                        result_files.append(rel_path)
            
            return jsonify({
                'success': True,
                'job_id': job_id,
                'message': 'Analysis completed successfully',
                'result_files': result_files,
                'output_dir': output_dir,
                'filename': os.path.basename(csv_file)
            })
        else:
            return jsonify({
                'error': f'Analysis failed: {process.stderr}',
                'stderr': process.stderr
            }), 500
    
    except subprocess.TimeoutExpired:
        return jsonify({'error': 'Analysis timed out after 5 minutes'}), 500
    except Exception as e:
        return jsonify({'error': f'Analysis error: {str(e)}'}), 500

@app.route('/results/<job_id>')
def view_results(job_id):
    # Check both the Unity results folder and manual results
    possible_paths = [
        os.path.join(LIDAR_RESULTS_FOLDER, job_id),  # Unity datasets
        os.path.join(UPLOAD_FOLDER, 'results', job_id)  # Manual upload results
    ]
    
    results_dir = None
    for path in possible_paths:
        if os.path.exists(path):
            results_dir = path
            break
    
    if not results_dir:
        return jsonify({'error': 'Results not found'}), 404
    
    # Scan for generated files
    result_files = []
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file.endswith(('.png', '.jpg', '.html', '.txt')):
                rel_path = os.path.relpath(os.path.join(root, file), results_dir)
                result_files.append(rel_path)
    
    job_data = {
        'status': 'completed',
        'result_files': result_files,
        'filename': f'Dataset {job_id}',
        'start_time': datetime.fromtimestamp(os.path.getctime(results_dir)).isoformat(),
        'end_time': datetime.fromtimestamp(os.path.getmtime(results_dir)).isoformat()
    }
    
    return render_template('results.html', job_id=job_id, job=job_data)

@app.route('/download/<job_id>/<path:filename>')
def download_file(job_id, filename):
    # Check both Unity results and manual upload results
    possible_paths = [
        os.path.join(LIDAR_RESULTS_FOLDER, job_id),
        os.path.join(UPLOAD_FOLDER, 'results', job_id)
    ]
    
    file_path = None
    for base_path in possible_paths:
        potential_path = os.path.join(base_path, filename)
        if os.path.exists(potential_path):
            file_path = potential_path
            break
    
    if not file_path:
        return jsonify({'error': 'File not found'}), 404
    
    return send_file(file_path, as_attachment=True)

@app.route('/view/<job_id>/<path:filename>')
def view_file(job_id, filename):
    # Check both Unity results and manual upload results  
    possible_paths = [
        os.path.join(LIDAR_RESULTS_FOLDER, job_id),
        os.path.join(UPLOAD_FOLDER, 'results', job_id)
    ]
    
    file_path = None
    for base_path in possible_paths:
        potential_path = os.path.join(base_path, filename)
        if os.path.exists(potential_path):
            file_path = potential_path
            break
    
    if not file_path:
        return jsonify({'error': 'File not found'}), 404
    
    return send_file(file_path)


if __name__ == '__main__':
    print("starting lidar analysis web dashboard")
    print(f"Project Root: {PATHS['project_root']}")
    print(f"LiDAR Results Folder: {LIDAR_RESULTS_FOLDER}")
    print(f"Python Script Path: {PYTHON_SCRIPT_PATH}")
    print(f"Unity Output CSV: {PATHS['unity_output_csv']}")
    print("access the dashboard at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)