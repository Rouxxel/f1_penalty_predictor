"""
This script recursively scans the project starting from the resolved root directory 
and deletes target folders matching the specified names (e.g., '__pycache__', 'logs').

### How It Works:
1. **Directory Climbing**: When executed, the script climbs up directory levels starting 
    from the file's current location (`__file__`) until it finds a parent folder whose 
    name matches `ROOT_FOLDER`. This ensures it stays within and operates precisely on 
    the project root without leaking into outer folders (like C:\\ or user directories).
2. **Recursive Traversal**: Starting from that matched root folder, it walks recursively 
    down through all subdirectories (using `os.walk`) to locate and completely delete 
    any folder matching the entries specified in `FOLDERS_TO_REMOVE`.

### Variables & Setup:
- **`ROOT_FOLDER` (string)**:
    Set this to the exact name of your project's root directory (e.g., `"drone_battery_performance_prediction"`).
  *Note: The script will fail to run if this is left empty, serving as a safety lock.*

- **`FOLDERS_TO_REMOVE` (list of strings)**:
    Specify the names of directories you want recursively deleted from the project.
    For example, to clean up Python caching files, logs, and build artifacts:
    ```python
    FOLDERS_TO_REMOVE = ["__pycache__", "logs", ".pytest_cache", "build", "dist"]
    ```
    If left as an empty list `[]`, it defaults to removing `["__pycache__", "logs"]`.

### Usage:
1. Open this script and set the variables:
    ```python
    ROOT_FOLDER = "your_root_folder_name"
    FOLDERS_TO_REMOVE = ["__pycache__", "logs"]
    ```
2. Run the script:
    ```bash
    python scripts/pycache_n_logs_deleter.py
    ```
"""

import os
import sys
import shutil

# FORCE USER TO SET THE ROOT FOLDER NAME (e.g. "drone_battery_performance_prediction")
ROOT_FOLDER = "python-various"  # <-- Set this to your project's root folder name (case-insensitive)
FOLDERS_TO_REMOVE = ["__pycache__", "logs", ".pytest_cache", "build", "dist"]

def find_project_root_by_name(start_path: str, target_name: str) -> str:
    """
    Climbs up directories starting from start_path until it finds a directory
    whose folder name matches target_name (case-insensitive).
    """
    current = os.path.abspath(start_path)
    if os.path.isfile(current):
        current = os.path.dirname(current)
        
    target_lower = target_name.strip().lower()
    while True:
        if os.path.basename(current).lower() == target_lower:
            return current
        
        parent = os.path.dirname(current)
        if parent == current:  # Reached the root of the file system
            break
        current = parent
        
    print(f"Error: Could not find any parent directory named '{target_name}' starting from '{os.path.abspath(start_path)}'.")
    sys.exit(1)

def remove_folders(root_dir: str, folders_to_remove: list = ["__pycache__", "logs"]):
    """
    Removes folders from the specified root directory and its subdirectories.
    by default it will remove __pycache__ and logs folders
    """  
    # Check if directory exists
    if not os.path.exists(root_dir):
        print(f"Error: The specified root directory does not exist: {root_dir}")
        sys.exit(1)

    # Folders to delete
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for folder in folders_to_remove:
            if folder in dirnames:
                folder_path = os.path.join(dirpath, folder)
                try:
                    shutil.rmtree(folder_path)
                    print(f"Deleted: {folder_path}")
                except Exception as e:
                    print(f"Failed to delete {folder_path}: {e}")

if __name__ == "__main__":
    if not ROOT_FOLDER:
        print("Error: ROOT_FOLDER is not set! Please open this script and specify your project's root folder name in the 'ROOT_FOLDER' variable.")
        sys.exit(1)
        
    # Find the actual root directory by climbing up from the script's location
    actual_root = find_project_root_by_name(__file__, ROOT_FOLDER)
    
    # Use user-specified folders to remove, otherwise default to ["__pycache__", "logs"]
    targets = FOLDERS_TO_REMOVE if FOLDERS_TO_REMOVE else ["__pycache__", "logs", "obj", "bin", ".pytest_cache", "build", "dist"]
    remove_folders(actual_root, targets)

