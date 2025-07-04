import os
import sys

# Add folder names to ignore here
ignore_folders = {'logs', '__pycache__', 'build', 'dist', 'venv'}

def print_tree(current_dir, depth=0, ignore_folders=None):
    if ignore_folders is None:
        ignore_folders = set()
    indent = '    ' * depth
    print(indent + os.path.basename(current_dir) + '/')
    try:
        entries = sorted(os.listdir(current_dir))  # Sort entries alphabetically
    except PermissionError:
        print(indent + '    [Permission Denied]')
        return
    for entry in entries:
        full_path = os.path.join(current_dir, entry)
        if os.path.isdir(full_path):
            if entry not in ignore_folders:
                print_tree(full_path, depth + 1, ignore_folders)
        else:
            print(indent + '    ' + entry)

if __name__ == '__main__':
    # Use command-line argument for root directory, or default to current directory
    if len(sys.argv) > 1:
        root_dir = sys.argv[1]
    else:
        root_dir = os.getcwd()
    print_tree("src", ignore_folders=ignore_folders)