import os

root_dir = 'C:/team/chym_aki'
exclude_dirs = {'.venv', 'node_modules', '.git', 'frontend', 'backend', '.vscode', '.pytest_cache'}

found_files = []

for root, dirs, files in os.walk(root_dir):
    dirs[:] = [d for d in dirs if d not in exclude_dirs]
    for file in files:
        if file.endswith(('.csv', '.json', '.log', '.txt')):
            # exclude some massive uninteresting files
            if file in ['package-lock.json', 'package.json', 'diff.txt']:
                continue
            filepath = os.path.join(root, file)
            # Get file size in MB
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            if size_mb < 50:  # only look at files under 50MB
                found_files.append(filepath)

print(f"Found {len(found_files)} potential experiment files.")
for f in found_files:
    print(f)
