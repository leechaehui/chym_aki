import os
import json

root_dir = 'C:/team/chym_aki/Pathology_model/results'
output = []

for root, dirs, files in os.walk(root_dir):
    for file in files:
        if file.endswith('.json'):
            filepath = os.path.join(root, file)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    output.append(f"\n======================================")
                    output.append(f"FILE: {filepath}")
                    output.append(f"======================================")
                    output.append(json.dumps(data, indent=2, ensure_ascii=False)[:1000] + "\n... (truncated if too long)")
            except Exception as e:
                pass

with open('C:/team/chym_aki/all_json_dump.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(output))

print("Dumped all json experiments.")
