import json
import re

file_path = r'c:\Users\ASUS\Downloads\DL_assignment\cadi_ai\CADI_AI_Final.ipynb'
try:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
except FileNotFoundError:
    pass

for cell in data['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            line = re.sub(r"os\.makedirs\([\'\"]imgs[\'\"]", "os.makedirs('img'", line)
            line = re.sub(r"savefig\([\'\"]imgs/", "savefig('img/", line)
            line = re.sub(r"imgs directory", "img directory", line)
            new_source.append(line)
        cell['source'] = new_source

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=1)

print("Updated notebook images from imgs/ to img/!")
