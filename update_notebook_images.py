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
            # replace plt.savefig('figX or savefig("figX or savefig('cadi_ai/figX
            line = re.sub(r'savefig\([\'\"](cadi_ai/)?(fig[^\'\"]+)[\'\"]', r"savefig('imgs/\2'", line)
            
            # handle the last match: plt.savefig("CBAM_YOLOv8n_Architecture.png"
            line = re.sub(r'savefig\([\'\"](CBAM_YOLOv8n_Architecture\.png)[\'\"]', r"savefig('imgs/\1'", line)
            
            new_source.append(line)
        cell['source'] = new_source

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=1)

print('Updated notebook images save locations!')
