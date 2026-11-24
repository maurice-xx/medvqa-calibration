import json
from pathlib import Path
D = Path('assets/data/VQAMed2019/VQAMed2019Test/VQAMed2019Test')
IMG = D / 'VQAMed2019_Test_Images'
rows = []
for line in open(D / 'VQAMed2019_Test_Questions_w_Ref_Answers.txt', encoding='utf-8', errors='ignore'):
    parts = line.strip().split('|')
    if len(parts) != 4: continue
    img, cat, q, ans = parts
    if cat not in ('plane', 'modality'): continue
    ip = IMG / (img if img.endswith(".jpg") else img + ".jpg")
    if not ip.exists(): continue
    gold = ans.lower().split(' - ')[0].strip().rstrip('.')
    rows.append({'dataset': 'vqamed2019', 'split': 'test',
                 'qid': f'vqamed:{img}:{cat}', 'image_path': str(ip),
                 'question': q.strip(), 'gold': gold,
                 'category': 'plane' if cat == 'plane' else 'vqamed_modality'})
out = Path('./data/seed_vqamed.jsonl')
with open(out, 'w', encoding='utf-8') as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + '\n')
from collections import Counter
print('sampled', len(rows), dict(Counter(r['category'] for r in rows)), '| images:', len(set(r['image_path'] for r in rows)))
