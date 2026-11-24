import json, random
from pathlib import Path
SRC = Path('assets/data/VQA-RED/VQA_RAD Dataset Public.json')
IMG_DIR = Path('assets/data/VQA-RED/VQA_RAD Image Folder')
rows = json.load(open(SRC))
out = []
for r in rows:
    at = (r.get('answer_type') or '').strip()
    ans = str(r.get('answer', '')).strip()
    img = IMG_DIR / r.get('image_name', '')
    if not img.exists():
        continue
    q = (r.get('question') or '').strip()
    if not q or len(q) > 200:
        continue
    if at == 'CLOSED' and ans.lower() in ('yes', 'no'):
        cat = 'abnormality'
    elif at == 'OPEN' and ans and len(ans.split()) <= 3 and ',' not in ans:
        cat = 'open'
    else:
        continue
    out.append({'dataset': 'vqa_rad', 'split': 'all',
                'qid': f"vqarad:{r['qid']}", 'image_path': str(img),
                'question': q, 'gold': ans, 'category': cat})
rng = random.Random(20261023)
closed = [r for r in out if r['category'] == 'abnormality']
opens = [r for r in out if r['category'] == 'open']
rng.shuffle(closed); rng.shuffle(opens)
sel = closed[:100] + opens[:50]
rng.shuffle(sel)
with open('./data/seed_vqarad.jsonl', 'w', encoding='utf-8') as f:
    for r in sel:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
from collections import Counter
print('sampled', len(sel), dict(Counter(r['category'] for r in sel)),
      '| images:', len(set(r['image_path'] for r in sel)))
