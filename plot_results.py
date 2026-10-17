"""Create a dependency-free SVG summary chart from results/results.json."""
import json
from pathlib import Path

data = json.loads(Path('results/results.json').read_text(encoding='utf-8'))
names = list(data['models'])
colors = ['#3657A7', '#D2783B', '#2F8B77', '#8B63A8']
panels = [
    ('Test cross-entropy loss (lower is better)', 'test_loss', False),
    ('Test accuracy (higher is better)', 'test_accuracy', True),
    ('Training time in seconds (lower is faster)', 'training_seconds', False),
]
w, h = 960, 600
parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
         '<rect width="100%" height="100%" fill="#ffffff"/>',
         '<style>text{font-family:Arial,sans-serif;fill:#182230}.title{font-size:24px;font-weight:bold}.label{font-size:15px}.tick{font-size:12px;fill:#667085}</style>',
         '<text x="48" y="38" class="title">Tiny Shakespeare Word Prediction Model Comparison</text>']
for pi, (title, key, _) in enumerate(panels):
    top = 75 + pi * 170
    values = [float(data['models'][n][key]) for n in names]
    vmax = max(values) * 1.12 or 1
    left, width = 270, 620
    parts.append(f'<text x="48" y="{top+20}" class="label" font-weight="bold">{title}</text>')
    for i, (name, val) in enumerate(zip(names, values)):
        y = top + 40 + i * 28
        barw = width * val / vmax
        shown = f'{val*100:.1f}%' if key == 'test_accuracy' else f'{val:.2f}' if key == 'test_loss' else f'{val:.1f}s'
        parts.extend([f'<text x="48" y="{y+15}" class="label">{name}</text>',
                      f'<rect x="{left}" y="{y}" width="{barw:.1f}" height="18" rx="3" fill="{colors[i]}"/>',
                      f'<text x="{left+barw+8:.1f}" y="{y+15}" class="label">{shown}</text>'])
parts.append('<text x="48" y="585" class="tick">Run: 30,000 tokens · 80/10/10 split · 8 epochs · CPU · seed 42</text>')
parts.append('</svg>')
Path('results/metrics_comparison.svg').write_text('\n'.join(parts), encoding='utf-8')
print('Wrote results/metrics_comparison.svg')
