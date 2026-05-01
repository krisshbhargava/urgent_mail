import json, os

def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

majority = load('results/majority.json')
tfidf_lr = load('results/tfidf_lr.json')
deberta = load('results/deberta.json')
modernbert = load('results/modernbert.json')
MODELS = [('Majority Class', majority), ('TF-IDF + LR', tfidf_lr), ('DeBERTa-v3-base', deberta), ('ModernBERT-base', modernbert)]

def get(src, task):
    if src is None:
        return None
    return src.get(task, {}).get('test')

def pct(val):
    return f'{val*100:.1f}%' if val is not None else '—'

for task, label in [('failure_impact_type', 'Task 1: Failure Impact Type'), ('action_urgency_type', 'Task 2: Action Urgency Type')]:
    print(f'\n{label}')
    print(f'{"Model":<22}  {"Accuracy":>9}  {"Macro F1":>9}  {"Wtd F1":>9}')
    for name, src in MODELS:
        m = get(src, task)
        if m is None:
            print(f'{name:<22}  {"(not run yet)":>31}')
        else:
            print(f'{name:<22}  {pct(m["accuracy"]):>9}  {pct(m["macro_f1"]):>9}  {pct(m["weighted_f1"]):>9}')

output = {}
for task in ['failure_impact_type', 'action_urgency_type']:
    output[task] = {}
    for name, src in MODELS:
        m = get(src, task)
        if m:
            output[task][name] = {'accuracy': pct(m['accuracy']), 'macro_f1': pct(m['macro_f1']), 'weighted_f1': pct(m['weighted_f1'])}

os.makedirs('results', exist_ok=True)
with open('results/summary.json', 'w') as f:
    json.dump(output, f, indent=2)
print('\ncompleted & saved')
