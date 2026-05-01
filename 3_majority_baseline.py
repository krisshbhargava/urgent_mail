import json, os
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

train = pd.read_csv('data/train.csv')
dev = pd.read_csv('data/dev.csv')
test = pd.read_csv('data/test.csv')
print(f'Train: {len(train)} | Dev: {len(dev)} | Test: {len(test)}')

def evaluate(y_true, y_pred, split_name):
    return {'accuracy': round(accuracy_score(y_true, y_pred), 4), 'macro_f1': round(f1_score(y_true, y_pred, average='macro', zero_division=0), 4), 'weighted_f1':  round(f1_score(y_true, y_pred, average='weighted', zero_division=0), 4)}

results = {}
for task in ['failure_impact_type', 'action_urgency_type']:
    majority_class = train[task].mode()[0]
    print(f'\n{task}')
    print(f'Majority: "{majority_class}"')

    task_results = {}
    for split_name, df in [('dev', dev), ('test', test)]:
        y_true = df[task]
        y_pred = [majority_class] * len(df)
        m = evaluate(y_true, y_pred, split_name)
        task_results[split_name] = m
        print(f'[{split_name}] accuracy={m["accuracy"]},  macro_f1={m["macro_f1"]},  weighted_f1={m["weighted_f1"]}')

    print(f'\nTest classification report:')
    print(classification_report(test[task], [majority_class] * len(test), zero_division=0))
    results[task] = task_results

os.makedirs('results', exist_ok=True)
with open('results/majority.json', 'w') as f:
    json.dump(results, f, indent=2)
print('completed & saved')
