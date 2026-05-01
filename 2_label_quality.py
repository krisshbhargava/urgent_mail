import json, os
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('bugs_sample_200.csv')
print(f'Loaded {len(df)} rows.')

#keep needed rows
annotated = df[df['human_failure_impact'].notna() & (df['human_failure_impact'].str.strip() != '') & df['human_action_urgency'].notna() & (df['human_action_urgency'].str.strip() != '')].copy()

print(f'rows: {len(annotated)} / {len(df)}')
if len(annotated) == 0:
    print('not filled in')
    raise SystemExit(0)

print('\nFailure Impact:')
acc1 = accuracy_score(annotated['human_failure_impact'], annotated['failure_impact_type'])
print(f'accuracy: {acc1:.1%}')
print(classification_report(annotated['human_failure_impact'], annotated['failure_impact_type'], zero_division=0))

print('\nAction Urgency:')
acc2 = accuracy_score(annotated['human_action_urgency'], annotated['action_urgency_type'])
print(f'accuracy: {acc2:.1%}')
print(classification_report(annotated['human_action_urgency'], annotated['action_urgency_type'], zero_division=0))

os.makedirs('results', exist_ok=True)
with open('results/label_quality.json', 'w') as f:
    json.dump({'n_annotated': len(annotated),'failure_impact_accuracy': round(acc1, 4),'action_urgency_accuracy': round(acc2, 4),}, f, indent=2)

print('\ncompleted & saved')