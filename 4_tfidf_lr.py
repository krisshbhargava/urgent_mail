import json, os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report

train = pd.read_csv('data/train.csv')
dev = pd.read_csv('data/dev.csv')
test = pd.read_csv('data/test.csv')
print(f'train: {len(train)}, dev: {len(dev)}, test: {len(test)}')

def evaluate(y_true, y_pred):
    return {'accuracy': round(accuracy_score(y_true, y_pred), 4), 'macro_f1': round(f1_score(y_true, y_pred, average='macro', zero_division=0), 4), 'weighted_f1': round(f1_score(y_true, y_pred, average='weighted', zero_division=0), 4),}

results = {}
for task in ['failure_impact_type', 'action_urgency_type']:
    print(f'\n{task}')
    best_c, best_score, best_pipe = None, -1, None
    for C in [0.1, 1.0, 10.0]:
        pipe = Pipeline([('tfidf', TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True)), ('clf',   LogisticRegression(class_weight='balanced', max_iter=1000, C=C, random_state=42))])
        pipe.fit(train['short_desc'], train[task])
        score = f1_score(dev[task], pipe.predict(dev['short_desc']), average='macro', zero_division=0)
        print(f'C={C:.1f}, dev macro_f1={score:.4f}')
        if score > best_score:
            best_c, best_score, best_pipe = C, score, pipe

    print(f'Best C: {best_c}')
    dev_metrics = evaluate(dev[task], best_pipe.predict(dev['short_desc']))
    test_metrics = evaluate(test[task], best_pipe.predict(test['short_desc']))
    print(f'dev accuracy={dev_metrics["accuracy"]}, macro_f1={dev_metrics["macro_f1"]}, weighted_f1={dev_metrics["weighted_f1"]}')
    print(f'test accuracy={test_metrics["accuracy"]}, macro_f1={test_metrics["macro_f1"]}, weighted_f1={test_metrics["weighted_f1"]}')
    print(f'\nTest classification report:')
    print(classification_report(test[task], best_pipe.predict(test['short_desc']), zero_division=0))
    results[task] = {'best_C': best_c, 'dev': dev_metrics, 'test': test_metrics}

os.makedirs('results', exist_ok=True)
with open('results/tfidf_lr.json', 'w') as f:
    json.dump(results, f, indent=2)
print('completed & saved')
