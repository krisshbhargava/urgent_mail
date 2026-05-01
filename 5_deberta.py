import json, os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

MODEL_NAME = 'microsoft/deberta-v3-base'
MAX_LEN = 128
BATCH_SIZE = 16
LR = 2e-5
EPOCHS = 5
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

#using local
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f'Device: {device}')

train = pd.read_csv('data/train.csv')
dev = pd.read_csv('data/dev.csv')
test = pd.read_csv('data/test.csv')
print(f'Train: {len(train)}, Dev: {len(dev)}, Test: {len(test)}')

class BugDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, label2id):
        self.encodings = tokenizer(list(texts), max_length=MAX_LEN, padding='max_length', truncation=True, return_tensors='pt')
        self.labels = torch.tensor([label2id[l] for l in labels], dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item


def run_task(task):
    print(f'\n{"="*60}')
    print(f'Task: {task}')
    print(f'{"="*60}')

    classes = sorted(train[task].unique())
    label2id = {c: i for i, c in enumerate(classes)}
    id2label = {i: c for c, i in label2id.items()}
    print(f'Classes: {classes}')

    weights = compute_class_weight('balanced', classes=np.array(classes), y=train[task])
    loss_weights = torch.tensor(weights, dtype=torch.float).to(device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = BugDataset(train['short_desc'], train[task], tokenizer, label2id)
    dev_ds = BugDataset(dev['short_desc'], dev[task], tokenizer, label2id)
    test_ds = BugDataset(test['short_desc'], test[task], tokenizer, label2id)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(classes), id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True,).to(device).float()
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)
    loss_fn = torch.nn.CrossEntropyLoss(weight=loss_weights)

    save_dir = f'results/checkpoints/deberta_{task}'
    os.makedirs(save_dir, exist_ok=True)
    best_dev_f1, best_epoch = -1, -1

    for epoch in range(1, EPOCHS + 1):
        #train
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**{k: v for k, v in batch.items() if k != 'labels'}).logits
            loss = loss_fn(logits, batch['labels'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        #eval
        model.eval()
        preds, truths = [], []
        with torch.no_grad():
            for batch in dev_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(**{k: v for k, v in batch.items() if k != 'labels'}).logits
                preds.extend(logits.argmax(-1).cpu().numpy())
                truths.extend(batch['labels'].cpu().numpy())

        dev_f1 = f1_score(truths, preds, average='macro', zero_division=0)
        dev_acc = accuracy_score(truths, preds)
        print(f'Epoch {epoch}/{EPOCHS}  loss={total_loss/len(train_loader):.4f}  dev_acc={dev_acc:.4f}  dev_macro_f1={dev_f1:.4f}')

        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_epoch = epoch
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)

    print(f'Best epoch: {best_epoch}  best dev macro F1: {best_dev_f1:.4f}')

    #test
    best_model = AutoModelForSequenceClassification.from_pretrained(save_dir).to(device).float()
    best_model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = best_model(**{k: v for k, v in batch.items() if k != 'labels'}).logits
            preds.extend(logits.argmax(-1).cpu().numpy())
            truths.extend(batch['labels'].cpu().numpy())

    y_true = [id2label[i] for i in truths]
    y_pred = [id2label[i] for i in preds]
    test_acc = accuracy_score(truths, preds)
    test_macro_f1    = f1_score(truths, preds, average='macro',    zero_division=0)
    test_weighted_f1 = f1_score(truths, preds, average='weighted', zero_division=0)
    print(f'\nTest  accuracy={test_acc:.4f}  macro_f1={test_macro_f1:.4f}  weighted_f1={test_weighted_f1:.4f}')
    print(classification_report(y_true, y_pred, zero_division=0))

    return {'best_dev_epoch':  best_epoch, 'best_dev_macro_f1': round(best_dev_f1, 4), 'dev':  {'macro_f1': round(best_dev_f1, 4)},
        'test': {'accuracy': round(test_acc, 4), 'macro_f1': round(test_macro_f1, 4), 'weighted_f1': round(test_weighted_f1, 4)}}

results = {}
for task in ['failure_impact_type', 'action_urgency_type']:
    results[task] = run_task(task)

os.makedirs('results', exist_ok=True)
with open('results/deberta.json', 'w') as f:
    json.dump(results, f, indent=2)
print('\ncompleted & saved')
