import os
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('bugs_classified.csv')
print(f'rows: {len(df)}')
print(f'columns: {list(df.columns)}')

print('\nFailure Impact Type:')
print(df['failure_impact_type'].value_counts().to_string())
print('\nAction Urgency Type:')
print(df['action_urgency_type'].value_counts().to_string())

df['strat'] = df['failure_impact_type'] + '|' + df['action_urgency_type']

#drop unneeded stuff
counts = df['strat'].value_counts()
df = df[df['strat'].isin(counts[counts >= 2].index)].copy()
dropped = 2000 - len(df)
if dropped:
    print(f'\nDropped {dropped} rows with rare label combinations (needed for stratified split)')

#80, 10, 10 split
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['strat'])
dev_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

#remove helper column
for split in [train_df, dev_df, test_df]:
    split.drop(columns=['strat'], inplace=True)

print(f'\nsplit sizes:')
print(f'train: {len(train_df)}')
print(f'dev: {len(dev_df)}')
print(f'test: {len(test_df)}')
os.makedirs('data', exist_ok=True)
train_df.to_csv('data/train.csv', index=False)
dev_df.to_csv('data/dev.csv', index=False)
test_df.to_csv('data/test.csv', index=False)
print('\ncompleted & saved')
