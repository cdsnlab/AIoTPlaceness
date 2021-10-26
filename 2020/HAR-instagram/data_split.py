import numpy as np
import pandas as pd

df_label = pd.read_json('data/label_df.json')

num_classes = 6
cross_val = 5

fdf_label = df_label[df_label['category'] < num_classes]

train_dataset = {k:[] for k in range(cross_val)}
test_dataset = {k:[] for k in range(cross_val)}
for i in range(num_classes):
    tdf = fdf_label[fdf_label['category'] == i].sample(frac=1, random_state=0)
    datasize = len(tdf)
    batch = int(len(tdf)//cross_val)
    for k in range(cross_val):
        if k == 0:
            train_dataset[k].append(tdf.iloc[batch:])
            test_dataset[k].append(tdf.iloc[:batch])
        elif k == cross_val-1:
            train_dataset[k].append(tdf.iloc[:batch*(cross_val-1)])
            test_dataset[k].append(tdf.iloc[batch*(cross_val-1):])
        else:
            train_dataset[k].append(pd.concat((tdf.iloc[:batch*k], tdf.iloc[batch*(k+1):])))
            test_dataset[k].append(tdf.iloc[batch*k:batch*(k+1)])


for k in range(cross_val): 
    train_dataset[k] = pd.concat(train_dataset[k]).sample(frac=1, random_state=0).to_csv(f'data/train_{k}_instagram_label.csv', index=None)
    test_dataset[k] = pd.concat(test_dataset[k]).sample(frac=1, random_state=0).to_csv(f'data/test_{k}_instagram_label.csv', index=None)
