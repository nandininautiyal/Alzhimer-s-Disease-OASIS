import os
import pandas as pd

DATA_DIR = "oasis_data"

label_map = {
    "Non Demented": 0,
    "Very mild Dementia": 1,
    "Mild Dementia": 2
    
}

data = []

for class_name, label in label_map.items():
    class_path = os.path.join(DATA_DIR, class_name)

    for file in os.listdir(class_path):
        if file.endswith(".jpg") or file.endswith(".png"):
            filepath = os.path.join(class_path, file)
            data.append([filepath, label])

df = pd.DataFrame(data, columns=["filepath", "label"])

# shuffle
df = df.sample(frac=1).reset_index(drop=True)

# split
train = df[:int(0.7*len(df))]
val   = df[int(0.7*len(df)):int(0.85*len(df))]
test  = df[int(0.85*len(df)):]

train.to_csv("df_train.csv", index=False)
val.to_csv("df_val.csv", index=False)
test.to_csv("df_test.csv", index=False)

print("CSV files created!")