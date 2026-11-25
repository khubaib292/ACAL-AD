import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedGroupKFold

SOURCE = "outputs/dataset_inspection/image_inventory.csv"
OUTPUT = "outputs/dataset_split/original_only_manifest_80_10_10_reconstructed.csv"

df = pd.read_csv(SOURCE)
df = df[df["valid"] == True].copy().reset_index(drop=True)

original_df = df[["path", "class", "sha256"]].copy()
original_df["original_split"] = "val"
original_df["kaggle_path"] = original_df["path"]

train_df, temp_df = train_test_split(
    original_df, test_size=0.30, random_state=42,
    stratify=original_df["class"]
)

val_df, test_df = train_test_split(
    temp_df, test_size=0.50, random_state=42,
    stratify=temp_df["class"]
)

train_df = train_df.copy()
val_df = val_df.copy()
test_df = test_df.copy()

train_df["split"] = "train"
val_df["split"] = "val"
test_df["split"] = "test"

original_manifest = pd.concat(
    [train_df, val_df, test_df], ignore_index=True
).sample(frac=1, random_state=42).reset_index(drop=True)

split_df = original_manifest.copy().reset_index(drop=True)

X = split_df[["kaggle_path"]]
y = split_df["class"]
groups = split_df["sha256"]

splitter_80 = StratifiedGroupKFold(
    n_splits=5, shuffle=True, random_state=42
)
train_idx, temp_idx = next(splitter_80.split(X, y, groups))

split_df["split"] = "train"
temp_df = split_df.iloc[temp_idx].copy()

splitter_10 = StratifiedGroupKFold(
    n_splits=2, shuffle=True, random_state=42
)
val_local_idx, test_local_idx = next(
    splitter_10.split(
        temp_df[["kaggle_path"]],
        temp_df["class"],
        temp_df["sha256"]
    )
)

split_df.loc[temp_df.index[val_local_idx], "split"] = "validation"
split_df.loc[temp_df.index[test_local_idx], "split"] = "test"

split_df.to_csv(OUTPUT, index=False)

print("RECONSTRUCTED 80:10:10 MANIFEST")
print("Split counts:")
print(split_df["split"].value_counts())
print("\nClass-wise distribution:")
print(pd.crosstab(split_df["class"], split_df["split"]))
print("\nDuplicate leakage groups:",
      (split_df.groupby("sha256")["split"].nunique() > 1).sum())
print("\nSaved:", OUTPUT)
print("\nNOTE: Reconstructed split; original Kaggle row order is unverified.")
