import pandas as pd

# Load Dataset
df = pd.read_csv("music library songs.csv")

# Create duplicate dataset
new_df = df.copy()

print("=" * 90)
print("PROGRAM 1 : LOAD AND VIEW THE DATASET")
print("=" * 90)

print("\nDataset Shape :", new_df.shape)

# Select only important columns
new_df = new_df[[
    "Song Title",
    "Album Title",
    "Artist Name 1",
    "Artist Name 2",
    "Artist Name 3"
]]

print("\nSelected Columns :")
print(list(new_df.columns))

print("\n" + "=" * 90)
print("FIRST 10 RECORDS")
print("=" * 90)
print(new_df.head(10))

print("\n" + "=" * 90)
print("DATASET INFORMATION")
print("=" * 90)
new_df.info()

print("\n" + "=" * 90)
print("SUMMARY STATISTICS")
print("=" * 90)
print(new_df.describe(include="all").T)

print("\n" + "=" * 90)
print("NULL VALUES")
print("=" * 90)
print(new_df.isnull().sum())

print("\n" + "=" * 90)
print("PROGRAM EXECUTED SUCCESSFULLY")
print("=" * 90)