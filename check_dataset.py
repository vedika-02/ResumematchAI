
import pandas as pd
import glob
import os

files = glob.glob("*.csv")

if not files:
    print("No CSV file found. Copy your dataset into the ResumeMatchAI folder.")
else:
    for file in files:
        print("\nDataset:", file)
        df = pd.read_csv(file)
        print("Shape:", df.shape)
        print("Columns:", df.columns.tolist())
        print("\nFirst 3 rows:")
        print(df.head(3).to_string())
