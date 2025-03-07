import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# 1. Load the Excel File
# ------------------------------
# Replace the filename with the correct path if needed.
file_path = "nhdplus_attributes.xlsx"
df = pd.read_excel(file_path)

# Basic overview of the data
print("First few rows:")
print(df.head())
print("\nData Info:")
print(df.info())
print("\nDescriptive Statistics:")
print(df.describe())

# ------------------------------
# 2. Data Quality Check
# ------------------------------
# Count missing values per column
missing_counts = df.isnull().sum()
print("\nMissing values per column:")
print(missing_counts)
