import pandas as pd
from sklearn.preprocessing import StandardScaler

# 1. Load the raw scraped data
df = pd.read_csv('ikman_bikes_20260221_113858.csv')
print(f"Loaded {len(df)} rows, columns: {df.columns.tolist()}")

# 2. Keep only the required columns
required_cols = ['title', 'price', 'make', 'model', 'yom', 'mileage', 'engine_cc', 'location']
df = df[required_cols]

# 3. CLEANING: Drop rows with missing crucial data
df = df.dropna(subset=['price', 'engine_cc', 'yom', 'mileage'])
print(f"After dropping nulls: {len(df)} rows")

# 4. Remove duplicates based on title + price + mileage
df = df.drop_duplicates(subset=['title', 'price', 'mileage'])
print(f"After removing duplicates: {len(df)} rows")

# 5. Data type conversions & basic cleaning
df['price'] = df['price'].astype(int)
df['yom'] = df['yom'].astype(int)
df['mileage'] = df['mileage'].astype(int)
df['engine_cc'] = df['engine_cc'].astype(int)
df['make'] = df['make'].str.strip().str.title()
df['model'] = df['model'].str.strip().str.title()
df['location'] = df['location'].str.strip().str.title()

# 6. Remove outliers (price <= 0, mileage < 0, unreasonable yom)
df = df[(df['price'] > 0) & (df['mileage'] >= 0) & (df['yom'] >= 1980) & (df['yom'] <= 2026)]
print(f"After removing outliers: {len(df)} rows")

# 7. Save the cleaned dataset
df.to_csv('ikman_bikes_cleaned.csv', index=False)
print(f"\n✅ Preprocessed data saved to 'ikman_bikes_cleaned.csv'")
print(f"   Final shape: {df.shape}")
print(f"   Columns: {df.columns.tolist()}")
print(f"\nSample data:")
print(df.head().to_string())
print(f"\nData types:\n{df.dtypes}")
print(f"\nNull counts:\n{df.isnull().sum()}")