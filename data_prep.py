import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_and_balance_fraud_data(data_path, output_path):
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # 1. Scale 'Time' and 'Amount' columns
    print("Scaling 'Time' and 'Amount' columns...")
    scaler = StandardScaler()
    df['Time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
    df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    
    # 2. Drop duplicate rows
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"Dropping {duplicates} duplicate rows...")
        df.drop_duplicates(inplace=True)
    
    # 3. Under-sampling to balance the dataset
    print("Balancing dataset (Under-sampling majority class)...")
    frauds = df[df['Class'] == 1]
    legitimates = df[df['Class'] == 0]
    
    # Randomly select a number of legitimate transactions equal to the number of frauds
    legitimates_sampled = legitimates.sample(n=len(frauds), random_state=42)
    
    # Combine the balanced classes and shuffle the rows
    balanced_df = pd.concat([frauds, legitimates_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Balanced dataset shape: {balanced_df.shape} (Frauds: {len(frauds)}, Legitimate: {len(legitimates_sampled)})")
    
    # 4. Save to a new CSV
    print(f"Saving balanced data to {output_path}...")
    balanced_df.to_csv(output_path, index=False)
    print("Preprocessing complete! You can load this directly into Orange or your Python Dashboard.")

if __name__ == "__main__":
    # Ensure these paths point to where your massive 150MB dataset is located!
    input_file = r"C:/Users/ShyamVenkatraman/Desktop/FDA/creditcard.csv"
    output_file = r"C:/Users/ShyamVenkatraman/Desktop/FDA/creditcard_balanced.csv"
    
    preprocess_and_balance_fraud_data(input_file, output_file)
