import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_cicids(filepath):
    print("1. Loading dataset...")
    df = pd.read_csv(filepath)
    
    # ---------------------------------------------------------
    # STEP 1: Basic Cleaning
    # ---------------------------------------------------------
    print("2. Cleaning column names and handling anomalies...")
    # CIC-IDS-2017 column names often have leading/trailing spaces
    df.columns = df.columns.str.strip()
    
    # Replace Infinite values with NaNs, then drop all rows with NaNs
    # (Since there are only 68 problematic rows out of 225k, dropping them is safest)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    # ---------------------------------------------------------
    # STEP 2: Feature and Label Separation
    # ---------------------------------------------------------
    print("3. Encoding target labels...")
    # The target column is named 'Label'
    X = df.drop(columns=['Label'])
    y_raw = df['Label']
    
    # Encode 'BENIGN' as 0 and 'DDoS' as 1
    y = y_raw.apply(lambda x: 0 if x == 'BENIGN' else 1)
    
    # Drop any non-numeric columns if they accidentally exist in the CSV 
    # (e.g., Flow ID, Timestamp, Source IP - though usually pre-removed in this specific CSV)
    X = X.select_dtypes(include=[np.number])
    
    # ---------------------------------------------------------
    # STEP 3: Train-Test Split (Crucial for preventing data leakage)
    # ---------------------------------------------------------
    print("4. Splitting data (Stratified)...")
    # We split BEFORE scaling to strictly prevent data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # ---------------------------------------------------------
    # STEP 4: Feature Scaling
    # ---------------------------------------------------------
    print("5. Scaling features...")
    scaler = StandardScaler()
    
    # Fit the scaler ONLY on the training data, then transform both
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    
    print("\n--- Preprocessing Complete ---")
    print(f"Training set shape: {X_train_scaled.shape}")
    print(f"Testing set shape:  {X_test_scaled.shape}")
    print(f"Class distribution in train:\n{y_train.value_counts()}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test

# ==========================================
# Run the pipeline
# ==========================================
if __name__ == "__main__":
    file_path = 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'
    X_train, X_test, y_train, y_test = load_and_preprocess_cicids(file_path)
    
    # You can now pass X_train and X_test into your K-Means and XGBoost models!