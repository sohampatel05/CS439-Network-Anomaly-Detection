import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import shap

# Import the preprocessing function from your first script
# (Ensuring preprocess.py is in the same directory)
from preprocess import load_and_preprocess_cicids

RANDOM_STATE = 42

def apply_clustering(X_train, X_test, n_clusters=5):
    """
    Applies K-Means clustering to discover latent behavioral patterns in the traffic.
    Engineers a new feature 'behavioral_cluster' for the supervised model.
    """
    print(f"\n--- Unsupervised Stage ---")
    print(f"Fitting K-Means with {n_clusters} clusters...")
    
    # n_init='auto' suppresses a common sklearn warning
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init='auto')
    
    # Fit and predict on training data
    train_clusters = kmeans.fit_predict(X_train)
    
    # Predict ONLY on testing data to prevent data leakage
    test_clusters = kmeans.predict(X_test)
    
    # Create copies to avoid SettingWithCopyWarning
    X_train_hybrid = X_train.copy()
    X_test_hybrid = X_test.copy()
    
    # Append the cluster IDs as a new engineered feature
    X_train_hybrid['behavioral_cluster'] = train_clusters
    X_test_hybrid['behavioral_cluster'] = test_clusters
    
    print("Clustering complete. 'behavioral_cluster' feature added.")
    return X_train_hybrid, X_test_hybrid, train_clusters

def plot_pca_clusters(X_train, cluster_labels):
    """
    Generates a 2D PCA projection of the behavioral clusters for the final report.
    """
    print("\nGenerating PCA projection for the report...")
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    
    # Sample 5,000 points so the plot is readable and renders quickly
    X_sample = X_train.sample(n=5000, random_state=RANDOM_STATE)
    labels_sample = cluster_labels[X_sample.index]
    
    pca_result = pca.fit_transform(X_sample)
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=labels_sample, palette='tab10', s=15, alpha=0.7)
    plt.title("PCA 2D Projection of Behavioral Clusters")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title='Cluster ID')
    plt.tight_layout()
    plt.show()

def balance_training_data(X_train, y_train):
    """
    Applies SMOTE to balance the minority class in the training set.
    """
    print(f"\n--- Class Balancing ---")
    print("Applying SMOTE to training data...")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    print(f"Original training distribution:\n{y_train.value_counts()}")
    print(f"Balanced training distribution:\n{y_train_balanced.value_counts()}")
    
    return X_train_balanced, y_train_balanced

def train_baseline(X_train, y_train, X_test, y_test):
    """
    Trains a basic Logistic Regression model as an ablation/baseline comparison.
    """
    print("\n--- Baseline Model (Logistic Regression) ---")
    # Using max_iter=1000 to ensure convergence
    baseline = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    baseline.fit(X_train, y_train)
    
    y_pred_base = baseline.predict(X_test)
    y_pred_proba_base = baseline.predict_proba(X_test)[:, 1]
    
    print("Baseline Classification Report:")
    print(classification_report(y_test, y_pred_base, target_names=['Benign', 'DDoS']))
    
    base_auc = roc_auc_score(y_test, y_pred_proba_base)
    print(f"Baseline ROC-AUC Score: {base_auc:.4f}")
    return baseline

def train_evaluate_xgboost(X_train, y_train, X_test, y_test):
    """
    Trains XGBoost on the hybrid feature set and evaluates its performance.
    """
    print("\n--- Supervised Stage (Hybrid Model) ---")
    print("Training XGBoost Classifier...")
    
    # Using 'hist' tree method for much faster training on large datasets
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        tree_method='hist', 
        random_state=RANDOM_STATE,
        eval_metric='auc'
    )
    
    xgb_model.fit(X_train, y_train)
    
    print("\nEvaluating Hybrid Model...")
    y_pred = xgb_model.predict(X_test)
    y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
    
    print("\nHybrid Model Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Benign', 'DDoS']))
    
    auc_score = roc_auc_score(y_test, y_pred_proba)
    print(f"Hybrid ROC-AUC Score: {auc_score:.4f}")
    
    # Confusion Matrix Visualization
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benign', 'DDoS'])
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix: Hybrid IDS (K-Means + XGBoost)")
    plt.show()
    
    return xgb_model

def generate_shap_explanations(model, X_train, X_test):
    """
    Generates SHAP values to explain model predictions (Explainable AI).
    """
    print("\n--- Explainable AI (XAI) ---")
    print("Generating SHAP explanations...")
    
    # Initialize the Tree Explainer
    explainer = shap.TreeExplainer(model)
    
    # Sample the test set to generate visuals in a reasonable timeframe
    print("Sampling 2,000 instances from the test set for SHAP calculation...")
    X_test_sampled = X_test.sample(n=2000, random_state=RANDOM_STATE)
    
    shap_values = explainer.shap_values(X_test_sampled)
    
    # 1. Summary Plot (Global Explainability - dot plot)
    plt.figure()
    plt.title("SHAP Summary Plot: Feature Impact")
    shap.summary_plot(shap_values, X_test_sampled, plot_type="dot")
    
    # 2. Bar Plot (Mean Absolute Impact)
    plt.figure()
    plt.title("SHAP Feature Importance (Mean Absolute Value)")
    shap.summary_plot(shap_values, X_test_sampled, plot_type="bar")


# ==========================================
# Main Execution Block
# ==========================================
if __name__ == "__main__":
    # 1. Load and Preprocess Data
    csv_path = 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'
    X_train, X_test, y_train, y_test = load_and_preprocess_cicids(csv_path)
    
    # 2. Unsupervised Stage: Clustering
    X_train_hybrid, X_test_hybrid, train_clusters = apply_clustering(X_train, X_test, n_clusters=5)
    
    # 3. Generate PCA Visualization for the report
    plot_pca_clusters(X_train, pd.Series(train_clusters, index=X_train.index))
    
    # 4. Handle Imbalance on the hybrid dataset
    X_train_balanced, y_train_balanced = balance_training_data(X_train_hybrid, y_train)
    
    # 5. Ablation/Baseline Comparison (Using original scaled data, without clusters)
    # We balance the original X_train first to ensure a fair comparison
    X_train_base_bal, y_train_base_bal = balance_training_data(X_train, y_train)
    baseline_model = train_baseline(X_train_base_bal, y_train_base_bal, X_test, y_test)
    
    # 6. Supervised Stage: Train and Evaluate Hybrid Model
    xgb_model = train_evaluate_xgboost(X_train_balanced, y_train_balanced, X_test_hybrid, y_test)
    
    # 7. Explainability Stage: SHAP
    generate_shap_explanations(xgb_model, X_train_balanced, X_test_hybrid)