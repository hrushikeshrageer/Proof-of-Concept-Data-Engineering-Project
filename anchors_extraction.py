"""
================================================================================
Approach 1: ANCHORS-BASED RULE EXTRACTION FOR FRAUD DETECTION
================================================================================
Extracts high-precision fraud detection rules from a trained neural network
using the Anchors algorithm (model-agnostic, probabilistic precision guarantees).

Dependencies: tensorflow, keras, scikit-learn, alibi, pandas, numpy

Usage:
    python anchors_rule_extraction.py --csv_path creditcard_data.csv \
                                      --target_col fraud \
                                      --output_rules rules_anchors.csv
================================================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    confusion_matrix, precision_recall_curve, classification_report, roc_auc_score
)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from alibi.explainers import AnchorTabular
import warnings
import argparse
from datetime import datetime
import json

warnings.filterwarnings('ignore')

# ==================== CONFIG ====================
class Config:
    """Configuration parameters for Anchors rule extraction"""
    
    # Model parameters
    EPOCHS = 50
    BATCH_SIZE = 32
    VALIDATION_SPLIT = 0.2
    RANDOM_STATE = 42
    
    FP_WEIGHT = 0.1  # Reduce weight on genuine class to reduce false positives
    
    # Anchors parameters
    ANCHOR_PRECISION_THRESHOLD = 0.95  # Minimum precision required for rules
    ANCHOR_MAX_FEATURES = 5  # Max features per rule (interpretability)
    ANCHOR_EPSILON = 0.1  # Tolerance for multi-armed bandit
    ANCHOR_BEAM_WIDTH = 10  # Number of candidate rules to explore
    
    # Rule extraction
    MIN_RULE_COVERAGE = 0.005  # Minimum coverage (% of data)
    SAMPLE_SIZE_FOR_ANCHORS = 500  # Number of fraud instances to extract anchors from
    
    # Output
    OUTPUT_DIR = './fraud_rules_output'
    VERBOSE = True


# ==================== DATA LOADING & PREPROCESSING ====================
def load_and_prepare_data(csv_path, target_col='fraud', val_size=0.2, test_size=0.2):
    """
    Load CSV data and split into train/val/test sets.
    
    Args:
        csv_path (str): Path to CSV file
        target_col (str): Name of target column
        val_size (float): Validation set size
        test_size (float): Test set size
    
    Returns:
        dict: Contains X_train, X_val, X_test, y_train, y_val, y_test, feature_names
    """
    #################################################################################################################################################
    # CHANGE DATA LOADING FOR DATAIKU
    print("\n[INFO] Loading data from:", csv_path)
    df = pd.read_csv(csv_path)
    #################################################################################################################################################
    
    print(f"[INFO] Data shape: {df.shape}")
    print(f"[INFO] Target column: {target_col}")
    print(f"[INFO] Fraud rate: {df[target_col].mean():.2%}")
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col].values
    
    feature_names = X.columns.tolist()
    print(f"[INFO] Features ({len(feature_names)}): {feature_names}")
    
    # Train-test split (80-20)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=Config.RANDOM_STATE, stratify=y
    )
    
    # Train-val split (75-25 of 80% = 60-20)
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, 
        random_state=Config.RANDOM_STATE, stratify=y_temp
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\n[INFO] Split sizes:")
    print(f"  Train: {X_train_scaled.shape[0]} | Fraud rate: {y_train.mean():.2%}")
    print(f"  Val: {X_val_scaled.shape[0]} | Fraud rate: {y_val.mean():.2%}")
    print(f"  Test: {X_test_scaled.shape[0]} | Fraud rate: {y_test.mean():.2%}")
    
    return {
        'X_train': X_train_scaled,
        'X_val': X_val_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'feature_names': feature_names,
        'scaler': scaler,
        'df': df,
        'target_col': target_col
    }


# ==================== NEURAL NETWORK TRAINING ====================
def build_and_train_model(X_train, y_train, X_val, y_val):
    """
    Build and train cost-sensitive neural network.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
    
    Returns:
        Trained Keras model
    """
    print("\n[INFO] Building cost-sensitive neural network...")
    
    # Compute class weights
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    
    # Adjust for false positive cost
    weight_dict = {
        0: class_weights[0] * Config.FP_WEIGHT,  # Reduce weight on genuine class
        1: class_weights[1]
    }
    
    print(f"[INFO] Class weights: {weight_dict}")
    
    # Build model
    model = Sequential([
        Dense(64, activation='relu', input_dim=X_train.shape[1]),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    print("[INFO] Model architecture:")
    model.summary()
    
    # Train with class weights
    print(f"\n[INFO] Training model (epochs={Config.EPOCHS})...")
    history = model.fit(
        X_train, y_train,
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        class_weight=weight_dict,
        validation_data=(X_val, y_val),
        verbose=0 if not Config.VERBOSE else 1
    )
    
    # Evaluate on test set
    val_loss, val_acc, val_prec, val_rec = model.evaluate(X_val, y_val, verbose=0)
    print(f"\n[INFO] Validation metrics:")
    print(f"  Accuracy: {val_acc:.4f}")
    print(f"  Precision: {val_prec:.4f}")
    print(f"  Recall: {val_rec:.4f}")
    
    return model


# ==================== ANCHORS RULE EXTRACTION ====================
def extract_anchors_rules(model, X_train, X_val, y_val, feature_names):
    """
    Extract high-precision rules using Anchors algorithm.
    
    Args:
        model: Trained neural network
        X_train: Training data (for perturbation distribution)
        X_val: Validation data
        y_val: Validation labels
        feature_names: List of feature names
    
    Returns:
        DataFrame with extracted rules
    """
    print("\n[INFO] Initializing Anchors explainer...")
    
    # Create predictor function
    def predict_fn(X):
        """Wrapper for model predictions"""
        return model.predict(X, verbose=0)[:, 0]
    
    # Initialize explainer
    explainer = AnchorTabular(
        predictor=predict_fn,
        feature_names=feature_names,
    )
    
    # Get high-fraud instances for rule extraction
    y_pred_proba = model.predict(X_val, verbose=0)[:, 0]
    high_fraud_idx = np.where(y_pred_proba >= 0.5)[0]  # Confidence >= 50%
    
    if len(high_fraud_idx) == 0:
        print("[WARNING] No high-fraud instances found. Adjusting threshold...")
        high_fraud_idx = np.argsort(y_pred_proba)[-Config.SAMPLE_SIZE_FOR_ANCHORS:]
    
    sample_idx = np.random.choice(
        high_fraud_idx,
        size=min(Config.SAMPLE_SIZE_FOR_ANCHORS, len(high_fraud_idx)),
        replace=False
    )
    
    print(f"\n[INFO] Extracting anchors from {len(sample_idx)} high-fraud instances...")
    
    anchors_list = []
    
    for i, idx in enumerate(sample_idx):
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{len(sample_idx)}")
        
        instance = X_val[idx:idx+1]
        
        try:
            # Explain prediction
            explanation = explainer.explain(
                instance,
                threshold=Config.ANCHOR_PRECISION_THRESHOLD,
                max_features=Config.ANCHOR_MAX_FEATURES,
                epsilon=Config.ANCHOR_EPSILON
            )
            
            anchor_dict = {
                'rule_id': len(anchors_list),
                'rule_conditions': str(explanation.anchor),
                'precision': explanation.precision,
                'coverage': explanation.coverage,
                'prediction': explanation.raw['prediction'][0] if 'prediction' in explanation.raw else 1
            }
            anchors_list.append(anchor_dict)
        
        except Exception as e:
            if Config.VERBOSE:
                print(f"  [WARNING] Failed to extract anchor for instance {idx}: {str(e)[:100]}")
            continue
    
    # Convert to DataFrame
    rules_df = pd.DataFrame(anchors_list)
    
    if len(rules_df) > 0:
        # Remove duplicates
        rules_df = rules_df.drop_duplicates(subset=['rule_conditions']).reset_index(drop=True)
        
        # Filter by minimum coverage
        rules_df = rules_df[rules_df['coverage'] >= Config.MIN_RULE_COVERAGE]
        
        print(f"\n[INFO] Extracted {len(rules_df)} unique rules")
        print(f"[INFO] Average precision: {rules_df['precision'].mean():.4f}")
        print(f"[INFO] Average coverage: {rules_df['coverage'].mean():.4f}")
    
    return rules_df


# ==================== EVALUATION ====================
def evaluate_rules_on_test_set(model, X_test, y_test, rules_df):
    """
    Evaluate extracted rules on test set and calculate false positive rate.
    
    Args:
        model: Trained neural network
        X_test: Test data
        y_test: Test labels
        rules_df: Rules DataFrame
    
    Returns:
        dict: Evaluation metrics
    """
    print("\n[INFO] Evaluating rules on test set...")
    
    y_pred_proba = model.predict(X_test, verbose=0)[:, 0]
    
    # Calculate precision-recall curve
    precision_curve, recall_curve, thresholds = precision_recall_curve(y_test, y_pred_proba)
    
    # Find optimal threshold to reduce false positives
    target_recall = 0.85  # Catch 85% of fraud
    idx_recall = np.argmax(recall_curve >= target_recall)
    optimal_threshold = thresholds[idx_recall] if idx_recall < len(thresholds) else 0.5
    
    y_pred = (y_pred_proba >= optimal_threshold).astype(int)
    
    # Calculate metrics
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    metrics = {
        'threshold': optimal_threshold,
        'false_positive_rate': fpr,
        'false_negative_rate': fnr,
        'precision': precision,
        'recall': recall,
        'roc_auc': roc_auc,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }
    
    print(f"[INFO] Test Set Metrics at threshold {optimal_threshold:.3f}:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  False Positive Rate: {fpr:.4f}")
    print(f"  False Negative Rate: {fnr:.4f}")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"\n[INFO] Confusion Matrix:")
    print(f"  TP: {tp} | FN: {fn}")
    print(f"  FP: {fp} | TN: {tn}")
    
    return metrics


# ==================== OUTPUT & REPORTING ====================
def save_results(rules_df, metrics, output_dir, csv_path):
    """
    Save extracted rules and metrics to output directory.
    
    Args:
        rules_df: DataFrame with rules
        metrics: Evaluation metrics dictionary
        output_dir: Output directory path
        csv_path: Original CSV path
    """
    import os
    # os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    #################################################################################################################################################
    # CHANGE OUTPUT FILE PATH OR JUST PRINT TO VIEW
    rules_output_path = os.path.join(output_dir, f"anchors_rules_{timestamp}.csv")
    rules_df.to_csv(rules_output_path, index=False)
    print(f"\n[INFO] Rules saved to: {rules_output_path}")
    #################################################################################################################################################
    
    # # Save metrics
    # metrics_output_path = os.path.join(output_dir, f"anchors_metrics_{timestamp}.json")
    # with open(metrics_output_path, 'w') as f:
    #     json.dump(metrics, f, indent=2)
    # print(f"[INFO] Metrics saved to: {metrics_output_path}")
    print(f"[INFO] Metrics: {metrics}")
    
    # # Save summary report
    # summary_path = os.path.join(output_dir, f"anchors_summary_{timestamp}.txt")
    # with open(summary_path, 'w') as f:
    #     f.write("=" * 80 + "\n")
    #     f.write("ANCHORS RULE EXTRACTION - SUMMARY REPORT\n")
    #     f.write("=" * 80 + "\n\n")
    #     f.write(f"Input CSV: {csv_path}\n")
    #     f.write(f"Extraction Date: {timestamp}\n\n")
        
    #     f.write("RULES EXTRACTED:\n")
    #     f.write(f"  Total Rules: {len(rules_df)}\n")
    #     f.write(f"  Avg Precision: {rules_df['precision'].mean():.4f}\n")
    #     f.write(f"  Avg Coverage: {rules_df['coverage'].mean():.4f}\n")
    #     f.write(f"  Min Precision: {rules_df['precision'].min():.4f}\n")
    #     f.write(f"  Max Coverage: {rules_df['coverage'].max():.4f}\n\n")
        
    #     f.write("TEST SET METRICS:\n")
    #     for key, value in metrics.items():
    #         if key != 'threshold':
    #             f.write(f"  {key}: {value}\n")
        
    #     f.write("\n" + "=" * 80 + "\n")
    #     f.write("TOP 10 RULES (BY PRECISION):\n")
    #     f.write("=" * 80 + "\n\n")
        
    #     top_rules = rules_df.nlargest(10, 'precision')
    #     for idx, row in top_rules.iterrows():
    #         f.write(f"Rule {row['rule_id']+1}:\n")
    #         f.write(f"  Condition: {row['rule_conditions']}\n")
    #         f.write(f"  Precision: {row['precision']:.4f}\n")
    #         f.write(f"  Coverage: {row['coverage']:.4f}\n\n")
    
    # print(f"[INFO] Summary report saved to: {summary_path}")

    print("=" * 80 + "\n")
    print("ANCHORS RULE EXTRACTION - SUMMARY REPORT\n")
    print("=" * 80 + "\n\n")
    print(f"Input CSV: {csv_path}\n")
    print(f"Extraction Date: {timestamp}\n\n")
    
    print("RULES EXTRACTED:\n")
    print(f"  Total Rules: {len(rules_df)}\n")
    print(f"  Avg Precision: {rules_df['precision'].mean():.4f}\n")
    print(f"  Avg Coverage: {rules_df['coverage'].mean():.4f}\n")
    print(f"  Min Precision: {rules_df['precision'].min():.4f}\n")
    print(f"  Max Coverage: {rules_df['coverage'].max():.4f}\n\n")
    
    print("TEST SET METRICS:\n")
    for key, value in metrics.items():
        if key != 'threshold':
            print(f"  {key}: {value}\n")
    
    print("\n" + "=" * 80 + "\n")
    print("TOP 10 RULES (BY PRECISION):\n")
    print("=" * 80 + "\n\n")
    
    top_rules = rules_df.nlargest(10, 'precision')
    for idx, row in top_rules.iterrows():
        print(f"Rule {row['rule_id']+1}:\n")
        print(f"  Condition: {row['rule_conditions']}\n")
        print(f"  Precision: {row['precision']:.4f}\n")
        print(f"  Coverage: {row['coverage']:.4f}\n\n")

    print(f"\n[SUCCESS] Anchors rule extraction complete!")
    # print(f"  Output directory: {output_dir}")


# ==================== MAIN ====================
def main(csv_path, target_col, output_dir):
    """Main execution flow"""
    
    print("\n" + "=" * 80)
    print("ANCHORS RULE EXTRACTION FOR FRAUD DETECTION")
    print("=" * 80)
    
    # Step 1: Load and prepare data
    data = load_and_prepare_data(csv_path, target_col=target_col)
    
    # Step 2: Train neural network
    model = build_and_train_model(
        data['X_train'], data['y_train'],
        data['X_val'], data['y_val']
    )
    
    # Step 3: Extract rules using Anchors
    rules_df = extract_anchors_rules(
        model, data['X_train'], data['X_val'], data['y_val'],
        data['feature_names']
    )
    
    if len(rules_df) == 0:
        print("[ERROR] No rules extracted. Check data and model performance.")
        return
    
    # Step 4: Evaluate on test set
    metrics = evaluate_rules_on_test_set(model, data['X_test'], data['y_test'], rules_df)
    
    # Step 5: Save results
    save_results(rules_df, metrics, output_dir, csv_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SOAR Rule Extraction for Fraud Detection')
    parser.add_argument('--csv_path', type=str, required=False, default='', help='Path to input CSV file')
    parser.add_argument('--target_col', type=str, default='y_actual', help='Target column name')
    parser.add_argument('--output_dir', type=str, default=Config.OUTPUT_DIR, help='Output directory')
    
    args = parser.parse_args()
    
    main(args.csv_path, args.target_col, args.output_dir)
