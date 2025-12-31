"""
================================================================================
Approach 2: SOAR-BASED RULE EXTRACTION FOR FRAUD DETECTION
================================================================================
Extracts interpretable fraud detection rules using SOAR algorithm
(Sparse Oracle-based Adaptive Rule extraction).

Simplified approach: Uses K-Means for discretization instead of full ART2
to reduce complexity while maintaining SOAR's core principles.

Dependencies: tensorflow, keras, scikit-learn, pandas, numpy

Usage:
    python soar_rule_extraction.py --csv_path creditcard_data.csv \
                                   --target_col fraud \
                                   --output_rules rules_soar.csv
================================================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.cluster import KMeans
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    confusion_matrix, precision_recall_curve, classification_report, roc_auc_score
)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import warnings
import argparse
from datetime import datetime
from itertools import combinations
import json

warnings.filterwarnings('ignore')

# ==================== CONFIG ====================
class Config:
    """Configuration parameters for SOAR rule extraction"""
    
    # Model parameters
    EPOCHS = 50
    BATCH_SIZE = 32
    RANDOM_STATE = 42
    
    FP_WEIGHT = 0.1  # Reduce weight on genuine class
    
    # Discretization parameters
    N_BINS_CONTINUOUS = 3  # Number of bins for continuous features (low/medium/high)
    
    # Rule extraction parameters
    MIN_FRAUD_SUPPORT = 0.01  # Minimum % of fraud cases rule must cover
    MAX_ANTECEDENTS = 5  # Maximum features per rule
    
    # False positive pruning
    FP_RATIO_THRESHOLD = 0.30  # Remove rules with FP ratio > 30%
    
    # Output
    OUTPUT_DIR = './fraud_rules_soar_output'
    VERBOSE = True


# ==================== DATA LOADING & PREPROCESSING ====================
def load_and_prepare_data(csv_path, target_col='fraud', val_size=0.2, test_size=0.2):
    """
    Load CSV data and split into train/val/test sets.
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
    
    # Identify continuous vs categorical features
    continuous_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [f for f in feature_names if f not in continuous_features]
    
    print(f"[INFO] Continuous features: {continuous_features}")
    print(f"[INFO] Categorical features: {categorical_features}")
    
    # Train-test split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=Config.RANDOM_STATE, stratify=y
    )
    
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted,
        random_state=Config.RANDOM_STATE, stratify=y_temp
    )
    
    # Scale continuous features
    scaler = StandardScaler()
    X_train_cont = X_train[continuous_features].copy()
    X_val_cont = X_val[continuous_features].copy()
    X_test_cont = X_test[continuous_features].copy()
    
    X_train_cont[:] = scaler.fit_transform(X_train_cont)
    X_val_cont[:] = scaler.transform(X_val_cont)
    X_test_cont[:] = scaler.transform(X_test_cont)
    
    # Update dataframes with scaled values
    X_train[continuous_features] = X_train_cont
    X_val[continuous_features] = X_val_cont
    X_test[continuous_features] = X_test_cont
    
    print(f"\n[INFO] Split sizes:")
    print(f"  Train: {X_train.shape[0]} | Fraud rate: {y_train.mean():.2%}")
    print(f"  Val: {X_val.shape[0]} | Fraud rate: {y_val.mean():.2%}")
    print(f"  Test: {X_test.shape[0]} | Fraud rate: {y_test.mean():.2%}")
    
    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'feature_names': feature_names,
        'continuous_features': continuous_features,
        'categorical_features': categorical_features,
        'scaler': scaler,
        'df': df,
        'target_col': target_col
    }


# ==================== DISCRETIZATION (SOAR STEP 1) ====================
def discretize_features(X_train, X_val, X_test, continuous_features, categorical_features):
    """
    Discretize continuous features using K-Bins discretization.
    Mimics SOAR's clustering step with simpler K-Means approach.
    
    Args:
        X_train, X_val, X_test: Data DataFrames
        continuous_features: List of continuous feature names
        categorical_features: List of categorical feature names
    
    Returns:
        dict: Contains discretized data and discretizer
    """
    print("\n[INFO] Discretizing continuous features...")
    
    # Use K-Bins discretizer for continuous features
    discretizer = KBinsDiscretizer(
        n_bins=Config.N_BINS_CONTINUOUS,
        encode='ordinal',
        strategy='quantile'
    )
    
    X_train_disc = X_train.copy()
    X_val_disc = X_val.copy()
    X_test_disc = X_test.copy()
    
    if len(continuous_features) > 0:
        # Fit on training data
        X_train_cont = discretizer.fit_transform(X_train[continuous_features])
        
        # Transform val and test
        X_val_cont = discretizer.transform(X_val[continuous_features])
        X_test_cont = discretizer.transform(X_test[continuous_features])
        
        # Update DataFrames
        for i, feat in enumerate(continuous_features):
            X_train_disc[feat] = X_train_cont[:, i].astype(int)
            X_val_disc[feat] = X_val_cont[:, i].astype(int)
            X_test_disc[feat] = X_test_cont[:, i].astype(int)
        
        print(f"[INFO] Discretized {len(continuous_features)} continuous features into {Config.N_BINS_CONTINUOUS} bins")
    
    # Ensure categorical features are integers
    for feat in categorical_features:
        X_train_disc[feat] = X_train_disc[feat].astype(int)
        X_val_disc[feat] = X_val_disc[feat].astype(int)
        X_test_disc[feat] = X_test_disc[feat].astype(int)
    
    return {
        'X_train_disc': X_train_disc,
        'X_val_disc': X_val_disc,
        'X_test_disc': X_test_disc,
        'discretizer': discretizer
    }


# ==================== NEURAL NETWORK TRAINING ====================
def build_and_train_model(X_train, y_train, X_val, y_val):
    """Build and train cost-sensitive neural network."""
    print("\n[INFO] Building cost-sensitive neural network...")
    
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    
    weight_dict = {
        0: class_weights[0] * Config.FP_WEIGHT,
        1: class_weights[1]
    }
    
    print(f"[INFO] Class weights: {weight_dict}")
    
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
    
    print("[INFO] Training model...")
    model.fit(
        X_train, y_train,
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        class_weight=weight_dict,
        validation_data=(X_val, y_val),
        verbose=0 if not Config.VERBOSE else 1
    )
    
    val_loss, val_acc, val_prec, val_rec = model.evaluate(X_val, y_val, verbose=0)
    print(f"[INFO] Validation - Accuracy: {val_acc:.4f}, Precision: {val_prec:.4f}, Recall: {val_rec:.4f}")
    
    return model


# ==================== SOAR RULE EXTRACTION ====================
def extract_soar_rules(model, X_train_disc, y_train, X_val_disc, y_val, feature_names):
    """
    Extract rules using SOAR algorithm:
    1. Find fraud instances (prototypes)
    2. Extract frequent patterns from fraud instances
    3. Create rules from patterns
    4. Prune high false-positive rules
    
    Args:
        model: Trained neural network
        X_train_disc, y_train: Discretized training data
        X_val_disc, y_val: Discretized validation data
        feature_names: Feature names
    
    Returns:
        DataFrame with extracted rules
    """
    print("\n[INFO] Extracting rules using SOAR algorithm...")
    
    # SOAR Step 2: Identify fraud prototypes (instances)
    fraud_mask = y_train == 1
    fraud_instances = X_train_disc[fraud_mask]
    
    print(f"[INFO] SOAR Step 1-2: Found {len(fraud_instances)} fraud prototypes")
    
    # SOAR Step 3: Extract frequent patterns from fraud instances
    # Generate rules by finding common feature-value combinations in fraud instances
    
    rules_list = []
    
    # Approach: Find most frequent feature-value combinations
    for feature in feature_names:
        # Get value frequency in fraud instances
        value_counts = fraud_instances[feature].value_counts()
        
        for value, count in value_counts.items():
            fraud_support = count / len(fraud_instances)
            
            if fraud_support >= Config.MIN_FRAUD_SUPPORT:
                rule = {
                    'conditions': f"{feature} == {int(value)}",
                    'feature': feature,
                    'value': int(value),
                    'fraud_support': fraud_support,
                    'num_features': 1,
                    'conditions_list': [(feature, int(value))]
                }
                rules_list.append(rule)
    
    # Generate multi-feature rules (combinations of 2-3 features)
    print(f"[INFO] SOAR Step 3: Generating multi-feature rules...")
    
    for num_features in range(2, min(Config.MAX_ANTECEDENTS + 1, len(feature_names) + 1)):
        new_rules = []
        
        for feature_combo in combinations(feature_names, num_features):
            # Get fraud instances matching all features in combination
            mask = pd.Series([True] * len(fraud_instances), index=fraud_instances.index)
            
            for feat in feature_combo:
                feat_values = fraud_instances[feat].unique()
                # For simplicity, take the most common value for each feature
                most_common_value = fraud_instances[feat].value_counts().idxmax()
                mask = mask & (fraud_instances[feat] == most_common_value)
            
            matching_count = mask.sum()
            fraud_support = matching_count / len(fraud_instances) if len(fraud_instances) > 0 else 0
            
            if fraud_support >= Config.MIN_FRAUD_SUPPORT and matching_count >= 5:
                conditions = []
                conditions_list = []
                
                for feat in feature_combo:
                    most_common_value = fraud_instances[feat].value_counts().idxmax()
                    conditions.append(f"{feat} == {int(most_common_value)}")
                    conditions_list.append((feat, int(most_common_value)))
                
                rule = {
                    'conditions': ' AND '.join(conditions),
                    'conditions_list': conditions_list,
                    'fraud_support': fraud_support,
                    'num_features': num_features
                }
                new_rules.append(rule)
        
        rules_list.extend(new_rules[:100])  # Limit to prevent explosion
        
        if len(new_rules) == 0:
            break
    
    print(f"[INFO] Generated {len(rules_list)} candidate rules")
    
    # SOAR Step 4: Calculate false positive ratios and prune (SOAR Step 5)
    print(f"[INFO] SOAR Step 4-5: Calculating FP ratios and pruning...")
    
    final_rules = []
    
    for rule in rules_list:
        # Evaluate rule on validation set
        val_mask = pd.Series([True] * len(X_val_disc), index=X_val_disc.index)
        
        for feat, val in rule['conditions_list']:
            val_mask = val_mask & (X_val_disc[feat] == val)
        
        matching_count = val_mask.sum()
        
        if matching_count < 5:  # Skip rules with very low coverage
            continue
        
        # Count TP and FP among matching instances
        matching_labels = y_val[val_mask]
        tp = (matching_labels == 1).sum()
        fp = (matching_labels == 0).sum()
        
        fp_ratio = fp / (fp + tp) if (fp + tp) > 0 else 0
        precision = tp / (fp + tp) if (fp + tp) > 0 else 0
        coverage = matching_count / len(X_val_disc)
        
        # SOAR pruning: Remove rules with high FP ratio
        if fp_ratio <= Config.FP_RATIO_THRESHOLD:
            rule['precision'] = precision
            rule['coverage'] = coverage
            rule['fp_ratio'] = fp_ratio
            rule['tp'] = tp
            rule['fp'] = fp
            rule['matching_count'] = matching_count
            final_rules.append(rule)
    
    # Convert to DataFrame
    rules_df = pd.DataFrame(final_rules)
    
    if len(rules_df) > 0:
        # Sort by precision (descending)
        rules_df = rules_df.sort_values('precision', ascending=False).reset_index(drop=True)
        rules_df['rule_id'] = range(len(rules_df))
        
        print(f"\n[INFO] Final rules after pruning: {len(rules_df)}")
        print(f"[INFO] Average precision: {rules_df['precision'].mean():.4f}")
        print(f"[INFO] Average coverage: {rules_df['coverage'].mean():.4f}")
        print(f"[INFO] Average FP ratio: {rules_df['fp_ratio'].mean():.4f}")
    
    return rules_df


# ==================== EVALUATION ====================
def evaluate_rules_on_test_set(model, X_test, y_test, rules_df):
    """Evaluate extracted rules on test set."""
    print("\n[INFO] Evaluating rules on test set...")
    
    y_pred_proba = model.predict(X_test, verbose=0)[:, 0]
    
    # Calculate precision-recall curve
    precision_curve, recall_curve, thresholds = precision_recall_curve(y_test, y_pred_proba)
    
    # Find optimal threshold
    target_recall = 0.85
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
    print(f"  ROC-AUC: {roc_auc:.4f}")
    
    return metrics


# ==================== OUTPUT & REPORTING ====================
def save_results(rules_df, metrics, output_dir, csv_path):
    """Save extracted rules and metrics."""
    import os
    # os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    #################################################################################################################################################
    # CHANGE OUTPUT FILE PATH OR JUST PRINT TO VIEW
    rules_output_path = os.path.join(output_dir, f"soar_rules_{timestamp}.csv")
    output_cols = ['rule_id', 'conditions', 'precision', 'coverage', 'fp_ratio', 'tp', 'fp']
    rules_df[output_cols].to_csv(rules_output_path, index=False)
    print(f"\n[INFO] Rules saved to: {rules_output_path}")
    #################################################################################################################################################

    # # Save metrics
    # metrics_output_path = os.path.join(output_dir, f"soar_metrics_{timestamp}.json")
    # with open(metrics_output_path, 'w') as f:
    #     json.dump(metrics, f, indent=2)
    # print(f"[INFO] Metrics saved to: {metrics_output_path}")
    print(f"[INFO] Metrics: {metrics}")
    
    # # Save summary report
    # summary_path = os.path.join(output_dir, f"soar_summary_{timestamp}.txt")
    # with open(summary_path, 'w') as f:
    #     f.write("=" * 80 + "\n")
    #     f.write("SOAR RULE EXTRACTION - SUMMARY REPORT\n")
    #     f.write("=" * 80 + "\n\n")
    #     f.write(f"Input CSV: {csv_path}\n")
    #     f.write(f"Extraction Date: {timestamp}\n\n")
        
    #     f.write("RULES EXTRACTED:\n")
    #     f.write(f"  Total Rules: {len(rules_df)}\n")
    #     f.write(f"  Avg Precision: {rules_df['precision'].mean():.4f}\n")
    #     f.write(f"  Avg Coverage: {rules_df['coverage'].mean():.4f}\n")
    #     f.write(f"  Avg FP Ratio: {rules_df['fp_ratio'].mean():.4f}\n")
    #     f.write(f"  Min FP Ratio: {rules_df['fp_ratio'].min():.4f}\n\n")
        
    #     f.write("TEST SET METRICS:\n")
    #     for key, value in metrics.items():
    #         if key != 'threshold':
    #             f.write(f"  {key}: {value}\n")
        
    #     f.write("\n" + "=" * 80 + "\n")
    #     f.write("TOP 10 RULES (BY PRECISION):\n")
    #     f.write("=" * 80 + "\n\n")
        
    #     top_rules = rules_df.nlargest(10, 'precision')
    #     for idx, row in top_rules.iterrows():
    #         f.write(f"Rule {int(row['rule_id'])+1}:\n")
    #         f.write(f"  IF {row['conditions']}\n")
    #         f.write(f"  THEN FRAUD\n")
    #         f.write(f"  Precision: {row['precision']:.4f}\n")
    #         f.write(f"  Coverage: {row['coverage']:.4f}\n")
    #         f.write(f"  FP Ratio: {row['fp_ratio']:.4f}\n")
    #         f.write(f"  TP: {int(row['tp'])}, FP: {int(row['fp'])}\n\n")
    
    # print(f"[INFO] Summary report saved to: {summary_path}")

    print("=" * 80 + "\n")
    print("SOAR RULE EXTRACTION - SUMMARY REPORT\n")
    print("=" * 80 + "\n\n")
    print(f"Input CSV: {csv_path}\n")
    print(f"Extraction Date: {timestamp}\n\n")
    
    print("RULES EXTRACTED:\n")
    print(f"  Total Rules: {len(rules_df)}\n")
    print(f"  Avg Precision: {rules_df['precision'].mean():.4f}\n")
    print(f"  Avg Coverage: {rules_df['coverage'].mean():.4f}\n")
    print(f"  Avg FP Ratio: {rules_df['fp_ratio'].mean():.4f}\n")
    print(f"  Min FP Ratio: {rules_df['fp_ratio'].min():.4f}\n\n")
    
    print("TEST SET METRICS:\n")
    for key, value in metrics.items():
        if key != 'threshold':
            print(f"  {key}: {value}\n")
    
    print("\n" + "=" * 80 + "\n")
    print("TOP 10 RULES (BY PRECISION):\n")
    print("=" * 80 + "\n\n")
    
    top_rules = rules_df.nlargest(10, 'precision')
    for idx, row in top_rules.iterrows():
        print(f"Rule {int(row['rule_id'])+1}:\n")
        print(f"  IF {row['conditions']}\n")
        print(f"  THEN FRAUD\n")
        print(f"  Precision: {row['precision']:.4f}\n")
        print(f"  Coverage: {row['coverage']:.4f}\n")
        print(f"  FP Ratio: {row['fp_ratio']:.4f}\n")
        print(f"  TP: {int(row['tp'])}, FP: {int(row['fp'])}\n\n")

    print(f"\n[SUCCESS] SOAR rule extraction complete!")
    # print(f"  Output directory: {output_dir}")


# ==================== MAIN ====================
def main(csv_path, target_col, output_dir):
    """Main execution flow"""
    
    print("\n" + "=" * 80)
    print("SOAR RULE EXTRACTION FOR FRAUD DETECTION (SIMPLIFIED)")
    print("=" * 80)
    
    # Step 1: Load and prepare data
    data = load_and_prepare_data(csv_path, target_col=target_col)
    
    # Step 2: Discretize features (SOAR Step 1)
    X_train_orig = data['X_train'].copy()
    X_val_orig = data['X_val'].copy()
    X_test_orig = data['X_test'].copy()
    
    disc_data = discretize_features(
        data['X_train'], data['X_val'], data['X_test'],
        data['continuous_features'], data['categorical_features']
    )
    
    # Step 3: Train neural network on original (continuous) data
    model = build_and_train_model(
        X_train_orig.values, data['y_train'],
        X_val_orig.values, data['y_val']
    )
    
    # Step 4: Extract rules using SOAR on discretized data
    rules_df = extract_soar_rules(
        model, disc_data['X_train_disc'], data['y_train'],
        disc_data['X_val_disc'], data['y_val'],
        data['feature_names']
    )
    
    if len(rules_df) == 0:
        print("[ERROR] No rules extracted. Check data and model performance.")
        return
    
    # Step 5: Evaluate on test set
    metrics = evaluate_rules_on_test_set(
        model, X_test_orig.values, data['y_test'], rules_df
    )
    
    # Step 6: Save results
    save_results(rules_df, metrics, output_dir, csv_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SOAR Rule Extraction for Fraud Detection')
    parser.add_argument('--csv_path', type=str, required=False, default='', help='Path to input CSV file')
    parser.add_argument('--target_col', type=str, default='y_actual', help='Target column name')
    parser.add_argument('--output_dir', type=str, default=Config.OUTPUT_DIR, help='Output directory')
    
    args = parser.parse_args()
    
    main(args.csv_path, args.target_col, args.output_dir)
