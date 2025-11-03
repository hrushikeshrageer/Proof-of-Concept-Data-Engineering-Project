import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_curve, f1_score, matthews_corrcoef
)
import warnings
warnings.filterwarnings('ignore')
from sklearn.tree import _tree
import graphviz

# Load your transactions dataset
df = ..........................

# Display basic information
print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nFraud distribution:")
print(df['fraud_label'].value_counts())
print("\nClass balance (%):")
print(df['fraud_label'].value_counts(normalize=True) * 100)


class FraudFeatureEngineer:
    """
    Class to handle feature engineering for fraud detection
    """
    
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_features = []
        self.categorical_features = []
        self.label_encoders = {}
    
    def identify_feature_types(self):
        """Identify numeric and categorical features"""
        numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target variable if present
        if 'fraud_label' in numeric_cols:
            numeric_cols.remove('fraud_label')
        if 'fraud_label' in categorical_cols:
            categorical_cols.remove('fraud_label')
        
        self.numeric_features = numeric_cols
        self.categorical_features = categorical_cols
        
        print(f"Numeric features ({len(numeric_cols)}): {numeric_cols}")
        print(f"\nCategorical features ({len(categorical_cols)}): {categorical_cols}")
        
        return numeric_cols, categorical_cols
    
    def handle_missing_values(self, strategy='mean'):
        """Handle missing values in the dataset"""
        # For numeric features
        for col in self.numeric_features:
            if self.df[col].isnull().sum() > 0:
                if strategy == 'mean':
                    self.df[col].fillna(self.df[col].mean(), inplace=True)
                elif strategy == 'median':
                    self.df[col].fillna(self.df[col].median(), inplace=True)
        
        # For categorical features
        for col in self.categorical_features:
            if self.df[col].isnull().sum() > 0:
                self.df[col].fillna(self.df[col].mode()[0], inplace=True)
        
        print("Missing values handled.")
        return self.df
    
    def encode_categorical_features(self):
        """Encode categorical features using LabelEncoder"""
        for col in self.categorical_features:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col].astype(str))
            self.label_encoders[col] = le
        
        print(f"Encoded {len(self.categorical_features)} categorical features.")
        return self.df
    
    def get_processed_data(self):
        """Pipeline to process all features"""
        self.identify_feature_types()
        self.handle_missing_values(strategy='median')
        self.encode_categorical_features()
        
        return self.df, self.numeric_features + self.categorical_features

# Usage
feature_engineer = FraudFeatureEngineer(df)
df_processed, all_features = feature_engineer.get_processed_data()

print("\nProcessed dataset shape:", df_processed.shape)
print("\nAll features for modeling:", all_features)

### 1.4 Data Splitting and Scaling

class DataPreprocessor:
    """
    Class to handle train-test split and scaling
    """
    
    def __init__(self, df, target_col, features, test_size=0.2, random_state=42):
        self.df = df
        self.target_col = target_col
        self.features = features
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_scaled = None
        self.X_test_scaled = None
    
    def split_data(self, stratify=True):
        """Split data into train and test sets"""
        X = self.df[self.features]
        y = self.df[self.target_col]
        
        if stratify:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, 
                test_size=self.test_size, 
                random_state=self.random_state,
                stratify=y  # Important for imbalanced datasets
            )
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y,
                test_size=self.test_size,
                random_state=self.random_state
            )
        
        print(f"Training set size: {self.X_train.shape}")
        print(f"Test set size: {self.X_test.shape}")
        print(f"Training fraud rate: {self.y_train.mean():.4f}")
        print(f"Test fraud rate: {self.y_test.mean():.4f}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def scale_features(self):
        """Scale features using StandardScaler"""
        # Note: Decision Trees don't require scaling, but other models do
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print("Features scaled using StandardScaler.")
        return self.X_train_scaled, self.X_test_scaled
    
    def get_data(self, scaled=False):
        """Get train-test split data"""
        if scaled:
            return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test
        else:
            return self.X_train, self.X_test, self.y_train, self.y_test

# Usage
preprocessor = DataPreprocessor(
    df_processed, 
    target_col='fraud_label', 
    features=all_features,
    test_size=0.2,
    random_state=42
)

X_train, X_test, y_train, y_test = preprocessor.split_data(stratify=True)

# Note: For Decision Trees, scaling is not necessary, but included for completeness
X_train_scaled, X_test_scaled = preprocessor.scale_features()

### 1.5 Model Training - Decision Tree

class FraudDetectionModels:
    """
    Class to train and evaluate tree-based models
    """
    
    def __init__(self, X_train, X_test, y_train, y_test, feature_names):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = feature_names
        self.models = {}
    
    def train_decision_tree(self, max_depth=10, min_samples_split=20, 
                            min_samples_leaf=10, class_weight='balanced'):
        """
        Train a Decision Tree Classifier
        
        Parameters:
        - max_depth: Controls tree depth (prevents overfitting)
        - min_samples_split: Minimum samples required to split
        - min_samples_leaf: Minimum samples in leaf node
        - class_weight: 'balanced' handles imbalanced data
        """
        dt_model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=42,
            criterion='entropy'  # or 'gini'
        )
        
        dt_model.fit(self.X_train, self.y_train)
        self.models['decision_tree'] = dt_model
        
        # Evaluate
        train_score = dt_model.score(self.X_train, self.y_train)
        test_score = dt_model.score(self.X_test, self.y_test)
        
        print("=" * 60)
        print("DECISION TREE CLASSIFIER")
        print("=" * 60)
        print(f"Training Accuracy: {train_score:.4f}")
        print(f"Test Accuracy: {test_score:.4f}")
        
        # Additional metrics
        y_pred = dt_model.predict(self.X_test)
        y_pred_proba = dt_model.predict_proba(self.X_test)[:, 1]
        
        print(f"Test ROC-AUC Score: {roc_auc_score(self.y_test, y_pred_proba):.4f}")
        print(f"Test F1 Score: {f1_score(self.y_test, y_pred):.4f}")
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred, 
                                   target_names=['Non-Fraud', 'Fraud']))
        
        return dt_model
    
    def train_random_forest(self, n_estimators=100, max_depth=15, 
                           min_samples_split=20, class_weight='balanced'):
        """Train a Random Forest Classifier"""
        rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            class_weight=class_weight,
            random_state=42,
            n_jobs=-1
        )
        
        rf_model.fit(self.X_train, self.y_train)
        self.models['random_forest'] = rf_model
        
        train_score = rf_model.score(self.X_train, self.y_train)
        test_score = rf_model.score(self.X_test, self.y_test)
        
        print("\n" + "=" * 60)
        print("RANDOM FOREST CLASSIFIER")
        print("=" * 60)
        print(f"Training Accuracy: {train_score:.4f}")
        print(f"Test Accuracy: {test_score:.4f}")
        
        y_pred = rf_model.predict(self.X_test)
        y_pred_proba = rf_model.predict_proba(self.X_test)[:, 1]
        
        print(f"Test ROC-AUC Score: {roc_auc_score(self.y_test, y_pred_proba):.4f}")
        print(f"Test F1 Score: {f1_score(self.y_test, y_pred):.4f}")
        
        return rf_model
    
    def get_feature_importance(self, model_name='decision_tree'):
        """Get and display feature importance"""
        model = self.models[model_name]
        importances = model.feature_importances_
        
        # Create dataframe for better visualization
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 6 Most Important Features ({model_name}):")
        print(importance_df.head(6))
        
        # Visualization
        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance_df.head(6), x='importance', y='feature')
        plt.title(f'Top 6 Feature Importances ({model_name.replace("_", " ").title()})')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.savefig(f'{model_name}_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return importance_df

# Usage
models = FraudDetectionModels(
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
    feature_names=all_features
)

# Train models
dt_model = models.train_decision_tree(max_depth=10, min_samples_split=20)
rf_model = models.train_random_forest(n_estimators=100, max_depth=15)

# Get feature importance
importance_df = models.get_feature_importance('decision_tree')




### 1.6 Extracting Rules from Decision Tree
class TreeRuleExtractor:
    """
    Extract business rules and feature ranges from trained Decision Trees
    """
    
    def __init__(self, tree_model, feature_names):
        self.tree_model = tree_model
        self.feature_names = feature_names
        self.rules = []
    
    def extract_rules_text(self):
        """Extract rules in text format"""
        tree_rules = export_text(self.tree_model, feature_names=self.feature_names)
        print("\n" + "=" * 60)
        print("DECISION TREE RULES (Text Format)")
        print("=" * 60)
        print(tree_rules)
        return tree_rules
    
    def extract_path_to_leaf(self):
        """
        Extract all decision paths from root to leaves
        Returns feature values and ranges for fraud classification
        """
        tree = self.tree_model.tree_
        feature_names = self.feature_names
        
        paths = []
        
        def recurse(node, path, depth):
            if tree.feature[node] != _tree.TREE_UNDEFINED:
                # Internal node
                threshold = tree.threshold[node]
                feature_idx = tree.feature[node]
                feature_name = feature_names[feature_idx]
                
                # Left child (<=)
                left_path = path + [f"{feature_name} <= {threshold:.4f}"]
                recurse(tree.children_left[node], left_path, depth + 1)
                
                # Right child (>)
                right_path = path + [f"{feature_name} > {threshold:.4f}"]
                recurse(tree.children_right[node], right_path, depth + 1)
            else:
                # Leaf node
                samples = tree.n_node_samples[node]
                value = tree.value[node][0]
                fraud_count = int(value[1])
                non_fraud_count = int(value[0])
                fraud_rate = fraud_count / samples if samples > 0 else 0
                
                paths.append({
                    'conditions': path,
                    'total_samples': samples,
                    'fraud_count': fraud_count,
                    'non_fraud_count': non_fraud_count,
                    'fraud_rate': fraud_rate,
                    'prediction': 'FRAUD' if fraud_count > non_fraud_count else 'NON-FRAUD'
                })
        
        recurse(0, [], 0)
        self.rules = paths
        return paths
    
    def get_fraud_rules(self, min_fraud_rate=0.5, min_samples=10):
        """
        Get rules that predict FRAUD with minimum fraud rate
        
        Parameters:
        - min_fraud_rate: Minimum fraud rate in the leaf to consider it a fraud rule
        - min_samples: Minimum samples in leaf for rule validity
        """
        fraud_rules = [
            rule for rule in self.rules
            if rule['fraud_rate'] >= min_fraud_rate and rule['total_samples'] >= min_samples
        ]
        
        # Sort by fraud rate (descending)
        fraud_rules = sorted(fraud_rules, key=lambda x: x['fraud_rate'], reverse=True)
        
        print("\n" + "=" * 60)
        print("FRAUD DETECTION RULES")
        print("=" * 60)
        print(f"Found {len(fraud_rules)} rules flagging transactions as fraud\n")
        
        for idx, rule in enumerate(fraud_rules, 1):
            print(f"Rule {idx}: (Fraud Rate: {rule['fraud_rate']:.2%})")
            print(f"  Samples: {rule['total_samples']} (Fraud: {rule['fraud_count']}, Non-Fraud: {rule['non_fraud_count']})")
            print("  Conditions:")
            for condition in rule['conditions']:
                print(f"    - {condition}")
            print()
        
        return fraud_rules
    
    def get_non_fraud_rules(self, min_non_fraud_rate=0.9, min_samples=10):
        """
        Get rules that predict NON-FRAUD with high confidence
        """
        non_fraud_rules = [
            rule for rule in self.rules
            if (1 - rule['fraud_rate']) >= min_non_fraud_rate and rule['total_samples'] >= min_samples
        ]
        
        non_fraud_rules = sorted(non_fraud_rules, 
                                 key=lambda x: (1 - x['fraud_rate']), 
                                 reverse=True)
        
        print("\n" + "=" * 60)
        print("NON-FRAUD RULES (SAFE TRANSACTIONS)")
        print("=" * 60)
        print(f"Found {len(non_fraud_rules)} rules flagging transactions as non-fraud\n")
        
        for idx, rule in enumerate(non_fraud_rules[:10], 1):  # Show top 10
            print(f"Rule {idx}: (Non-Fraud Rate: {(1-rule['fraud_rate']):.2%})")
            print(f"  Samples: {rule['total_samples']} (Fraud: {rule['fraud_count']}, Non-Fraud: {rule['non_fraud_count']})")
            print("  Conditions:")
            for condition in rule['conditions']:
                print(f"    - {condition}")
            print()
        
        return non_fraud_rules
    
    def create_rule_dataframe(self):
        """Convert rules to DataFrame for easier analysis"""
        rules_data = []
        
        for rule in self.rules:
            conditions_str = " AND ".join(rule['conditions'])
            rules_data.append({
                'conditions': conditions_str,
                'total_samples': rule['total_samples'],
                'fraud_count': rule['fraud_count'],
                'non_fraud_count': rule['non_fraud_count'],
                'fraud_rate': rule['fraud_rate'],
                'prediction': rule['prediction']
            })
        
        rules_df = pd.DataFrame(rules_data)
        return rules_df.sort_values('fraud_rate', ascending=False)

# Usage
rule_extractor = TreeRuleExtractor(dt_model, all_features)

# Extract rules in text format
text_rules = rule_extractor.extract_rules_text()

# Extract all paths and decision rules
all_rules = rule_extractor.extract_path_to_leaf()

# Get fraud detection rules
fraud_rules = rule_extractor.get_fraud_rules(min_fraud_rate=0.6, min_samples=15)

# Get non-fraud rules
non_fraud_rules = rule_extractor.get_non_fraud_rules(min_non_fraud_rate=0.85, min_samples=15)

# Convert to DataFrame for analysis
rules_df = rule_extractor.create_rule_dataframe()
print("\n" + "=" * 60)
print("SUMMARY OF ALL DECISION PATHS")
print("=" * 60)
print(rules_df.head(20))

# Save rules to CSV
rules_df.to_csv('extracted_rules.csv', index=False)
print("\nRules saved to 'extracted_rules.csv'")

### 1.7 Feature Range Analysis

class FeatureRangeAnalyzer:
    """
    Analyze feature ranges and thresholds that trigger fraud/non-fraud predictions
    """
    
    def __init__(self, X_train, X_test, y_train, y_test, feature_names):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = feature_names
    
    def analyze_feature_distributions(self):
        """
        Analyze how features are distributed for fraud vs non-fraud
        """
        print("\n" + "=" * 60)
        print("FEATURE DISTRIBUTION ANALYSIS")
        print("=" * 60)
        
        X_combined = pd.concat([self.X_train, self.X_test], axis=0)
        y_combined = pd.concat([self.y_train, self.y_test], axis=0)
        X_combined['fraud_label'] = y_combined.values
        
        analysis_results = []
        
        for feature in self.feature_names:
            fraud_data = X_combined[X_combined['fraud_label'] == 1][feature]
            non_fraud_data = X_combined[X_combined['fraud_label'] == 0][feature]
            
            analysis_results.append({
                'feature': feature,
                'fraud_mean': fraud_data.mean(),
                'fraud_median': fraud_data.median(),
                'fraud_std': fraud_data.std(),
                'fraud_min': fraud_data.min(),
                'fraud_max': fraud_data.max(),
                'non_fraud_mean': non_fraud_data.mean(),
                'non_fraud_median': non_fraud_data.median(),
                'non_fraud_std': non_fraud_data.std(),
                'non_fraud_min': non_fraud_data.min(),
                'non_fraud_max': non_fraud_data.max(),
                'mean_difference': abs(fraud_data.mean() - non_fraud_data.mean())
            })
        
        analysis_df = pd.DataFrame(analysis_results)
        analysis_df = analysis_df.sort_values('mean_difference', ascending=False)
        
        print("\nTop features with largest differences between Fraud and Non-Fraud:")
        print(analysis_df[['feature', 'fraud_mean', 'non_fraud_mean', 'mean_difference']].head(15))
        
        return analysis_df
    
    def extract_threshold_ranges(self):
        """
        Extract ranges from decision tree thresholds
        """
        print("\n" + "=" * 60)
        print("DECISION TREE THRESHOLD RANGES")
        print("=" * 60)
        
        # Get thresholds from tree
        thresholds_by_feature = {}
        
        def get_thresholds(node, tree):
            if tree.feature[node] != _tree.TREE_UNDEFINED:
                feature_idx = tree.feature[node]
                feature_name = self.feature_names[feature_idx]
                threshold = tree.threshold[node]
                
                if feature_name not in thresholds_by_feature:
                    thresholds_by_feature[feature_name] = []
                
                thresholds_by_feature[feature_name].append(threshold)
                
                # Recurse
                get_thresholds(tree.children_left[node], tree)
                get_thresholds(tree.children_right[node], tree)
        
        # This would need a model - assuming dt_model is available
        # get_thresholds(0, dt_model.tree_)
        
        return thresholds_by_feature
    
    def visualize_feature_distributions(self, top_n=10):
        """
        Visualize distributions of top features for fraud vs non-fraud
        """
        X_combined = pd.concat([self.X_train, self.X_test], axis=0)
        y_combined = pd.concat([self.y_train, self.y_test], axis=0)
        X_combined['fraud_label'] = y_combined.values
        
        analysis_df = self.analyze_feature_distributions()
        top_features = analysis_df.head(top_n)['feature'].tolist()
        
        fig, axes = plt.subplots((top_n + 1) // 2, 2, figsize=(15, 5 * ((top_n + 1) // 2)))
        axes = axes.flatten()
        
        for idx, feature in enumerate(top_features):
            ax = axes[idx]
            
            fraud_data = X_combined[X_combined['fraud_label'] == 1][feature]
            non_fraud_data = X_combined[X_combined['fraud_label'] == 0][feature]
            
            ax.hist(non_fraud_data, bins=30, alpha=0.6, label='Non-Fraud', color='green')
            ax.hist(fraud_data, bins=30, alpha=0.6, label='Fraud', color='red')
            ax.set_title(f'{feature} Distribution')
            ax.set_xlabel(feature)
            ax.set_ylabel('Frequency')
            ax.legend()
        
        # Remove extra subplots
        for idx in range(len(top_features), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        plt.savefig('feature_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()

# Usage
feature_analyzer = FeatureRangeAnalyzer(
    X_train, X_test, y_train, y_test, all_features
)

analysis_df = feature_analyzer.analyze_feature_distributions()
feature_analyzer.visualize_feature_distributions(top_n=10)

### 1.8 Model Visualization and Performance

def visualize_decision_tree(model, feature_names, max_depth_display=3):
    """
    Visualize the decision tree structure
    """
    plt.figure(figsize=(20, 10))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=['Non-Fraud', 'Fraud'],
        filled=True,
        max_depth=max_depth_display,
        fontsize=10
    )
    plt.title('Decision Tree for Fraud Detection (Limited Depth for Clarity)')
    plt.tight_layout()
    plt.savefig('decision_tree_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Fraud', 'Fraud'],
                yticklabels=['Non-Fraud', 'Fraud'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_roc_curve(y_true, y_pred_proba):
    """Plot ROC curve"""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    auc_score = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
    plt.show()

# Usage
visualize_decision_tree(dt_model, all_features, max_depth_display=4)

y_pred = dt_model.predict(X_test)
y_pred_proba = dt_model.predict_proba(X_test)[:, 1]

plot_confusion_matrix(y_test, y_pred)
plot_roc_curve(y_test, y_pred_proba)