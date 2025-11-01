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