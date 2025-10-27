import json
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import f1_score
import numpy as np
import sys
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer

# Add project root to path to access modules
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

def parse_json_to_strings(json_data, file=False):
    # Load the JSON data
    if file:
        with open(json_data, 'r') as f:
            json_data = f.read()

    data = json.loads(json_data)

    # Extract the models dictionary
    models = data["models"]

    # Initialize an empty list to store the results
    results = []

    # Iterate over each model family and its variants
    for model_family, variants in models.items():
        for variant in variants:
            preprocessing = variant["preprocessing"]
            classifier = variant["classifier"]
            drop = variant["drop"]

            # Construct the string in the desired format
            result_string = f"results{'_preprocessing' if preprocessing else ''}\
                {'_drop' if drop else ''}/{model_family}/{classifier}.joblib"

            # Append the result string to the list
            results.append(result_string)

    return results

import json

def get_classifier_paths(json_file_path):
    """
    Reads the global JSON file and returns a dictionary with instance names as keys
    and the paths to the best classifiers as values.
    
    :param json_file_path: Path to the global JSON file.
    :return: Dictionary with instance names as keys and classifier paths as values.
    """
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # Create a dictionary to store the results
    classifier_paths = {}

    # Iterate over the data and populate the dictionary
    for entry in data:
        instance_name = entry['instance']
        classifier_path = entry['path']
        classifier_paths[instance_name] = classifier_path

    return classifier_paths

def load_test_data(datasets_path, dataset_name):
    """Load test dataset with basic preprocessing"""
    file_path = Path(datasets_path) / dataset_name
    data = pd.read_csv(file_path)
    
    # Basic preprocessing (same as training)
    data = data.fillna(data.median(numeric_only=True))
    data['3D_available'] = data['3D_available'].map({'YES': 1, 'NO': 0})
    
    if 'Genre' in data.columns:
        genre_dummies = pd.get_dummies(data['Genre'], prefix='Genre')
        data = pd.concat([data.drop('Genre', axis=1), genre_dummies], axis=1)
    
    return data

def adaptive_outlier_preprocessing(data, method='none', **kwargs):
    """Apply outlier preprocessing (same as training) - FIXED VERSION"""
    if method == 'none':
        return data
        
    # Separate target variable to prevent scaling it
    target_column = 'Start_Tech_Oscar'
    y_data = None
    if target_column in data.columns:
        y_data = data[target_column].copy()
        data = data.drop(target_column, axis=1)
    
    numerical_cols = data.select_dtypes(include=[np.number]).columns
    print(f"  Applying {method} preprocessing...")
    
    if method == 'robust_scaling':
        scaler = RobustScaler()
        data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
        
    elif method == 'winsorize':
        lower_q = kwargs.get('lower_quantile', 0.25)
        upper_q = kwargs.get('upper_quantile', 0.75)
        for col in numerical_cols:
            lower_bound = data[col].quantile(lower_q)
            upper_bound = data[col].quantile(upper_q)
            data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)
            
    elif method == 'quantile':
        n_quantiles = kwargs.get('n_quantiles', 5)
        transformer = QuantileTransformer(n_quantiles=n_quantiles, random_state=42)
        data[numerical_cols] = transformer.fit_transform(data[numerical_cols])
        
    elif method == 'isolation_forest':
        contamination = kwargs.get('contamination', 0.1)
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        numerical_data = data[numerical_cols].copy()
        outlier_labels = iso_forest.fit_predict(numerical_data)
        
        for col in numerical_cols:
            mask = outlier_labels == -1
            if mask.any():
                median_val = numerical_data.loc[~mask, col].median()
                numerical_data.loc[mask, col] = median_val
        data[numerical_cols] = numerical_data
    
    # Re-add target variable if it was separated
    if y_data is not None:
        data[target_column] = y_data
    
    return data

def drop_columns(data, drop_flag):
    """Drop columns if drop_flag is True"""
    if drop_flag:
        drop_list = ['Twitter_hastags', 'Avg_age_actors', 'Marketing expense']
        for col in drop_list:
            if col in data.columns:
                data = data.drop(col, axis=1)
        print(f"  Dropped columns: {drop_list}")
    return data

def load_model(family, preprocessing, classifier, drop):
    """Load the trained model based on configuration"""
    if preprocessing == "none":
        preprocessing = "unclean"
    if drop:
        results_dir = project_root / f"results_{preprocessing}_drop"
    else:
        results_dir = project_root / f"results_{preprocessing}"
    
    model_path = results_dir / family / "trained_models" / f"{classifier}.joblib"
    
    if model_path.exists():
        print(f"    Loading model: {model_path}")
        return joblib.load(model_path)
    else:
        print(f"    ❌ Model not found: {model_path}")
        return None

def calculate_f1_score(y_true, y_pred):
    """Calculate F1 score with appropriate averaging - ROBUST VERSION"""
    # Ensure targets are in the correct format [0, 1]
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Handle case where target might have negative values
    if set(np.unique(y_true)).issubset({-1, 0}):
        # Convert back to {0, 1} for proper binary classification
        y_true = np.where(y_true == -1, 1, 0)
        y_pred = np.where(y_pred == -1, 1, 0)
    
    unique_classes = len(np.unique(y_true))
    
    if unique_classes == 2:
        # Binary classification
        return f1_score(y_true, y_pred, average='binary')
    else:
        # Multiclass classification - use weighted average
        return f1_score(y_true, y_pred, average='weighted')
