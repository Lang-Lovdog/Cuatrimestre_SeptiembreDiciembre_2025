import json
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.base import clone
import random as rnd
from sklearn.model_selection import KFold
import numpy as np
import sys

def load_data(datasets_path, dataset_name=None):
    """Load test dataset with basic preprocessing"""
    file_path = Path(datasets_path) / dataset_name if dataset_name else Path(datasets_path)
    data = pd.read_csv(file_path)
    data = data[ data['Time_taken'].notna() ]
    d3_available_dummies = pd.get_dummies(data['3D_available'], prefix='3D_available',dtype=int).drop('3D_available_NO',axis=1)
    data = pd.concat([data.drop('3D_available',axis=1), d3_available_dummies], axis=1)
    genre_dummies = pd.get_dummies(data['Genre'], prefix='Genre',dtype=int).drop('Genre_Thriller',axis=1)
    data = pd.concat([data.drop('Genre', axis=1), genre_dummies], axis=1)

    return data, file_path.stem

def drop_columns(data, drop_flag):
    """Drop columns if drop_flag is True"""
    if drop_flag:
        drop_list = ['Twitter_hastags', 'Avg_age_actors', 'Marketing expense']
        for col in drop_list:
            if col in data.columns:
                data = data.drop(col, axis=1)
        print(f"  Dropped columns: {drop_list}")
    return data

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

def retrain_model(joblib_path, dataframe, dataset_name=""):
    """
    Retrains a model or pipeline loaded from a joblib file using a 70/30 train-test split.
    """
    # Load the object from the joblib file
    loaded_obj = joblib.load(joblib_path)

    # Get the model name
    model_name = Path(joblib_path).stem.split(".")[-1]

    # Assuming the target column is named "Start_Tech_Oscar"
    X = dataframe.drop(columns=["Start_Tech_Oscar"])
    y_true = dataframe["Start_Tech_Oscar"]

    # Split data 70/30
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_true, test_size=0.3,
        random_state=rnd.randint(20, 1000),
        metric=f1_score,
        stratify=y_true
    )

    # Use sklearn's clone to properly duplicate the estimator (works for both models and pipelines)
    retrained_obj = clone(loaded_obj)
    
    # Retrain on 70% of data
    retrained_obj.fit(X_train, y_train)

    # Evaluate on 30% test data
    y_pred = retrained_obj.predict(X_test)
    f1 = f1_score(y_test, y_pred)

    # Create the result dictionary
    result = {
        "model": model_name,
        "dataset": dataset_name,
        "f1_score": f1
    }

    return result

def evaluate_model(joblib_path, dataframe, dataset_name="", n_splits=5):
    """
    Evaluates a model loaded from a joblib file on a given DataFrame using K-Fold cross-validation
    and returns a dictionary with the model name, dataset name, and average F1 score.
    
    :param joblib_path: Path to the joblib file containing the model.
    :param dataframe: Pandas DataFrame containing the dataset.
    :param dataset_name: Name of the dataset (default is an empty string).
    :param n_splits: Number of folds for cross-validation (default is 5).
    :return: Dictionary with "model", "dataset", and "f1_score" keys.
    """
    # Load the model from the joblib file
    model = joblib.load(joblib_path)

    # Get the model name
    model_name = Path(joblib_path).stem.split(".")[-1]

    # Assuming the target column is named "Start_Tech_Oscar"
    X = dataframe.drop(columns=["Start_Tech_Oscar"])
    y_true = dataframe["Start_Tech_Oscar"]

    # Initialize K-Fold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=rnd.randint(20, 1000))
    
    f1_scores = []
    
    # Perform K-Fold cross-validation
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y_true.iloc[train_index], y_true.iloc[test_index]
        
        # Clone the model for this fold to ensure we start fresh each time
        fold_model = clone(model)
        
        # Train on training fold
        fold_model.fit(X_train, y_train)

        # Evaluate on test fold
        y_pred = fold_model.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        f1_scores.append(f1)

    # Calculate average F1 score across all folds
    avg_f1 = sum(f1_scores) / len(f1_scores)

    # Create the result dictionary
    result = {
        "model": model_name,
        "dataset": dataset_name,
        "f1_score": avg_f1
    }

    return result

def load_main_json(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    return data

def run_tests(json_file):
    global project_root
    conf = load_main_json(project_root / json_file)
    classifiers = get_classifier_paths(project_root / conf['models_json'])
    datasets = conf['datasets_names']
    datasets_path = project_root / conf['datasets_path']

    results_list = []

    for classifier in classifiers:
        for dataset in datasets:
            for a in range(20):
                print(f"  Testing {classifier} on {dataset} iteration {a}")
                joblib_path = classifiers[classifier]
                data, dataset_name = load_data(datasets_path, dataset)
                result = evaluate_model(joblib_path, data, dataset_name)
                #result = retrain_model(joblib_path, data, dataset_name)
                print(f"    {result}")
                results_list.append(result)

    results_dir = project_root / Path(conf['models_json']).parent / "test_results.csv"
    pd.DataFrame(results_list).to_csv(results_dir, index=False)

if __name__ == "__main__":
    # set dir as root of prject
    global project_root
    project_root = Path(__file__).parent.parent.parent.parent
    run_tests(project_root / "datasets.json")
