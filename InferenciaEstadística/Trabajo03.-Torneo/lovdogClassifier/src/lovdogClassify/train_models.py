import os
import pandas as pd #type: ignore
import numpy as np #type: ignore
from pathlib import Path
import sys

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from lovdogClassifier.LovdogModelSelector.LovdogMS import ModelSelection #type: ignore

# Import all required classifiers
from sklearn.preprocessing import RobustScaler        #type: ignore
from sklearn.preprocessing import QuantileTransformer #type: ignore
from sklearn.preprocessing import PowerTransformer    #type: ignore
from sklearn.ensemble import IsolationForest          #type: ignore

from GridInstances import instances as grid_instances


def adaptive_outlier_preprocessing(data, method='robust_scaling', **kwargs):
    """
    Adaptive outlier preprocessing with multiple methods
    
    Parameters:
    - method: 'robust_scaling', 'winsorize', 'quantile', 'power', 'isolation_forest'
    - **kwargs: method-specific parameters
    """
    numerical_cols = data.select_dtypes(include=[np.number]).columns

    print(f"Applying {method} preprocessing...")
    
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
        output_dist = kwargs.get('output_distribution', 'normal')
        transformer = QuantileTransformer(
            n_quantiles=n_quantiles,
            output_distribution=output_dist,
            random_state=42
        )
        data[numerical_cols] = transformer.fit_transform(data[numerical_cols])
        
    elif method == 'power':
        method_type = kwargs.get('method', 'yeo-johnson')
        transformer = PowerTransformer(method=method_type)
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
    
    return data

def drop_em(data, droplist):
    for col in droplist:
        if col in data.columns:
            print(f"Dropping column: {col}")
            data = data.drop(col, axis=1)
    return data

def preprocess_movie_dataset(data):
    """Custom preprocessing for the movie dataset"""
    print("Applying movie dataset-specific preprocessing...")
    print("Dropping nan rows...")
    data = data[ data['Time_taken'].notna() ]
    print("3D_available column into binary integer dummy...")
    d3_available_dummies = pd.get_dummies(data['3D_available'], prefix='3D_available',dtype=int).drop('3D_available_NO',axis=1)
    data = pd.concat([data.drop('3D_available',axis=1), d3_available_dummies], axis=1)
    print("Genre column into binary integer dummies...")
    genre_dummies = pd.get_dummies(data['Genre'], prefix='Genre',dtype=int).drop('Genre_Thriller',axis=1)
    data = pd.concat([data.drop('Genre', axis=1), genre_dummies], axis=1)
    print(f"Preprocessed data shape: {data.shape}")
    return data

def train_separate_instances():
    """Train each grid instance completely separately"""
    
    DATASETS_PATH = project_root / "lovdogClassifier" / "res" / "datasets"
    RESULTS_BASE = project_root / "results"
    
    # Use first dataset for training
    training_file = DATASETS_PATH / "shuffle_01.csv"
    
    grid_best_overview = pd.DataFrame(index=grid_instances.keys(),columns=['Classifier', 'f1_score'])
    
    for instance_name, instance_config in grid_instances.items():
        print(f"\n{'='*60}")
        print(f"TRAINING INSTANCE: {instance_name}")
        print(f"{'='*60}")
        
        # Create separate results directory for each instance
        instance_results_path = RESULTS_BASE / instance_name
        instance_models_path = instance_results_path / "trained_models"
        instance_plots_path = instance_results_path / "plots"
        
        os.makedirs(instance_results_path, exist_ok=True)
        os.makedirs(instance_models_path, exist_ok=True)
        os.makedirs(instance_plots_path, exist_ok=True)
        
        # Create ModelSelection instance for this specific grid
        ms = ModelSelection(
            name=f"instance_{instance_name}",
            csv_path=str(training_file),
            target_column="Start_Tech_Oscar",
            test_size=0.3,
            classifiers_dict=instance_config['classifiers'],
            grid_parameters_dict=instance_config['grid_params'],
            best_only_mode=True,
            plots_output_directory=str(instance_plots_path),
            custom_data_preprocessing=preprocess_movie_dataset
        )
        
        # Train models for this instance
        ms.debug_data()
        ms.selectModels()
        
        # Save instance-specific results
        instance_results = ms.getMetricsDataFrame()
        instance_results.to_csv(instance_results_path / "training_metrics.csv", index=False)
        
        # Save models for this instance
        ms.saveModels(str(instance_models_path))
        
        # Generate plots for this instance
        ms.generatePlots()
        
        # Save instance summary
        best_model = instance_results.loc[instance_results['F1_Score'].idxmax()]
        with open(instance_results_path / "instance_summary.txt", "w") as f:
            f.write(f"INSTANCE: {instance_name}\n")
            f.write("=" * 40 + "\n")
            f.write(f"Best Model: {best_model['Classifier']}\n")
            f.write(f"F1 Score: {best_model['F1_Score']:.4f}\n")
            f.write(f"Accuracy: {best_model['Accuracy']:.4f}\n")
            f.write(f"Precision: {best_model['Precision']:.4f}\n")
            f.write(f"Recall: {best_model['Recall']:.4f}\n")
            if best_model['AUC_Score'] != 'N/A':
                f.write(f"AUC Score: {best_model['AUC_Score']:.4f}\n")
            f.write(f"CV Score: {best_model['CV_Score']:.4f}\n")
            f.write(f"\nBest Parameters:\n{best_model['Best_Params']}\n")
        
        print(f"✅ {instance_name} completed")
        print(f"   Best Model: {best_model['Classifier']}")
        print(f"   F1 Score: {best_model['F1_Score']:.4f}")
        print(f"   Results saved to: {instance_results_path}")

        grid_best_overview.loc[instance_name, 'Classifier'] = best_model['Classifier']
        grid_best_overview.loc[instance_name, 'f1_score'] = best_model['F1_Score']

    grid_best_overview.to_csv(RESULTS_BASE / "grid_best_overview.csv")
    
    print(f"\n🎯 All instances trained separately!")
    print(f"📁 Each instance has its own directory in: {RESULTS_BASE}")

if __name__ == "__main__":
    train_separate_instances()
