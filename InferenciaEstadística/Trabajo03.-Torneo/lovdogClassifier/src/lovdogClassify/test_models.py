import json
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import f1_score
import numpy as np
import sys

# Add project root to path to access modules
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

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
    """Apply outlier preprocessing (same as training)"""
    return data
    if method == 'none':
        return data
        
    numerical_cols = data.select_dtypes(include=[np.number]).columns
    print(f"  Applying {method} preprocessing...")
    
    if method == 'robust_scaling':
        from sklearn.preprocessing import RobustScaler
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
        from sklearn.preprocessing import QuantileTransformer
        n_quantiles = kwargs.get('n_quantiles', 5)
        transformer = QuantileTransformer(n_quantiles=n_quantiles, random_state=42)
        data[numerical_cols] = transformer.fit_transform(data[numerical_cols])
        
    elif method == 'isolation_forest':
        from sklearn.ensemble import IsolationForest
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

def run_tests():
    """Run all tests and generate results"""
    
    # Load test configuration
    config_path = project_root / "test_elements.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Initialize results storage
    all_results = []
    
    # Create results directory
    results_dir = project_root / "model_test_results"
    results_dir.mkdir(exist_ok=True)
    
    # Test each model configuration
    for family, models in config['models'].items():
        print(f"\n{'='*60}")
        print(f"Testing {family}")
        print(f"{'='*60}")
        
        family_results = []
        
        for model_config in models:
            preprocessing = model_config['preprocessing']
            classifier = model_config['classifier']
            drop = model_config['drop']
            
            print(f"  Testing {classifier} with {preprocessing} (drop: {drop})")
            
            # Load the trained model
            model = load_model(family, preprocessing, classifier, drop)
            if model is None:
                continue
            
            # Test on each dataset
            for dataset_name in config['test_datasets_names']:
                print(f"    Dataset: {dataset_name}")
                
                try:
                    # Load and preprocess test data
                    datasets_path = project_root / config['test_datasets_path']
                    test_data = load_test_data(datasets_path, dataset_name)
                    test_data = drop_columns(test_data, drop)
                    test_data = adaptive_outlier_preprocessing(test_data, method=preprocessing)
                    
                    # Prepare features and target
                    X_test = test_data.drop('Start_Tech_Oscar', axis=1)
                    y_test = test_data['Start_Tech_Oscar']
                    
                    # Make predictions and calculate F1 score
                    y_pred = model.predict(X_test)
                    f1 = f1_score(y_test, y_pred)
                    
                    # Store result
                    result = {
                        'Model': classifier,
                        'Dataset': dataset_name,
                        'Preprocessing': preprocessing,
                        'Columns_Dropped': drop,
                        'F1_Score': f1
                    }
                    family_results.append(result)
                    all_results.append(result)
                    
                    print(f"      F1 Score: {f1:.4f}")
                    
                except Exception as e:
                    print(f"      ❌ Error testing {classifier} on {dataset_name}: {e}")
                    continue
        
        # Save family results
        if family_results:
            family_df = pd.DataFrame(family_results)
            # Order by Model > Dataset > Preprocessing
            family_df = family_df.sort_values(['Model', 'Dataset', 'Preprocessing'])
            
            # Save family CSV
            family_df.to_csv(results_dir / f"{family}_results.csv", index=False)
            print(f"  ✅ Saved {family}_results.csv")
    
    # Create summary of best models
    if all_results:
        summary_df = pd.DataFrame(all_results)
        
        # Find best model from each family (highest average F1 score)
        best_models = []
        for family in config['models'].keys():
            family_models = [m['classifier'] for m in config['models'][family]]
            family_data = summary_df[summary_df['Model'].isin(family_models)]
            
            if not family_data.empty:
                # Calculate average F1 score for each model in this family
                model_avg_f1 = family_data.groupby('Model')['F1_Score'].mean().reset_index()
                best_model_row = model_avg_f1.loc[model_avg_f1['F1_Score'].idxmax()]
                best_model_name = best_model_row['Model']
                
                # Get the full configuration for the best model
                best_model_config = next(
                    (m for m in config['models'][family] if m['classifier'] == best_model_name), 
                    None
                )
                
                if best_model_config:
                    # Get all results for this best model
                    best_model_results = summary_df[
                        (summary_df['Model'] == best_model_name) & 
                        (summary_df['Preprocessing'] == best_model_config['preprocessing']) &
                        (summary_df['Columns_Dropped'] == best_model_config['drop'])
                    ]
                    
                    if not best_model_results.empty:
                        # Take the first row (all will have same config) and add average F1
                        best_row = best_model_results.iloc[0].copy()
                        best_row['F1_Score'] = best_model_row['F1_Score']  # Use average F1
                        best_models.append(best_row)
        
        # Create summary dataframe and sort by F1 score
        if best_models:
            summary_best_df = pd.DataFrame(best_models)
            summary_best_df = summary_best_df.sort_values('F1_Score', ascending=False)
            
            # Save summary
            summary_best_df.to_csv(results_dir / "summary_best_models.csv", index=False)
            print(f"\n✅ Saved summary_best_models.csv")
        
        # Also save complete results
        complete_df = pd.DataFrame(all_results)
        complete_df = complete_df.sort_values(['Model', 'Dataset', 'Preprocessing'])
        complete_df.to_csv(results_dir / "complete_test_results.csv", index=False)
        print(f"✅ Saved complete_test_results.csv")
        
        print(f"\n🎯 All tests completed!")
        print(f"📁 Results saved to: {results_dir}")

if __name__ == "__main__":
    run_tests()
