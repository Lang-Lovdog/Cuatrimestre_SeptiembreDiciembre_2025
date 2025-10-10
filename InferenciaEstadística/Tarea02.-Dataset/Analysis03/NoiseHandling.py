import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.backends.backend_pdf import PdfPages
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
import warnings
warnings.filterwarnings('ignore')

class AdvancedDataNoiseHandler:
    def __init__(self, file_path):
        """
        Initialize the AdvancedDataNoiseHandler with a CSV file path
        """
        self.original_data = pd.read_csv(file_path)
        self.file_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # Display basic info about the dataset
        print(f"Dataset loaded: {self.file_name}")
        print(f"Shape: {self.original_data.shape}")
        print(f"Columns: {list(self.original_data.columns)}")
        print(f"Data types:\n{self.original_data.dtypes}")
        print(f"\nMissing values:\n{self.original_data.isnull().sum()}")
        
        # Analyze data types
        self.numeric_columns = self.original_data.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = self.original_data.select_dtypes(include=['object']).columns.tolist()
        self.bool_columns = self.original_data.select_dtypes(include=['bool']).columns.tolist()
        
        print(f"\nNumeric columns: {self.numeric_columns}")
        print(f"Categorical columns: {self.categorical_columns}")
        print(f"Boolean columns: {self.bool_columns}")
    
    def detect_outliers_iqr(self, data, column, threshold=1.5):
        """
        Detect outliers using Interquartile Range (IQR) method
        """
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
        return outliers, lower_bound, upper_bound
    
    def detect_outliers_zscore(self, data, column, threshold=3):
        """
        Detect outliers using Z-score method
        """
        z_scores = np.abs(stats.zscore(data[column].dropna()))
        outlier_indices = np.where(z_scores > threshold)[0]
        outliers = data.iloc[outlier_indices]
        return outliers, threshold
    
    def detect_outliers_mad(self, data, column, threshold=3):
        """
        Detect outliers using Median Absolute Deviation (MAD) method
        """
        median = data[column].median()
        mad = np.median(np.abs(data[column] - median))
        modified_z_scores = 0.6745 * (data[column] - median) / mad
        outlier_indices = np.where(np.abs(modified_z_scores) > threshold)[0]
        outliers = data.iloc[outlier_indices]
        return outliers, threshold
    
    def handle_outliers_iqr(self, data, columns=None, threshold=1.5):
        """
        Handle outliers using IQR method (capping)
        """
        if columns is None:
            columns = self.numeric_columns
        
        cleaned_data = data.copy()
        
        for column in columns:
            if column in data.columns and column in self.numeric_columns:
                Q1 = data[column].quantile(0.25)
                Q3 = data[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                # Cap the outliers
                cleaned_data[column] = np.where(cleaned_data[column] < lower_bound, lower_bound, cleaned_data[column])
                cleaned_data[column] = np.where(cleaned_data[column] > upper_bound, upper_bound, cleaned_data[column])
        
        return cleaned_data
    
    def handle_outliers_zscore(self, data, columns=None, threshold=3, method='remove'):
        """
        Handle outliers using Z-score method
        method: 'remove' or 'cap'
        """
        if columns is None:
            columns = self.numeric_columns
        
        cleaned_data = data.copy()
        
        for column in columns:
            if column in data.columns and column in self.numeric_columns:
                # Handle missing values first
                non_null_data = data[column].dropna()
                if len(non_null_data) == 0:
                    continue
                    
                z_scores = np.abs(stats.zscore(non_null_data))
                outlier_indices = non_null_data.index[z_scores > threshold]
                
                if method == 'remove':
                    # Remove outliers
                    cleaned_data = cleaned_data.drop(index=outlier_indices)
                elif method == 'cap':
                    # Cap outliers using median
                    median_val = data[column].median()
                    std_val = data[column].std()
                    upper_bound = median_val + threshold * std_val
                    lower_bound = median_val - threshold * std_val
                    
                    cleaned_data.loc[cleaned_data[column] > upper_bound, column] = upper_bound
                    cleaned_data.loc[cleaned_data[column] < lower_bound, column] = lower_bound
        
        return cleaned_data
    
    def handle_outliers_mad(self, data, columns=None, threshold=3, method='remove'):
        """
        Handle outliers using MAD method
        """
        if columns is None:
            columns = self.numeric_columns
        
        cleaned_data = data.copy()
        
        for column in columns:
            if column in data.columns and column in self.numeric_columns:
                # Handle missing values first
                non_null_data = data[column].dropna()
                if len(non_null_data) == 0:
                    continue
                    
                median = non_null_data.median()
                mad = np.median(np.abs(non_null_data - median))
                
                if mad == 0:  # Avoid division by zero
                    continue
                    
                modified_z_scores = 0.6745 * (non_null_data - median) / mad
                outlier_indices = non_null_data.index[np.abs(modified_z_scores) > threshold]
                
                if method == 'remove':
                    cleaned_data = cleaned_data.drop(index=outlier_indices)
                elif method == 'cap':
                    # Cap using median ± threshold * MAD / 0.6745
                    upper_bound = median + threshold * mad / 0.6745
                    lower_bound = median - threshold * mad / 0.6745
                    
                    cleaned_data.loc[cleaned_data[column] > upper_bound, column] = upper_bound
                    cleaned_data.loc[cleaned_data[column] < lower_bound, column] = lower_bound
        
        return cleaned_data
    
    def handle_outliers_isolation_forest(self, data, columns=None, contamination=0.1):
        """
        Handle outliers using Isolation Forest (machine learning approach)
        """
        from sklearn.ensemble import IsolationForest
        
        if columns is None:
            columns = self.numeric_columns
        
        # Select only numeric columns that exist in data
        available_columns = [col for col in columns if col in data.columns]
        if not available_columns:
            return data.copy()
            
        numeric_data = data[available_columns].copy()
        
        # Handle missing values
        numeric_data = numeric_data.fillna(numeric_data.median())
        
        # Initialize and fit Isolation Forest
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outlier_labels = iso_forest.fit_predict(numeric_data)
        
        # Remove outliers (outlier_labels == -1 are outliers, == 1 are inliers)
        cleaned_data = data[outlier_labels == 1].copy()
        
        return cleaned_data
    
    def handle_missing_values(self, data, method='median', columns=None):
        """
        Handle missing values using different imputation methods
        methods: 'remove', 'mean', 'median', 'mode', 'knn'
        """
        cleaned_data = data.copy()
        
        if method == 'remove':
            cleaned_data = cleaned_data.dropna()
        else:
            if columns is None:
                columns = self.numeric_columns + self.categorical_columns
            
            for column in columns:
                if column in cleaned_data.columns and cleaned_data[column].isnull().sum() > 0:
                    if method == 'mean' and column in self.numeric_columns:
                        cleaned_data[column] = cleaned_data[column].fillna(cleaned_data[column].mean())
                    elif method == 'median' and column in self.numeric_columns:
                        cleaned_data[column] = cleaned_data[column].fillna(cleaned_data[column].median())
                    elif method == 'mode':
                        # For both numeric and categorical
                        mode_val = cleaned_data[column].mode()
                        if len(mode_val) > 0:
                            cleaned_data[column] = cleaned_data[column].fillna(mode_val[0])
                    elif method == 'knn' and column in self.numeric_columns:
                        # Use KNN imputer for numeric columns
                        knn_imputer = KNNImputer(n_neighbors=5)
                        numeric_cols = [col for col in self.numeric_columns if col in cleaned_data.columns]
                        if numeric_cols:
                            cleaned_data[numeric_cols] = knn_imputer.fit_transform(cleaned_data[numeric_cols])
                        break  # KNN imputer handles all columns at once
        
        return cleaned_data
    
    def handle_categorical_encoding(self, data, method='onehot', columns=None):
        """
        Handle categorical variables encoding
        methods: 'onehot', 'label', 'frequency'
        """
        cleaned_data = data.copy()
        
        if columns is None:
            columns = self.categorical_columns
        
        for column in columns:
            if column in cleaned_data.columns:
                if method == 'onehot':
                    # One-hot encoding
                    dummies = pd.get_dummies(cleaned_data[column], prefix=column)
                    cleaned_data = pd.concat([cleaned_data, dummies], axis=1)
                    cleaned_data = cleaned_data.drop(column, axis=1)
                
                elif method == 'label':
                    # Label encoding
                    le = LabelEncoder()
                    cleaned_data[column] = le.fit_transform(cleaned_data[column].astype(str))
                
                elif method == 'frequency':
                    # Frequency encoding
                    freq_encoding = cleaned_data[column].value_counts().to_dict()
                    cleaned_data[column] = cleaned_data[column].map(freq_encoding)
        
        return cleaned_data
    
    def handle_duplicates(self, data, method='remove'):
        """
        Handle duplicate rows
        methods: 'remove', 'keep_first', 'keep_last'
        """
        cleaned_data = data.copy()
        
        if method == 'remove':
            cleaned_data = cleaned_data.drop_duplicates()
        elif method == 'keep_first':
            cleaned_data = cleaned_data.drop_duplicates(keep='first')
        elif method == 'keep_last':
            cleaned_data = cleaned_data.drop_duplicates(keep='last')
        
        return cleaned_data
    
    def handle_scale_normalize(self, data, method='standard', columns=None):
        """
        Handle feature scaling and normalization
        methods: 'standard', 'minmax', 'robust'
        """
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
        
        cleaned_data = data.copy()
        
        if columns is None:
            columns = self.numeric_columns
        
        numeric_cols = [col for col in columns if col in cleaned_data.columns and col in self.numeric_columns]
        
        if not numeric_cols:
            return cleaned_data
            
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        
        # Ensure we're working with DataFrame
        cleaned_data[numeric_cols] = pd.DataFrame(
            scaler.fit_transform(cleaned_data[numeric_cols]),
            columns=numeric_cols,
            index=cleaned_data.index
        )
        
        return cleaned_data
    
    def handle_inconsistent_data(self, data, columns=None):
        """
        Handle inconsistent data formats and values
        """
        cleaned_data = data.copy()
        
        if columns is None:
            columns = self.categorical_columns + self.numeric_columns
        
        for column in columns:
            if column in cleaned_data.columns:
                # Remove leading/trailing whitespaces for string columns
                if cleaned_data[column].dtype == 'object':
                    cleaned_data[column] = cleaned_data[column].astype(str).str.strip()
                    
                    # Standardize common categorical values
                    if cleaned_data[column].str.upper().isin(['YES', 'NO']).any():
                        cleaned_data[column] = cleaned_data[column].str.upper()
                    if cleaned_data[column].str.upper().isin(['TRUE', 'FALSE']).any():
                        cleaned_data[column] = cleaned_data[column].str.upper()
        
        return cleaned_data
    
    def create_dummy_variables(self, data, columns=None, drop_original=True):
        """
        Create dummy variables for categorical columns (specialized one-hot encoding)
        """
        cleaned_data = data.copy()
        
        if columns is None:
            columns = self.categorical_columns
        
        dummy_columns = [col for col in columns if col in cleaned_data.columns]
        
        for column in dummy_columns:
            dummies = pd.get_dummies(cleaned_data[column], prefix=column)
            cleaned_data = pd.concat([cleaned_data, dummies], axis=1)
        
        if drop_original:
            cleaned_data = cleaned_data.drop(dummy_columns, axis=1)
        
        return cleaned_data
    
    def create_comparison_plots(self, original_data, cleaned_data, method_name):
        """
        Create comparison plots for before and after noise handling
        """
        numeric_columns_original = original_data.select_dtypes(include=[np.number]).columns
        numeric_columns_cleaned = cleaned_data.select_dtypes(include=[np.number]).columns
        
        # Find common numeric columns
        common_numeric = list(set(numeric_columns_original) & set(numeric_columns_cleaned))
        
        if not common_numeric:
            print(f"No common numeric columns for comparison plots in {method_name}")
            return None
        
        # Determine layout
        n_plots = len(common_numeric)
        n_cols = 2
        n_rows = (n_plots + 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_plots > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
        
        for i, column in enumerate(common_numeric):
            if i < len(axes):
                # Create boxplot comparison
                data_to_plot = [original_data[column].dropna(), cleaned_data[column].dropna()]
                axes[i].boxplot(data_to_plot, labels=['Before', 'After'])
                axes[i].set_title(f'{column}\n({method_name})')
                axes[i].set_ylabel('Values')
                axes[i].grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(len(common_numeric), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def generate_statistics_report(self, original_data, cleaned_data, method_name):
        """
        Generate statistical comparison report
        """
        numeric_columns_original = original_data.select_dtypes(include=[np.number]).columns
        numeric_columns_cleaned = cleaned_data.select_dtypes(include=[np.number]).columns
        
        # Find common numeric columns
        common_numeric = list(set(numeric_columns_original) & set(numeric_columns_cleaned))
        
        stats_data = []
        for column in common_numeric:
            orig_stats = original_data[column].describe()
            clean_stats = cleaned_data[column].describe()
            
            stats_data.append({
                'Column': column,
                'Method': method_name,
                'Original_Count': orig_stats['count'],
                'Cleaned_Count': clean_stats['count'],
                'Original_Mean': orig_stats['mean'],
                'Cleaned_Mean': clean_stats['mean'],
                'Original_Std': orig_stats['std'],
                'Cleaned_Std': clean_stats['std'],
                'Removed_Rows': len(original_data) - len(cleaned_data)
            })
        
        return pd.DataFrame(stats_data)
    
    def get_processing_selection(self):
        """
        Interactive method to let user select which data processing methods to use
        """
        processing_options = {
            '1': {'name': 'Outlier_Handling', 'description': 'Handle outliers using various methods'},
            '2': {'name': 'Missing_Values', 'description': 'Handle missing values using imputation or removal'},
            '3': {'name': 'Categorical_Encoding', 'description': 'Encode categorical variables'},
            '4': {'name': 'Dummy_Variables', 'description': 'Create dummy variables for categorical columns'},
            '5': {'name': 'Duplicate_Removal', 'description': 'Remove duplicate rows'},
            '6': {'name': 'Feature_Scaling', 'description': 'Scale and normalize features'},
            '7': {'name': 'Inconsistent_Data', 'description': 'Fix inconsistent data formats'}
        }
        
        outlier_methods = {
            '1': {'name': 'IQR_Capping', 'description': 'Caps outliers using Interquartile Range method'},
            '2': {'name': 'ZScore_Removal', 'description': 'Removes outliers using Z-score method'},
            '3': {'name': 'ZScore_Capping', 'description': 'Caps outliers using Z-score method'},
            '4': {'name': 'MAD_Removal', 'description': 'Removes outliers using Median Absolute Deviation'},
            '5': {'name': 'Isolation_Forest', 'description': 'Uses Isolation Forest (ML) to remove outliers'}
        }
        
        missing_methods = {
            '1': {'name': 'Remove', 'description': 'Remove rows with missing values'},
            '2': {'name': 'Mean_Imputation', 'description': 'Impute missing values with mean'},
            '3': {'name': 'Median_Imputation', 'description': 'Impute missing values with median'},
            '4': {'name': 'Mode_Imputation', 'description': 'Impute missing values with mode'},
            '5': {'name': 'KNN_Imputation', 'description': 'Impute using K-Nearest Neighbors'}
        }
        
        encoding_methods = {
            '1': {'name': 'OneHot_Encoding', 'description': 'One-hot encoding for categorical variables'},
            '2': {'name': 'Label_Encoding', 'description': 'Label encoding for categorical variables'},
            '3': {'name': 'Frequency_Encoding', 'description': 'Frequency encoding for categorical variables'}
        }
        
        scaling_methods = {
            '1': {'name': 'Standard_Scaling', 'description': 'Standard scaling (mean=0, std=1)'},
            '2': {'name': 'MinMax_Scaling', 'description': 'MinMax scaling (range 0-1)'},
            '3': {'name': 'Robust_Scaling', 'description': 'Robust scaling (robust to outliers)'}
        }
        
        print("\n" + "="*70)
        print("DATA PROCESSING AND NOISE HANDLING OPTIONS")
        print("="*70)
        for key, option in processing_options.items():
            print(f"{key}. {option['name']}: {option['description']}")
        
        selected_methods = {}
        
        # Outlier handling selection
        outlier_choice = input("\nApply outlier handling? (y/n): ").strip().lower()
        if outlier_choice == 'y':
            print("\nOutlier Handling Methods:")
            for key, method in outlier_methods.items():
                print(f"  {key}. {method['name']}: {method['description']}")
            outlier_selection = input("Select outlier methods (comma-separated or 'all'): ").strip()
            if outlier_selection.lower() == 'all':
                selected_methods['Outlier_Handling'] = {method['name']: method for method in outlier_methods.values()}
            else:
                selected_keys = [s.strip() for s in outlier_selection.split(',')]
                selected_methods['Outlier_Handling'] = {}
                for key in selected_keys:
                    if key in outlier_methods:
                        selected_methods['Outlier_Handling'][outlier_methods[key]['name']] = outlier_methods[key]
        
        # Missing values handling
        missing_choice = input("\nApply missing values handling? (y/n): ").strip().lower()
        if missing_choice == 'y':
            print("\nMissing Values Methods:")
            for key, method in missing_methods.items():
                print(f"  {key}. {method['name']}: {method['description']}")
            missing_selection = input("Select missing values method: ").strip()
            if missing_selection in missing_methods:
                selected_methods['Missing_Values'] = {missing_methods[missing_selection]['name']: missing_methods[missing_selection]}
        
        # Categorical encoding
        encoding_choice = input("\nApply categorical encoding? (y/n): ").strip().lower()
        if encoding_choice == 'y':
            print("\nCategorical Encoding Methods:")
            for key, method in encoding_methods.items():
                print(f"  {key}. {method['name']}: {method['description']}")
            encoding_selection = input("Select encoding method: ").strip()
            if encoding_selection in encoding_methods:
                selected_methods['Categorical_Encoding'] = {encoding_methods[encoding_selection]['name']: encoding_methods[encoding_selection]}
        
        # Dummy variables
        dummy_choice = input("\nCreate dummy variables? (y/n): ").strip().lower()
        if dummy_choice == 'y':
            selected_methods['Dummy_Variables'] = {'Dummy_Variables': {'description': 'Create dummy variables for categorical columns'}}
        
        # Duplicate removal
        duplicate_choice = input("\nRemove duplicate rows? (y/n): ").strip().lower()
        if duplicate_choice == 'y':
            selected_methods['Duplicate_Removal'] = {'Remove_Duplicates': {'description': 'Remove duplicate rows'}}
        
        # Feature scaling
        scaling_choice = input("\nApply feature scaling? (y/n): ").strip().lower()
        if scaling_choice == 'y':
            print("\nFeature Scaling Methods:")
            for key, method in scaling_methods.items():
                print(f"  {key}. {method['name']}: {method['description']}")
            scaling_selection = input("Select scaling method: ").strip()
            if scaling_selection in scaling_methods:
                selected_methods['Feature_Scaling'] = {scaling_methods[scaling_selection]['name']: scaling_methods[scaling_selection]}
        
        # Inconsistent data handling
        inconsistent_choice = input("\nFix inconsistent data formats? (y/n): ").strip().lower()
        if inconsistent_choice == 'y':
            selected_methods['Inconsistent_Data'] = {'Inconsistent_Data': {'description': 'Fix inconsistent data formats and values'}}
        
        print(f"\nSelected processing steps: {list(selected_methods.keys())}")
        return selected_methods
    
    def process_data(self, output_dir='output'):
        """
        Main method to process data using selected techniques
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Let user select processing methods
        selected_methods = self.get_processing_selection()
        
        if not selected_methods:
            print("No methods selected. Exiting.")
            return
        
        # Define methods functions
        outlier_functions = {
            'IQR_Capping': lambda data: self.handle_outliers_iqr(data),
            'ZScore_Removal': lambda data: self.handle_outliers_zscore(data, method='remove'),
            'ZScore_Capping': lambda data: self.handle_outliers_zscore(data, method='cap'),
            'MAD_Removal': lambda data: self.handle_outliers_mad(data, method='remove'),
            'Isolation_Forest': lambda data: self.handle_outliers_isolation_forest(data)
        }
        
        missing_functions = {
            'Remove': lambda data: self.handle_missing_values(data, method='remove'),
            'Mean_Imputation': lambda data: self.handle_missing_values(data, method='mean'),
            'Median_Imputation': lambda data: self.handle_missing_values(data, method='median'),
            'Mode_Imputation': lambda data: self.handle_missing_values(data, method='mode'),
            'KNN_Imputation': lambda data: self.handle_missing_values(data, method='knn')
        }
        
        encoding_functions = {
            'OneHot_Encoding': lambda data: self.handle_categorical_encoding(data, method='onehot'),
            'Label_Encoding': lambda data: self.handle_categorical_encoding(data, method='label'),
            'Frequency_Encoding': lambda data: self.handle_categorical_encoding(data, method='frequency')
        }
        
        scaling_functions = {
            'Standard_Scaling': lambda data: self.handle_scale_normalize(data, method='standard'),
            'MinMax_Scaling': lambda data: self.handle_scale_normalize(data, method='minmax'),
            'Robust_Scaling': lambda data: self.handle_scale_normalize(data, method='robust')
        }
        
        all_stats = []
        
        # Create PDF for comparison plots
        pdf_path = os.path.join(output_dir, f'{self.file_name}_advanced_processing_report.pdf')
        
        with PdfPages(pdf_path) as pdf:
            # First page: Summary statistics
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.axis('tight')
            ax.axis('off')
            
            # Create initial summary
            summary_text = f"""
            Advanced Data Processing Report
            File: {self.file_name}.csv
            Original Data Shape: {self.original_data.shape}
            Numeric Columns: {self.numeric_columns}
            Categorical Columns: {self.categorical_columns}
            Selected Processing Steps: {list(selected_methods.keys())}
            Processing Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
            
            Processing Steps Applied:
            {chr(10).join(['- ' + step for step in selected_methods.keys()])}
            
            Output Files:
            - CSV files for each processing combination
            - PDF processing report
            """
            
            ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=12, 
                   verticalalignment='top', fontfamily='monospace')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Apply processing steps sequentially
            current_data = self.original_data.copy()
            processing_steps = []
            
            for step_name, methods in selected_methods.items():
                for method_name, method_info in methods.items():
                    print(f"\nApplying {step_name} - {method_name}...")
                    
                    try:
                        # Apply the appropriate processing method
                        if step_name == 'Outlier_Handling' and method_name in outlier_functions:
                            current_data = outlier_functions[method_name](current_data)
                        elif step_name == 'Missing_Values' and method_name in missing_functions:
                            current_data = missing_functions[method_name](current_data)
                        elif step_name == 'Categorical_Encoding' and method_name in encoding_functions:
                            current_data = encoding_functions[method_name](current_data)
                        elif step_name == 'Dummy_Variables':
                            current_data = self.create_dummy_variables(current_data)
                        elif step_name == 'Duplicate_Removal':
                            current_data = self.handle_duplicates(current_data)
                        elif step_name == 'Feature_Scaling' and method_name in scaling_functions:
                            current_data = scaling_functions[method_name](current_data)
                        elif step_name == 'Inconsistent_Data':
                            current_data = self.handle_inconsistent_data(current_data)
                        
                        processing_steps.append(f"{step_name}_{method_name}")
                        
                        # Save intermediate result
                        combo_name = "_".join(processing_steps)
                        csv_filename = os.path.join(output_dir, f'{self.file_name}_{combo_name}_processed.csv')
                        current_data.to_csv(csv_filename, index=False)
                        
                        # Create comparison plots if possible
                        fig = self.create_comparison_plots(self.original_data, current_data, combo_name)
                        if fig is not None:
                            pdf.savefig(fig, bbox_inches='tight')
                            plt.close()
                        
                        # Generate statistics
                        stats_df = self.generate_statistics_report(self.original_data, current_data, combo_name)
                        if not stats_df.empty:
                            all_stats.append(stats_df)
                        
                        print(f"  - Current data shape: {current_data.shape}")
                        print(f"  - Saved to: {csv_filename}")
                        
                    except Exception as e:
                        print(f"  - Error applying {step_name} - {method_name}: {str(e)}")
                        print(f"  - Continuing with next step...")
            
            # Final combined statistics
            if all_stats:
                combined_stats = pd.concat(all_stats, ignore_index=True)
                
                fig, ax = plt.subplots(figsize=(16, 10))
                ax.axis('tight')
                ax.axis('off')
                
                table = ax.table(cellText=combined_stats.round(4).values,
                               colLabels=combined_stats.columns,
                               cellLoc='center',
                               loc='center',
                               bbox=[0, 0, 1, 1])
                
                table.auto_set_font_size(False)
                table.set_fontsize(6)
                table.scale(1, 2)
                
                ax.set_title('Combined Statistics - All Processing Steps', fontsize=16, pad=20)
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
        
        print(f"\n" + "="*70)
        print(f"ADVANCED PROCESSING COMPLETE!")
        print(f"Output directory: {output_dir}")
        print(f"PDF report: {pdf_path}")
        print("="*70)
        
        return all_stats

def main():
    """
    Main function to run the advanced data noise handler on House Price data
    """
    # Initialize the advanced noise handler with your house price data
    handler = AdvancedDataNoiseHandler('House_PriceReg.csv')
    
    # Process data with selected methods
    statistics = handler.process_data()
    
    print("\nGenerated files:")
    output_files = os.listdir('output')
    for file in sorted(output_files):
        print(f"  - output/{file}")

if __name__ == "__main__":
    main()
