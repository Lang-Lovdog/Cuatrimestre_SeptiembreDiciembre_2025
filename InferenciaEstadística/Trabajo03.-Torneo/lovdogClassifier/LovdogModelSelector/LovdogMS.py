## Importation des bibliothèques
### Mathématiques
import pandas as pd                                  # type: ignore
import numpy as np                                   # type: ignore
import random
import os
### Visualisation
#import seaborn as sns                               # type: ignore
import matplotlib.pyplot as plt                      # type: ignore
### Machine learning
#### Preprocessing des données
from sklearn.preprocessing import LabelEncoder       # type: ignore
from sklearn.preprocessing import label_binarize     # type: ignore
#### Selection processus
from sklearn.pipeline import Pipeline                # type: ignore
from sklearn.preprocessing import StandardScaler     # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.model_selection import GridSearchCV     # type: ignore
from sklearn.model_selection import cross_val_score  # type: ignore
#### Algorithmes de classification
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.svm import SVC                          # type: ignore
from sklearn.neighbors import KNeighborsClassifier   # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore
#### Métriques
from sklearn.metrics import accuracy_score           # type: ignore
from sklearn.metrics import f1_score                 # type: ignore
from sklearn.metrics import confusion_matrix         # type: ignore
from sklearn.metrics import ConfusionMatrixDisplay   # type: ignore
from sklearn.metrics import precision_score          # type: ignore
from sklearn.metrics import recall_score             # type: ignore
from sklearn.metrics import classification_report    # type: ignore
from sklearn.metrics import roc_curve                # type: ignore
from sklearn.metrics import auc                      # type: ignore
from sklearn.metrics import RocCurveDisplay          # type: ignore
from sklearn.model_selection import learning_curve   # type: ignore
### Datatypes
from typing import List                              # type: ignore
from typing import Optional                          # type: ignore
from typing import Tuple                             # type: ignore
from typing import Dict                              # type: ignore
from typing import Union                             # type: ignore
from typing import Any                               # type: ignore
### Temps (pour random_state
import time                                          # type: ignore
## Model import/export
import joblib                                        # type: ignore
# DEAP for EDAs
#from deap import base                                # type: ignore
#from deap import creator                             # type: ignore
#from deap import tools                               # type: ignore

_ModelSelectionClassesNames_ = []
_ModelSelectionClassesObjects_ = []

class ModelSelection:

    def __init__(
        self, 
        name: str, 
        # Multiple data input options for flexibility
        csv_path: Optional[str] = None,
        features_dataframe: Optional[pd.DataFrame] = None,
        ml_data_tuple: Optional[Tuple] = None,
        # Dataset configuration
        target_column: str = "class",
        test_size: float = 0.2,
        random_state: Optional[int] = None,
        # Performance optimization
        best_only_mode: bool = False,
        # Output configuration
        plots_output_directory: str = "plots",
        # Classifier configuration - NEW APPROACH
        classifiers_dict: Optional[Dict[str, Any]] = None,
        grid_parameters_dict: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a ModelSelection instance for comprehensive classifier evaluation.
        """
        
        # Validate unique name
        if name in _ModelSelectionClassesNames_:
            raise ValueError(f"ModelSelection instance '{name}' already exists")
        _ModelSelectionClassesNames_.append(name)
        _ModelSelectionClassesObjects_.append(self)
        
        # Store initialization parameters
        self.name = name
        self.target_column = target_column
        self.test_size = test_size
        self.best_only_mode = best_only_mode
        self.plots_output_dir = plots_output_directory
        
        # Set random state if not provided
        self.random_state = random_state if random_state is not None else np.floor(time.time()).astype(int) % 4294967296
        
        # Initialize data structures
        self._initialize_data_structures()
        
        # Register classifiers - NEW APPROACH
        self._register_classifiers(classifiers_dict, grid_parameters_dict)
        
        # Load data from the provided source
        self._load_data_from_provided_source(
            csv_path, 
            features_dataframe, 
            ml_data_tuple, 
            target_column
        )
        
        # Create output directory
        os.makedirs(plots_output_directory, exist_ok=True)

    def _register_classifiers(self, classifiers_dict=None, grid_parameters_dict=None):
        """
        Register classifiers and their grid parameters.
        If no custom dictionaries provided, uses default values.
        """
        # Default classifiers
        default_classifiers = {
            'SVC': SVC(),
            'KNeighbors': KNeighborsClassifier(),
            'RandomForest': RandomForestClassifier(),
            'LogisticRegression': LogisticRegression(max_iter=850),
        }
        
        # Default grid parameters
        default_grid_parameters = {
            'SVC': {
                'clf__C': [0.001, 0.01, 0.1, 1, 10, 20],
                'clf__kernel': ['poly', 'sigmoid', 'rbf'],
                'clf__gamma': ['scale', 'auto'],
            },
            'KNeighbors': {
                'clf__n_neighbors': [12, 24, 48, 96],
                'clf__weights': ['uniform', 'distance'],
            },
            'RandomForest': {
                'clf__min_samples_split': [2, 4, 8, 16],
                'clf__n_estimators': [72, 144, 288, 576],
                'clf__max_depth': [3, 9, 27],
            },
            'LogisticRegression': {
                'clf__C': [0.001, 0.01, 0.1, 1, 10, 20],
                'clf__penalty': ['elasticnet', 'l2'],
                'clf__solver': ['saga'],
            },
        }
        
        # Use custom dictionaries if provided, otherwise use defaults
        self.classifiers = classifiers_dict if classifiers_dict is not None else default_classifiers
        self.gridParameters = grid_parameters_dict if grid_parameters_dict is not None else default_grid_parameters
        
        # Validate that all classifiers in grid parameters exist in classifiers dict
        self._validate_classifier_registration()

    def _validate_classifier_registration(self):
        """Ensure consistency between classifiers and grid parameters"""
        classifier_names = set(self.classifiers.keys())
        grid_param_names = set(self.gridParameters.keys())
        
        # Check for grid parameters without corresponding classifiers
        missing_classifiers = grid_param_names - classifier_names
        if missing_classifiers:
            raise ValueError(f"Grid parameters defined for non-existent classifiers: {missing_classifiers}")
        
        # Check for classifiers without grid parameters (warn but don't error)
        missing_grid_params = classifier_names - grid_param_names
        if missing_grid_params:
            print(f"Warning: No grid parameters defined for classifiers: {missing_grid_params}. "
                  f"They will use default parameters.")

    def _initialize_data_structures(self):
        """Initialize all data storage structures"""
        self.data = None
        self.target = None
        self.features_df = None
        self.training_data = {"x": None, "y": None}  # FIXED: consistent naming
        self.test_data = {"x": None, "y": None}      # FIXED: consistent naming
        
        # Results storage
        self.results = {}
        self.detailed_results = {}
        self.metrics_dataframe = pd.DataFrame()
        self.best_model_tag = None
        self.best_model = None
        self.plots_data = {}

    def _load_data_from_provided_source(self, csv_path, features_df, ml_data_tuple, target_column):
        """Load data from the provided source with priority handling"""
        data_loaded = False
        
        # Priority 1: ML data tuple (X, y) from LBP output
        if ml_data_tuple is not None:
            self._load_from_ml_data_tuple(ml_data_tuple)
            data_loaded = True
            print(f"Loaded data from ML tuple for {self.name}")
        
        # Priority 2: Features DataFrame
        if not data_loaded and features_df is not None:
            self.load_from_dataframe(features_df, target_column)
            data_loaded = True
            print(f"Loaded data from DataFrame for {self.name}")
        
        # Priority 3: CSV file path
        if not data_loaded and csv_path is not None:
            self.load_dataset(csv_path, target_column, self.test_size)
            data_loaded = True
            print(f"Loaded data from CSV for {self.name}")
        
        if not data_loaded:
            raise ValueError("No data source provided. Specify one of: csv_path, features_dataframe, or ml_data_tuple")

    def _load_from_ml_data_tuple(self, ml_data_tuple):
        """
        Load data from LBP ML output format (X_features, y_target)
        """
        if not isinstance(ml_data_tuple, tuple) or len(ml_data_tuple) != 2:
            raise ValueError("ml_data_tuple must be a tuple of format (X_features, y_target)")
        
        X_features, y_target = ml_data_tuple
        
        # Convert to pandas DataFrame/Series if they are numpy arrays
        if isinstance(X_features, np.ndarray):
            # Create DataFrame with generic column names
            X_features = pd.DataFrame(X_features, columns=[f'feature_{i}' for i in range(X_features.shape[1])])
        
        if isinstance(y_target, (np.ndarray, list)):
            y_target = pd.Series(y_target, name=self.target_column)
        
        # Validate types
        if not isinstance(X_features, pd.DataFrame):
            raise ValueError("X_features must be pandas DataFrame or numpy array")
        if not isinstance(y_target, pd.Series):
            raise ValueError("y_target must be pandas Series, numpy array, or list")
        
        # Store the combined dataframe for reference
        self.features_df = X_features.copy()
        self.features_df[self.target_column] = y_target.values
        
        # Store separated data
        self.data = X_features
        self.target = y_target
        
        # Split the data
        self._split_data()

    def load_from_dataframe(self, features_df, target_column="class"):
        """Load data from a pandas DataFrame"""
        if not isinstance(features_df, pd.DataFrame):
            raise ValueError("features_df must be a pandas DataFrame")
        
        self.data = features_df.drop(target_column, axis=1)
        self.target = features_df[target_column]
        self.features_df = features_df
        
        self._split_data()

    def load_dataset(self, dataset_path, target_column="class", test_size=0.2):
        """Load dataset from CSV file"""
        if not isinstance(dataset_path, str):
            raise ValueError("dataset_path must be a string")
        
        df = pd.read_csv(dataset_path)
        self.data = df.drop(target_column, axis=1)
        self.target = df[target_column]
        self.features_df = df
        
        self._split_data()

    def _split_data(self):
        """Split data into training and test sets"""
        (self.training_data['x'], self.test_data['x'], 
         self.training_data['y'], self.test_data['y']) = train_test_split(
            self.data, self.target,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=self.target
        )

    def selectModels(self):
        """Run grid search for all classifiers, including AUC calculation"""
        self.results = {}
        all_metrics = []
        
        for tag, algorithm in self.classifiers.items():
            print(f"Training {tag}...")
            
            # Prepare pipeline
            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', algorithm)
            ])
            
            # SetUp grid
            modelsGrid = GridSearchCV(
                pipe,
                param_grid=self.gridParameters[tag],
                scoring={
                    "accuracy": "accuracy",
                    "precision": "precision_weighted",
                    "recall": "recall_weighted", 
                    "f1": "f1_weighted"
                },
                refit="accuracy",
                cv=5,
                n_jobs=-1,
                verbose=1
            )

            # Run grid test
            modelsGrid.fit(self.training_data['x'], self.training_data['y'])  # FIXED: training_data
            
            # Get basic predictions and metrics
            best_model = modelsGrid.best_estimator_
            y_pred = best_model.predict(self.test_data['x'])  # FIXED: test_data
            
            # Calculate all basic metrics
            accuracy = accuracy_score(self.test_data['y'], y_pred)  # FIXED: test_data
            precision = precision_score(self.test_data['y'], y_pred, average='weighted')  # FIXED
            recall = recall_score(self.test_data['y'], y_pred, average='weighted')  # FIXED
            f1 = f1_score(self.test_data['y'], y_pred, average='weighted')  # FIXED
            
            # Calculate AUC if the model supports probability predictions
            auc_score = None
            if hasattr(best_model, 'predict_proba'):
                try:
                    y_pred_proba = best_model.predict_proba(self.test_data['x'])  # FIXED: test_data
                    
                    # For multi-class classification, use One-vs-Rest approach
                    if len(np.unique(self.test_data['y'])) > 2:  # FIXED: test_data
                        y_test_bin = label_binarize(self.test_data['y'],  # FIXED: test_data
                                                  classes=np.unique(self.test_data['y']))  # FIXED
                        # Calculate micro-average AUC
                        fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_pred_proba.ravel())
                        auc_score = auc(fpr, tpr)
                    else:
                        # For binary classification
                        fpr, tpr, _ = roc_curve(self.test_data['y'], y_pred_proba[:, 1])  # FIXED
                        auc_score = auc(fpr, tpr)
                except Exception as e:
                    print(f"Warning: Could not calculate AUC for {tag}: {e}")
                    auc_score = None
            
            # Store comprehensive results
            self.results[tag] = {
                "best_params": modelsGrid.best_params_,
                "best_estimator": best_model,
                "cv_best_score": modelsGrid.best_score_,
                "test_accuracy": accuracy,
                "test_precision": precision,
                "test_recall": recall,
                "test_f1": f1,
                "test_auc": auc_score,
                "confusion_matrix": confusion_matrix(self.test_data['y'], y_pred),  # FIXED: test_data
                "supports_proba": hasattr(best_model, 'predict_proba')
            }
            
            # Store metrics for CSV (including AUC)
            metrics_row = {
                'Classifier': tag,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1_Score': f1,
                'AUC_Score': auc_score if auc_score is not None else 'N/A',
                'CV_Score': modelsGrid.best_score_,
                'Best_Params': str(modelsGrid.best_params_),
                'Supports_Probability': hasattr(best_model, 'predict_proba')
            }
            all_metrics.append(metrics_row)
            
            print(f"  {tag} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, AUC: {auc_score if auc_score else 'N/A'}")

        # Create metrics dataframe
        self.metrics_df = pd.DataFrame(all_metrics)
        
        # Determine the best classifier (considering AUC if available)
        if not self.metrics_df.empty:
            # Prefer classifiers that have AUC scores for ranking
            has_auc_df = self.metrics_df[self.metrics_df['AUC_Score'] != 'N/A'].copy()
            if not has_auc_df.empty:
                # Convert AUC_Score to float for sorting
                has_auc_df['AUC_Score'] = has_auc_df['AUC_Score'].astype(float)
                best_idx = has_auc_df['AUC_Score'].idxmax()
                self.best_model_tag = has_auc_df.loc[best_idx, 'Classifier']
            else:
                # Fallback to accuracy if no AUC available
                best_idx = self.metrics_df['Accuracy'].idxmax()
                self.best_model_tag = self.metrics_df.loc[best_idx, 'Classifier']
            
            self.best_model = self.results[self.best_model_tag]['best_estimator']
            
            print(f"\nBest classifier: {self.best_model_tag}")
            best_result = self.results[self.best_model_tag]
            print(f"  Accuracy: {best_result['test_accuracy']:.4f}")
            print(f"  F1-Score: {best_result['test_f1']:.4f}")
            if best_result['test_auc'] is not None:
                print(f"  AUC: {best_result['test_auc']:.4f}")

    def getMetricsDataFrame(self):
        """Create a DataFrame with all metrics including AUC for CSV export"""
        if self.metrics_df.empty:
            # If selectModels hasn't been run, create basic metrics dataframe
            metrics_data = []
            for tag, result in self.results.items():
                metrics_data.append({
                    'Classifier': tag,
                    'Accuracy': result['test_accuracy'],
                    'Precision': result['test_precision'],
                    'Recall': result['test_recall'],
                    'F1_Score': result['test_f1'],
                    'AUC_Score': result['test_auc'] if result['test_auc'] is not None else 'N/A',
                    'CV_Score': result['cv_best_score'],
                    'Best_Params': str(result['best_params']),
                    'Supports_Probability': result['supports_proba']
                })
            
            self.metrics_df = pd.DataFrame(metrics_data)
        
        return self.metrics_df

    def _generateDetailedResultsForBest(self):
        """Generate detailed results (plots, proba, etc.) only for the best classifier"""
        if self.best_model_tag is None:
            raise ValueError("No best model identified. Run selectModels() first.")
            
        best_result = self.results[self.best_model_tag]
        best_model = best_result['best_estimator']
        
        # Get detailed predictions including probabilities if available
        y_pred = best_model.predict(self.test_data['x'])  # FIXED: test_data
        y_pred_proba = best_model.predict_proba(self.test_data['x']) if hasattr(best_model, 'predict_proba') else None  # FIXED
        
        # Store detailed results
        self.detailed_results = {
            "best_params": best_result['best_params'],
            "best_estimator": best_model,
            "cv_best_score": best_result['cv_best_score'],
            "test_accuracy": best_result['test_accuracy'],
            "test_precision": best_result['test_precision'],
            "test_recall": best_result['test_recall'],
            "test_f1": best_result['test_f1'],
            "confusion_matrix": best_result['confusion_matrix'],
            "y_true": self.test_data['y'],  # FIXED: test_data
            "y_pred": y_pred,
            "y_pred_proba": y_pred_proba,
            "classification_report": classification_report(self.test_data['y'], y_pred, output_dict=True)  # FIXED
        }

    def generatePlots(self):
        """Generate plots - only for best classifier if best_only=True, otherwise for all"""
        if self.best_only_mode:
            self._generatePlotsForBest()
        else:
            self._generatePlotsForAll()

    def _generatePlotsForBest(self):
        """Generate all plots only for the best classifier"""
        if not self.detailed_results:
            self._generateDetailedResultsForBest()
            
        tag = self.best_model_tag
        print(f"Generating plots for best classifier: {tag}")
        
        # Learning Curve
        self._generateLearningCurve(tag)
        
        # ROC Curve (if supported)
        if hasattr(self.detailed_results['best_estimator'], 'predict_proba'):
            self._generateROCCurve(tag)
        
        # Confusion Matrix
        self._generateConfusionMatrix(tag)

    def _generatePlotsForAll(self):
        """Generate plots for all classifiers (original behavior)"""
        # Generate plots for each classifier
        for tag in self.classifiers.keys():
            self._generateLearningCurve(tag)
            if self.results[tag]['supports_proba']:
                self._generateROCCurve(tag)
            self._generateConfusionMatrix(tag)

    def _generateLearningCurve(self, tag):
        """Generate learning curve for a specific classifier"""
        model = self.results[tag]['best_estimator'] if not self.best_only_mode else self.detailed_results['best_estimator']
        
        train_sizes, train_scores, test_scores = learning_curve(
            model, self.data, self.target, cv=5, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 5),
            scoring='accuracy'
        )
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Training score', linewidth=2)
        plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', label='Cross-validation score', linewidth=2)
        plt.title(f'Learning Curve for {tag}', fontsize=14)
        plt.xlabel('Training examples', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{self.plots_output_dir}/learning_curve_{tag}.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _generateROCCurve(self, tag):
        """Generate ROC curve for a specific classifier"""
        model = self.results[tag]['best_estimator'] if not self.best_only_mode else self.detailed_results['best_estimator']
        
        if hasattr(model, 'predict_proba'):
            y_test_bin = label_binarize(self.test_data['y'], classes=np.unique(self.test_data['y']))  # FIXED
            y_score = model.predict_proba(self.test_data['x'])  # FIXED
            n_classes = y_test_bin.shape[1]
            
            fpr, tpr, roc_auc = {}, {}, {}
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            
            plt.figure(figsize=(10, 8))
            colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
            
            for i, color in zip(range(n_classes), colors):
                plt.plot(fpr[i], tpr[i], color=color, lw=2,
                        label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
            
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            plt.title(f'ROC Curve for {tag}', fontsize=14)
            plt.legend(loc="lower right", fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.savefig(f"{self.plots_output_dir}/roc_curve_{tag}.png", dpi=300, bbox_inches='tight')
            plt.close()

    def _generateConfusionMatrix(self, tag):
        """Generate confusion matrix for a specific classifier"""
        cm = self.results[tag]['confusion_matrix'] if not self.best_only_mode else self.detailed_results['confusion_matrix']
        
        plt.figure(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(cm)
        disp.plot(cmap=plt.cm.Blues, values_format='d')
        plt.title(f'Confusion Matrix for {tag}', fontsize=14)
        plt.savefig(f"{self.plots_output_dir}/confusion_matrix_{tag}.png", dpi=300, bbox_inches='tight')
        plt.close()

    # Additional helper methods for legacy compatibility
    def saveMetricsToCSV(self, filename="classification_metrics.csv"):
        """Save metrics to CSV file"""
        if self.metrics_df.empty:
            self.getMetricsDataFrame()
        self.metrics_df.to_csv(filename, index=False)

    def getAUCsDataFrame(self):
        """Create a DataFrame with AUC scores for ROC curves"""
        auc_data = []
        for tag, result in self.results.items():
            if result['test_auc'] is not None:
                auc_data.append({
                    'Classifier': tag,
                    'AUC_Score': result['test_auc']
                })
        return pd.DataFrame(auc_data)

    # Legacy methods (keep for backward compatibility)
    def setGridParameters(self, gridParameters):
        self.gridParameters = gridParameters

    def saveModels(self, path):
        """Save all trained models to disk"""
        os.makedirs(path, exist_ok=True)
        for tag, result in self.results.items():
            joblib.dump(result['best_estimator'], f"{path}/{tag}.joblib")
