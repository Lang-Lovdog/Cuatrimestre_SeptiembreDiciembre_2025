# house_price_analysis_complete.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.inspection import PartialDependenceDisplay
import scipy.stats as stats
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings('ignore')

class HousePriceAnalysis:
    def __init__(self, file_path):
        """Initialize the analysis with the dataset"""
        self.df = pd.read_csv(file_path)
        self.numerical_cols = ['price', 'crime_rate', 'resid_area', 'air_qual', 'room_num', 'age', 
                              'dist1', 'dist2', 'dist3', 'dist4', 'teachers', 'poor_prop', 
                              'n_hos_beds', 'n_hot_rooms', 'rainfall', 'parks']
        self.categorical_cols = ['airport', 'waterbody', 'bus_ter']
        self.df_imputed = None
        self.outlier_flags = None
        self.best_model = None
        self.feature_importance = None
        
    def detect_outliers_iqr(self, data, column):
        """Detect outliers using IQR method for summary statistics"""
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
        return outliers, lower_bound, upper_bound
    
    def get_outliers_summary(self):
        """Generate comprehensive outliers summary"""
        outliers_summary = {}
        for col in self.numerical_cols:
            outliers, lower, upper = self.detect_outliers_iqr(self.df, col)
            outliers_summary[col] = {
                'outlier_count': len(outliers),
                'outlier_percentage': (len(outliers) / len(self.df)) * 100,
                'lower_bound': lower,
                'upper_bound': upper
            }
        return pd.DataFrame(outliers_summary).T
    
    def multiple_imputation_outliers(self, outlier_threshold=3.5):
        """
        Implement multiple imputation for handling outliers as described in the paper
        Outliers are treated as missing data and imputed using Random Forest
        """
        data_clean = self.df.copy()
        self.outlier_flags = pd.DataFrame(index=self.df.index)
        
        for col in self.numerical_cols:
            if col == 'price':  # Don't treat price as outlier for imputation
                continue
                
            # Calculate modified Z-score (more robust to outliers)
            median_val = self.df[col].median()
            mad = stats.median_abs_deviation(self.df[col].dropna(), scale='normal')
            
            if mad > 0:  # Avoid division by zero
                modified_z_scores = 0.6745 * (self.df[col] - median_val) / mad
                # Mark outliers (absolute modified Z-score > threshold)
                is_outlier = np.abs(modified_z_scores) > outlier_threshold
                self.outlier_flags[col] = is_outlier
                
                # Set outliers to NaN for imputation
                data_clean.loc[is_outlier, col] = np.nan
        
        print(f"Total outliers detected: {self.outlier_flags.sum().sum()}")
        print(f"Outlier percentage: {(self.outlier_flags.sum().sum() / (len(self.df) * len(self.numerical_cols))) * 100:.2f}%")
        
        # Multiple Imputation using Random Forest
        imputer = IterativeImputer(
            estimator=RandomForestRegressor(n_estimators=100, random_state=42),
            max_iter=10,
            random_state=42
        )
        
        # Impute missing values (including outliers set to NaN)
        imputed_data = imputer.fit_transform(data_clean[self.numerical_cols])
        self.df_imputed = data_clean.copy()
        self.df_imputed[self.numerical_cols] = imputed_data
        
        return self.df_imputed, self.outlier_flags

    def create_regression_plots(self, output_file='regression_analysis.pdf'):
        """Create comprehensive regression plots showing price behavior"""
        # Select features for regression
        feature_cols = [col for col in self.numerical_cols if col != 'price']
        
        with PdfPages(output_file) as pdf:
            # 1. Overall regression performance
            self._create_overall_regression_plot(pdf, feature_cols)
            
            # 2. Individual feature regression plots
            self._create_individual_regression_plots(pdf, feature_cols)
            
            # 3. Residual analysis
            self._create_residual_analysis_plots(pdf, feature_cols)
            
            # 4. Partial dependence plots
            self._create_partial_dependence_plots(pdf, feature_cols)
            
            # 5. Model comparison
            self._create_model_comparison_plots(pdf, feature_cols)

    def _create_overall_regression_plot(self, pdf, feature_cols):
        """Create overall regression performance plot"""
        # Prepare data
        X = self.df_imputed[feature_cols]
        y = self.df_imputed['price']
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model
        model = LinearRegression()
        model.fit(X_scaled, y)
        y_pred = model.predict(X_scaled)
        
        # Create figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Overall Regression Analysis: Price Behavior', fontsize=16, fontweight='bold')
        
        # Plot 1: Predicted vs Actual
        ax1.scatter(y, y_pred, alpha=0.6, color='steelblue')
        ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
        ax1.set_xlabel('Actual Price')
        ax1.set_ylabel('Predicted Price')
        ax1.set_title(f'Predicted vs Actual (R² = {r2_score(y, y_pred):.3f})')
        ax1.grid(True, alpha=0.3)
        
        # Add regression line to scatter
        z = np.polyfit(y, y_pred, 1)
        p = np.poly1d(z)
        ax1.plot(y, p(y), "g--", alpha=0.8, label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
        ax1.legend()
        
        # Plot 2: Residuals vs Predicted
        residuals = y - y_pred
        ax2.scatter(y_pred, residuals, alpha=0.6, color='coral')
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.8)
        ax2.set_xlabel('Predicted Price')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Residuals vs Predicted')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Distribution of residuals
        ax3.hist(residuals, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        ax3.axvline(residuals.mean(), color='red', linestyle='--', label=f'Mean: {residuals.mean():.2f}')
        ax3.set_xlabel('Residuals')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Residuals')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Q-Q plot for normality
        stats.probplot(residuals, dist="norm", plot=ax4)
        ax4.set_title('Q-Q Plot: Normality Check of Residuals')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

    def _create_individual_regression_plots(self, pdf, feature_cols):
        """Create individual regression plots for top features"""
        # Get top 6 features by correlation with price
        correlations = []
        for col in feature_cols:
            corr = self.df_imputed[[col, 'price']].corr().iloc[0,1]
            correlations.append((col, abs(corr)))
        
        correlations.sort(key=lambda x: x[1], reverse=True)
        top_features = [corr[0] for corr in correlations[:6]]
        
        # Create 2x3 grid of plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Individual Feature Regression: Relationship with Price', fontsize=16, fontweight='bold')
        
        for idx, feature in enumerate(top_features):
            ax = axes[idx//3, idx%3]
            
            # Create scatter plot with regression line
            x = self.df_imputed[feature]
            y = self.df_imputed['price']
            
            # Calculate regression line
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            line = slope * x + intercept
            
            # Plot
            ax.scatter(x, y, alpha=0.6, color='steelblue', s=50)
            ax.plot(x, line, 'r-', linewidth=2, label=f'y = {slope:.2f}x + {intercept:.2f}')
            
            # Formatting
            ax.set_xlabel(feature)
            ax.set_ylabel('Price')
            ax.set_title(f'{feature} vs Price\n(R = {r_value:.3f}, p = {p_value:.3e})')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add confidence interval
            # confidence = 0.95
            # n = len(x)
            # dof = n - 2
            # t = stats.t.ppf(confidence, dof)
            # margin = t * std_err * np.sqrt(1/n + (x - x.mean())**2 / np.sum((x - x.mean())**2))
            # ax.fill_between(x, line - margin, line + margin, color='red', alpha=0.1)
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

    def _create_residual_analysis_plots(self, pdf, feature_cols):
        """Create detailed residual analysis plots"""
        X = self.df_imputed[feature_cols]
        y = self.df_imputed['price']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = LinearRegression()
        model.fit(X_scaled, y)
        y_pred = model.predict(X_scaled)
        residuals = y - y_pred
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Comprehensive Residual Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Residuals vs Fitted
        ax1.scatter(y_pred, residuals, alpha=0.6, color='purple')
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.8)
        ax1.set_xlabel('Fitted Values')
        ax1.set_ylabel('Residuals')
        ax1.set_title('Residuals vs Fitted Values')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Scale-Location plot
        standardized_residuals = residuals / np.std(residuals)
        ax2.scatter(y_pred, np.sqrt(np.abs(standardized_residuals)), alpha=0.6, color='orange')
        ax2.set_xlabel('Fitted Values')
        ax2.set_ylabel('√|Standardized Residuals|')
        ax2.set_title('Scale-Location Plot')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Residuals distribution with normal curve
        n, bins, patches = ax3.hist(residuals, bins=30, density=True, alpha=0.7, 
                                  color='lightblue', edgecolor='black')
        
        # Add normal distribution curve
        mu, sigma = residuals.mean(), residuals.std()
        x = np.linspace(residuals.min(), residuals.max(), 100)
        y_norm = stats.norm.pdf(x, mu, sigma)
        ax3.plot(x, y_norm, 'r-', linewidth=2, label=f'N(μ={mu:.2f}, σ={sigma:.2f})')
        ax3.set_xlabel('Residuals')
        ax3.set_ylabel('Density')
        ax3.set_title('Residual Distribution vs Normal')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Residuals over time/index (if we had time series)
        ax4.plot(range(len(residuals)), residuals, 'o-', alpha=0.6, color='green')
        ax4.axhline(y=0, color='red', linestyle='--', alpha=0.8)
        ax4.set_xlabel('Observation Index')
        ax4.set_ylabel('Residuals')
        ax4.set_title('Residuals Order Plot')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

    def _create_partial_dependence_plots(self, pdf, feature_cols):
        """Create partial dependence plots to show marginal effects"""
        X = self.df_imputed[feature_cols]
        y = self.df_imputed['price']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)
        
        # Get top 4 features by importance
        importances = model.feature_importances_
        feature_importance = list(zip(feature_cols, importances))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        top_features = [fi[0] for fi in feature_importance[:4]]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Partial Dependence Plots: Marginal Effect on Price', fontsize=16, fontweight='bold')
        
        for idx, feature in enumerate(top_features):
            ax = axes[idx//2, idx%2]
            
            # Create partial dependence manually
            feature_idx = feature_cols.index(feature)
            unique_vals = np.unique(X_scaled[:, feature_idx])
            pd_values = []
            
            for val in unique_vals:
                X_temp = X_scaled.copy()
                X_temp[:, feature_idx] = val
                predictions = model.predict(X_temp)
                pd_values.append(np.mean(predictions))
            
            ax.plot(unique_vals, pd_values, 'b-', linewidth=2, marker='o')
            ax.set_xlabel(f'{feature} (Standardized)')
            ax.set_ylabel('Partial Dependence')
            ax.set_title(f'Partial Dependence: {feature}')
            ax.grid(True, alpha=0.3)
            
            # Add actual data points for reference
            ax.scatter(X_scaled[:, feature_idx], y, alpha=0.1, color='gray', s=20)
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

    def _create_model_comparison_plots(self, pdf, feature_cols):
        """Compare different regression models"""
        X = self.df_imputed[feature_cols]
        y = self.df_imputed['price']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Define models to compare
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        
        # Perform cross-validation
        cv_scores = {}
        for name, model in models.items():
            scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
            cv_scores[name] = scores.mean()
        
        # Train models and get predictions
        predictions = {}
        for name, model in models.items():
            model.fit(X_scaled, y)
            predictions[name] = model.predict(X_scaled)
        
        # Create comparison plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Regression Model Comparison', fontsize=16, fontweight='bold')
        
        # Plot 1: Cross-validation scores
        names = list(cv_scores.keys())
        scores = list(cv_scores.values())
        bars = ax1.bar(names, scores, color=['steelblue', 'lightcoral', 'lightgreen', 'orange'])
        ax1.set_ylabel('R² Score (Cross-Validation)')
        ax1.set_title('Model Performance Comparison')
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        
        # Add values on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Prediction comparison for best model
        best_model_name = max(cv_scores, key=cv_scores.get)
        best_predictions = predictions[best_model_name]
        
        ax2.scatter(y, best_predictions, alpha=0.6, color='steelblue')
        ax2.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
        ax2.set_xlabel('Actual Price')
        ax2.set_ylabel('Predicted Price')
        ax2.set_title(f'Best Model: {best_model_name}\n(R² = {r2_score(y, best_predictions):.3f})')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Feature importance for tree-based models
        if hasattr(models['Random Forest'], 'feature_importances_'):
            importances = models['Random Forest'].feature_importances_
            indices = np.argsort(importances)[::-1][:8]  # Top 8 features
            
            ax3.bar(range(len(indices)), importances[indices], color='lightgreen')
            ax3.set_xlabel('Feature Index')
            ax3.set_ylabel('Importance')
            ax3.set_title('Random Forest Feature Importance')
            ax3.set_xticks(range(len(indices)))
            ax3.set_xticklabels([feature_cols[i] for i in indices], rotation=45)
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Residual distribution comparison
        residuals_comparison = {}
        for name, pred in predictions.items():
            residuals_comparison[name] = y - pred
        
        ax4.boxplot([residuals_comparison[name] for name in names], labels=names)
        ax4.axhline(y=0, color='red', linestyle='--', alpha=0.8)
        ax4.set_ylabel('Residuals')
        ax4.set_title('Residual Distribution Comparison')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
        
        # Store best model
        self.best_model = models[best_model_name]
        self.feature_importance = list(zip(feature_cols, models['Random Forest'].feature_importances_))
        self.feature_importance.sort(key=lambda x: x[1], reverse=True)

    def create_cox_style_analysis(self, output_file='cox_style_analysis.pdf'):
        """
        Create Cox-regression style analysis adapted for house prices
        This adapts the proportional hazards concept to price distribution analysis
        """
        with PdfPages(output_file) as pdf:
            # 1. Price distribution and survival-style analysis
            self._create_price_survival_analysis(pdf)
            
            # 2. Hazard-ratio style analysis for features
            self._create_hazard_ratio_analysis(pdf)
            
            # 3. Stratified analysis by key features
            self._create_stratified_analysis(pdf)

    def _create_price_survival_analysis(self, pdf):
        """Create survival-style analysis for price distribution"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Cox-Style Analysis: Price Distribution and Risk Factors', fontsize=16, fontweight='bold')
        
        # Plot 1: Price distribution (similar to survival function)
        prices_sorted = np.sort(self.df_imputed['price'])
        n = len(prices_sorted)
        survival_prob = 1 - np.arange(n) / n
        
        ax1.plot(prices_sorted, survival_prob, 'b-', linewidth=2)
        ax1.set_xlabel('Price')
        ax1.set_ylabel('Proportion Above Price')
        ax1.set_title('Price Distribution Function\n(Similar to Survival Function)')
        ax1.grid(True, alpha=0.3)
        
        # Add quartiles
        for q in [0.25, 0.5, 0.75]:
            price_q = np.percentile(prices_sorted, q * 100)
            ax1.axvline(x=price_q, color='red', linestyle='--', alpha=0.7,
                       label=f'{int(q*100)}th percentile: ${price_q:.2f}')
        ax1.legend()
        
        # Plot 2: Hazard function (price density / survival)
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(prices_sorted)
        x_range = np.linspace(prices_sorted.min(), prices_sorted.max(), 200)
        density = kde(x_range)
        
        # Approximate hazard function
        survival = 1 - np.array([np.mean(prices_sorted <= x) for x in x_range])
        hazard = density / np.maximum(survival, 1e-8)  # Avoid division by zero
        
        ax2.plot(x_range, hazard, 'r-', linewidth=2)
        ax2.set_xlabel('Price')
        ax2.set_ylabel('Hazard Rate')
        ax2.set_title('Approximate Hazard Function\n(Probability density / Survival)')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Log-log plot for proportional hazards check
        log_price = np.log(prices_sorted[prices_sorted > 0])
        log_survival = np.log(survival_prob[prices_sorted > 0])
        
        ax3.plot(log_price, log_survival, 'g-', linewidth=2)
        ax3.set_xlabel('Log(Price)')
        ax3.set_ylabel('Log(Survival Probability)')
        ax3.set_title('Log-Log Plot: Proportional Hazards Check')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Cumulative hazard
        cumulative_hazard = -np.log(survival_prob)
        ax4.plot(prices_sorted, cumulative_hazard, 'purple', linewidth=2)
        ax4.set_xlabel('Price')
        ax4.set_ylabel('Cumulative Hazard')
        ax4.set_title('Cumulative Hazard Function')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

    def _create_hazard_ratio_analysis(self, pdf):
        """Create hazard-ratio style analysis for feature effects"""
        # Select top features for analysis
        top_features = [feat for feat, imp in self.feature_importance[:6]]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Feature Effects: Hazard-Ratio Style Analysis', fontsize=16, fontweight='bold')
        
        for idx, feature in enumerate(top_features):
            ax = axes[idx//3, idx%3]
            
            # Create groups based on feature median
            median_val = self.df_imputed[feature].median()
            high_group = self.df_imputed[self.df_imputed[feature] > median_val]
            low_group = self.df_imputed[self.df_imputed[feature] <= median_val]
            
            # Create survival curves for each group
            high_prices = np.sort(high_group['price'])
            low_prices = np.sort(low_group['price'])
            
            high_survival = 1 - np.arange(len(high_prices)) / len(high_prices)
            low_survival = 1 - np.arange(len(low_prices)) / len(low_prices)
            
            # Plot survival curves
            ax.plot(high_prices, high_survival, 'r-', linewidth=2, 
                   label=f'High {feature} (>{median_val:.2f})')
            ax.plot(low_prices, low_survival, 'b-', linewidth=2, 
                   label=f'Low {feature} (≤{median_val:.2f})')
            
            ax.set_xlabel('Price')
            ax.set_ylabel('Survival Probability')
            ax.set_title(f'Stratified by {feature}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Calculate and display hazard ratio (approximate)
            high_median_price = high_group['price'].median()
            low_median_price = low_group['price'].median()
            hazard_ratio = high_median_price / low_median_price
            
            ax.text(0.05, 0.95, f'HR: {hazard_ratio:.2f}', transform=ax.transAxes,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                   verticalalignment='top')
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

    def _create_stratified_analysis(self, pdf):
        """Create stratified analysis by categorical variables"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Stratified Analysis by Categorical Variables', fontsize=16, fontweight='bold')
        
        categorical_vars = ['airport', 'waterbody', 'bus_ter']
        
        for idx, cat_var in enumerate(categorical_vars):
            if cat_var in self.df_imputed.columns:
                ax = axes[idx]
                
                # Get unique categories
                categories = self.df_imputed[cat_var].unique()
                
                # Plot survival curves for each category
                for category in categories:
                    category_data = self.df_imputed[self.df_imputed[cat_var] == category]
                    prices_sorted = np.sort(category_data['price'])
                    survival_prob = 1 - np.arange(len(prices_sorted)) / len(prices_sorted)
                    
                    ax.plot(prices_sorted, survival_prob, linewidth=2, 
                           label=f'{category} (n={len(category_data)})')
                
                ax.set_xlabel('Price')
                ax.set_ylabel('Survival Probability')
                ax.set_title(f'Stratified by {cat_var}')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("="*70)
        print("HOUSE PRICE ANALYSIS WITH MULTIPLE IMPUTATION")
        print("="*70)
        
        # Step 1: Data overview
        print(f"\n1. DATASET OVERVIEW:")
        print(f"   - Rows: {len(self.df)}, Columns: {len(self.df.columns)}")
        print(f"   - Price range: ${self.df['price'].min():.2f} - ${self.df['price'].max():.2f}")
        print(f"   - Average price: ${self.df['price'].mean():.2f}")
        
        # Step 2: Outlier detection summary
        print(f"\n2. OUTLIER DETECTION:")
        outliers_summary = self.get_outliers_summary()
        top_outliers = outliers_summary.nlargest(3, 'outlier_count')
        print(f"   - Variables with most outliers: {top_outliers.index.tolist()}")
        print(f"   - Maximum outliers in a variable: {top_outliers['outlier_count'].max()}")
        
        # Step 3: Apply multiple imputation
        print(f"\n3. APPLYING MULTIPLE IMPUTATION...")
        self.multiple_imputation_outliers()
        
        # Step 4: Generate regression plots
        print(f"\n4. CREATING REGRESSION PLOTS...")
        self.create_regression_plots()
        
        # Step 5: Generate Cox-style analysis
        print(f"\n5. CREATING COX-STYLE ANALYSIS...")
        self.create_cox_style_analysis()
        
        # Final summary
        print(f"\n6. ANALYSIS COMPLETE!")
        print(f"   ✓ Regression analysis: 'regression_analysis.pdf'")
        print(f"   ✓ Cox-style analysis: 'cox_style_analysis.pdf'")
        print(f"   ✓ Total outliers handled: {self.outlier_flags.sum().sum()}")
        
        if self.best_model:
            print(f"   ✓ Best model: {type(self.best_model).__name__}")
            print(f"   ✓ Top 3 features: {[feat for feat, imp in self.feature_importance[:3]]}")
        
        print(f"\n" + "="*70)

# Additional utility function for quick analysis
def quick_price_behavior_analysis(file_path):
    """Quick analysis function for immediate price behavior visualization"""
    analyzer = HousePriceAnalysis(file_path)
    analyzer.multiple_imputation_outliers()
    
    # Create a quick overview PDF
    with PdfPages('price_behavior_quick.pdf') as pdf:
        # Price distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram
        ax1.hist(analyzer.df_imputed['price'], bins=30, alpha=0.7, color='lightblue', edgecolor='black')
        ax1.set_xlabel('Price')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Price Distribution After Multiple Imputation')
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2.boxplot(analyzer.df_imputed['price'])
        ax2.set_ylabel('Price')
        ax2.set_title('Price Box Plot')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
        
        # Top correlations
        correlations = []
        for col in analyzer.numerical_cols:
            if col != 'price':
                corr = analyzer.df_imputed[[col, 'price']].corr().iloc[0,1]
                correlations.append((col, corr))
        
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        features = [corr[0] for corr in correlations[:10]]
        corr_values = [corr[1] for corr in correlations[:10]]
        
        bars = ax.barh(features, corr_values, color=['steelblue' if x > 0 else 'lightcoral' for x in corr_values])
        ax.set_xlabel('Correlation with Price')
        ax.set_title('Top 10 Features Correlated with Price')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3)
        
        # Add correlation values
        for bar, corr in zip(bars, corr_values):
            width = bar.get_width()
            ax.text(width + (0.01 if width >= 0 else -0.03), bar.get_y() + bar.get_height()/2,
                   f'{corr:.3f}', ha='left' if width >= 0 else 'right', va='center', fontweight='bold')
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
    
    print("Quick analysis complete: 'price_behavior_quick.pdf' created")

# Main execution
if __name__ == "__main__":
    # Initialize the analysis
    analyzer = HousePriceAnalysis('House_PriceReg.csv')
    
    # Run the complete analysis
    analyzer.run_complete_analysis()
    
    # Also create quick analysis
    quick_price_behavior_analysis('House_PriceReg.csv')
