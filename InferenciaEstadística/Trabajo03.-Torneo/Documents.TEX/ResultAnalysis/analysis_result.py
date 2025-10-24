import pandas as pd
import numpy as np
from scipy.stats import shapiro, ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    """
    Load and prepare data from both files
    """
    # Load peer's Excel file
    df_peer = pd.read_excel('F1_Score.xlsx', sheet_name='Hoja1')
    
    # Load your CSV file
    df_you = pd.read_csv('all_matching_records.csv')
    
    return df_peer, df_you

def extract_model_data(df_peer, df_you):
    """
    Extract SVM and Random Forest F1 scores from both datasets
    """
    # Extract from peer's data (Excel)
    svm_peer = df_peer[df_peer['Model'] == 'RbfSVM']['F1_Score'].values
    rf_peer = df_peer[df_peer['Model'] == 'Rf']['F1_Score'].values
    
    # Extract from your data (CSV)
    svm_you = df_you[df_you['Model'] == 'RbfSVM']['F1_Score'].values
    rf_you = df_you[df_you['Model'] == 'RandomForest']['F1_Score'].values
    
    return svm_peer, rf_peer, svm_you, rf_you

def perform_normality_test(data, name):
    """
    Perform Shapiro-Wilk test for normality
    """
    if len(data) < 3:
        return {"name": name, "statistic": None, "p_value": None, "normal": None, "error": "Insufficient data"}
    
    stat, p_value = shapiro(data)
    is_normal = p_value > 0.05
    
    return {
        "name": name,
        "statistic": stat,
        "p_value": p_value,
        "normal": is_normal,
        "error": None
    }

def perform_ttest(data1, data2, name1, name2):
    """
    Perform independent T-test between two datasets
    """
    if len(data1) < 2 or len(data2) < 2:
        return {
            "comparison": f"{name1} vs {name2}",
            "t_statistic": None,
            "p_value": None,
            "significant": None,
            "cohens_d": None,
            "error": "Insufficient data"
        }
    
    t_stat, p_value = ttest_ind(data1, data2)
    is_significant = p_value <= 0.05
    
    # Calculate effect size (Cohen's d)
    n1, n2 = len(data1), len(data2)
    pooled_std = np.sqrt(((n1-1)*np.var(data1, ddof=1) + (n2-1)*np.var(data2, ddof=1)) / (n1+n2-2))
    cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std
    
    return {
        "comparison": f"{name1} vs {name2}",
        "t_statistic": t_stat,
        "p_value": p_value,
        "significant": is_significant,
        "cohens_d": cohens_d,
        "error": None
    }

def get_descriptive_stats(data, name):
    """
    Calculate descriptive statistics for a dataset
    """
    if len(data) == 0:
        return {
            "name": name,
            "n": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "range": None
        }
    
    return {
        "name": name,
        "n": len(data),
        "mean": np.mean(data),
        "std": np.std(data, ddof=1),
        "min": np.min(data),
        "max": np.max(data),
        "range": np.max(data) - np.min(data)
    }

def create_summary_dataframe(normality_results, ttest_results, stats_results):
    """
    Create a comprehensive summary DataFrame
    """
    summary_data = []
    
    # Add normality test results
    for norm_result in normality_results:
        summary_data.append({
            'Analysis_Type': 'Normality_Test',
            'Dataset': norm_result['name'],
            'Statistic': norm_result['statistic'],
            'P_Value': norm_result['p_value'],
            'Interpretation': 'Normal' if norm_result['normal'] else 'Not Normal' if norm_result['normal'] is not None else 'N/A',
            'Effect_Size': None,
            'Significance': None
        })
    
    # Add t-test results
    for ttest_result in ttest_results:
        summary_data.append({
            'Analysis_Type': 'T_Test',
            'Dataset': ttest_result['comparison'],
            'Statistic': ttest_result['t_statistic'],
            'P_Value': ttest_result['p_value'],
            'Interpretation': 'Significant' if ttest_result['significant'] else 'Not Significant' if ttest_result['significant'] is not None else 'N/A',
            'Effect_Size': ttest_result['cohens_d'],
            'Significance': ttest_result['significant']
        })
    
    # Add descriptive statistics
    for stat_result in stats_results:
        summary_data.append({
            'Analysis_Type': 'Descriptive_Stats',
            'Dataset': stat_result['name'],
            'Statistic': stat_result['mean'],  # Using mean as the main statistic
            'P_Value': None,
            'Interpretation': f"N={stat_result['n']}, Range={stat_result['range']:.4f}",
            'Effect_Size': None,
            'Significance': None
        })
    
    return pd.DataFrame(summary_data)

def export_detailed_report(normality_results, ttest_results, stats_results, filename="statistical_analysis_report.xlsx"):
    """
    Export a detailed Excel report with multiple sheets
    """
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        
        # Sheet 1: Executive Summary
        exec_summary = []
        for ttest in ttest_results:
            if ttest['error'] is None:
                exec_summary.append({
                    'Comparison': ttest['comparison'],
                    'Significant_Difference': 'Yes' if ttest['significant'] else 'No',
                    'P_Value': ttest['p_value'],
                    'Effect_Size_Cohens_d': ttest['cohens_d'],
                    'Effect_Size_Interpretation': get_effect_size_interpretation(ttest['cohens_d'])
                })
        
        exec_df = pd.DataFrame(exec_summary)
        exec_df.to_excel(writer, sheet_name='Executive_Summary', index=False)
        
        # Sheet 2: Normality Tests
        normality_df = pd.DataFrame(normality_results)
        normality_df.to_excel(writer, sheet_name='Normality_Tests', index=False)
        
        # Sheet 3: T-Test Results
        ttest_df = pd.DataFrame(ttest_results)
        ttest_df.to_excel(writer, sheet_name='T_Test_Results', index=False)
        
        # Sheet 4: Descriptive Statistics
        stats_df = pd.DataFrame(stats_results)
        stats_df.to_excel(writer, sheet_name='Descriptive_Statistics', index=False)
        
        # Sheet 5: Combined Summary
        summary_df = create_summary_dataframe(normality_results, ttest_results, stats_results)
        summary_df.to_excel(writer, sheet_name='Combined_Summary', index=False)
        
        # Sheet 6: Raw Data Comparison
        raw_data = []
        datasets = {
            'Peer_SVM': stats_results[0],
            'Your_SVM': stats_results[1],
            'Peer_RF': stats_results[2],
            'Your_RF': stats_results[3]
        }
        
        for name, stats in datasets.items():
            raw_data.append({
                'Dataset': name,
                'Sample_Size': stats['n'],
                'Mean_F1_Score': stats['mean'],
                'Std_Deviation': stats['std'],
                'Min_F1_Score': stats['min'],
                'Max_F1_Score': stats['max'],
                'Range': stats['range']
            })
        
        raw_df = pd.DataFrame(raw_data)
        raw_df.to_excel(writer, sheet_name='Raw_Data_Summary', index=False)

def get_effect_size_interpretation(cohens_d):
    """
    Interpret Cohen's d effect size
    """
    if cohens_d is None:
        return "N/A"
    
    abs_d = abs(cohens_d)
    if abs_d < 0.2:
        return "Negligible"
    elif abs_d < 0.5:
        return "Small"
    elif abs_d < 0.8:
        return "Medium"
    else:
        return "Large"

def plot_comparisons(svm_peer, rf_peer, svm_you, rf_you):
    """
    Create and save comparison plots in EPS format
    """
    # Set style for better looking plots
    plt.style.use('default')
    sns.set_palette("Set2")
    
    # Create figure 1: Boxplots
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # SVM Boxplot
    svm_data = [svm_peer, svm_you]
    svm_labels = ['Peer SVM', 'Your SVM']
    ax1.boxplot(svm_data, labels=svm_labels)
    ax1.set_title('SVM F1 Score Comparison')
    ax1.set_ylabel('F1 Score')
    ax1.grid(True, alpha=0.3)
    
    # Random Forest Boxplot
    rf_data = [rf_peer, rf_you]
    rf_labels = ['Peer RF', 'Your RF']
    ax2.boxplot(rf_data, labels=rf_labels)
    ax2.set_title('Random Forest F1 Score Comparison')
    ax2.set_ylabel('F1 Score')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_comparison_boxplots.eps', format='eps', dpi=300, bbox_inches='tight')
    plt.savefig('model_comparison_boxplots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create figure 2: Distribution plots
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # SVM Distributions
    ax1.hist(svm_peer, alpha=0.7, label='Peer SVM', bins=8, density=True, edgecolor='black')
    ax1.hist(svm_you, alpha=0.7, label='Your SVM', bins=8, density=True, edgecolor='black')
    ax1.set_title('SVM F1 Score Distributions')
    ax1.set_xlabel('F1 Score')
    ax1.set_ylabel('Density')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Random Forest Distributions
    ax2.hist(rf_peer, alpha=0.7, label='Peer RF', bins=8, density=True, edgecolor='black')
    ax2.hist(rf_you, alpha=0.7, label='Your RF', bins=8, density=True, edgecolor='black')
    ax2.set_title('Random Forest F1 Score Distributions')
    ax2.set_xlabel('F1 Score')
    ax2.set_ylabel('Density')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_comparison_distributions.eps', format='eps', dpi=300, bbox_inches='tight')
    plt.savefig('model_comparison_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """
    Main analysis function
    """
    print("Loading data...")
    df_peer, df_you = load_data()
    
    print("Extracting SVM and Random Forest data...")
    svm_peer, rf_peer, svm_you, rf_you = extract_model_data(df_peer, df_you)
    
    print(f"Peer SVM samples: {len(svm_peer)}")
    print(f"Peer RF samples: {len(rf_peer)}")
    print(f"Your SVM samples: {len(svm_you)}")
    print(f"Your RF samples: {len(rf_you)}")
    
    # Perform analyses
    print("\nPerforming statistical analyses...")
    
    # Normality tests
    normality_results = [
        perform_normality_test(svm_peer, "Peer SVM"),
        perform_normality_test(svm_you, "Your SVM"),
        perform_normality_test(rf_peer, "Peer Random Forest"),
        perform_normality_test(rf_you, "Your Random Forest")
    ]
    
    # T-tests
    ttest_results = [
        perform_ttest(svm_you, svm_peer, "Your SVM", "Peer SVM"),
        perform_ttest(rf_you, rf_peer, "Your Random Forest", "Peer Random Forest")
    ]
    
    # Descriptive statistics
    stats_results = [
        get_descriptive_stats(svm_peer, "Peer SVM"),
        get_descriptive_stats(svm_you, "Your SVM"),
        get_descriptive_stats(rf_peer, "Peer Random Forest"),
        get_descriptive_stats(rf_you, "Your Random Forest")
    ]
    
    # Export results
    report_filename = "statistical_analysis_report.xlsx"
    
    print(f"\nExporting comprehensive report to {report_filename}...")
    export_detailed_report(normality_results, ttest_results, stats_results, report_filename)
    
    print("Generating and saving plots...")
    plot_comparisons(svm_peer, rf_peer, svm_you, rf_you)
    
    # Print console summary
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    
    for ttest in ttest_results:
        if ttest['error'] is None:
            print(f"\n{ttest['comparison']}:")
            print(f"  Significant: {'YES' if ttest['significant'] else 'NO'}")
            print(f"  p-value: {ttest['p_value']:.4f}")
            print(f"  Effect Size (Cohen's d): {ttest['cohens_d']:.4f} ({get_effect_size_interpretation(ttest['cohens_d'])})")
    
    print(f"\nFiles created:")
    print(f"1. {report_filename} - Complete statistical analysis report")
    print(f"2. model_comparison_boxplots.eps - Boxplot comparison (EPS)")
    print(f"3. model_comparison_boxplots.png - Boxplot comparison (PNG)")
    print(f"4. model_comparison_distributions.eps - Distribution plots (EPS)")
    print(f"5. model_comparison_distributions.png - Distribution plots (PNG)")
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
