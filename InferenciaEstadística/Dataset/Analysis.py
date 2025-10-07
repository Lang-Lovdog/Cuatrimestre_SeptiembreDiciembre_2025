# house_price_permutation_analysis.py
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl
import numpy as np
from itertools import permutations
import os
import json
from matplotlib import rcParams
import seaborn as sns

def load_and_preprocess_data(csv_path):
    """Load CSV file and handle basic preprocessing"""
    df = pd.read_csv(csv_path)
    print(f"📁 Loaded dataset with shape: {df.shape}")
    print(f"📊 Columns: {list(df.columns)}")
    return df

def analyze_categorical_variables(df):
    """Analyze and return categorical variables information"""
    categorical_info = {}
    
    for column in df.columns:
        if df[column].dtype == 'object' or df[column].dtype.name == 'category':
            unique_values = df[column].unique()
            value_counts = df[column].value_counts()
            
            categorical_info[column] = {
                'original_values': list(unique_values),
                'value_counts': value_counts.to_dict(),
                'mapping_to_numeric': {value: idx for idx, value in enumerate(unique_values)},
                'total_unique': len(unique_values)
            }
            print(f"   📋 '{column}': {len(unique_values)} unique values")
    
    return categorical_info

def convert_categorical_to_numeric(df, categorical_info):
    """Convert categorical columns to numeric codes using provided mapping"""
    df_processed = df.copy()
    
    for column, info in categorical_info.items():
        mapping = info['mapping_to_numeric']
        df_processed[column] = df[column].map(mapping)
        print(f"   🔢 '{column}': {len(mapping)} categories → numeric codes")
    
    return df_processed

def setup_theme(theme_name='viridis', use_tex=False):
    """Setup matplotlib theme and style"""
    # Set color cycle based on theme
    if theme_name in plt.colormaps():
        plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.get_cmap(theme_name)(np.linspace(0, 1, 10)))
    
    # LaTeX setup if requested
    if use_tex:
        try:
            rcParams['text.usetex'] = True
            rcParams['font.family'] = 'serif'
            rcParams['font.serif'] = ['Computer Modern Roman']
            print("   ✅ LaTeX rendering enabled")
        except:
            print("   ❌ LaTeX not available, falling back to default rendering")
            rcParams['text.usetex'] = False
    else:
        rcParams['text.usetex'] = False
    
    # General style improvements
    plt.style.use('default')
    rcParams['figure.titlesize'] = 16
    rcParams['figure.titleweight'] = 'bold'
    rcParams['axes.titlesize'] = 14
    rcParams['axes.labelsize'] = 12

def create_single_page_plots(df, categorical_info, output_pdf_path, theme='viridis', use_tex=False):
    """Create permutation plots with one plot per page and section titles"""
    
    # Convert categorical columns to numeric
    df_processed = convert_categorical_to_numeric(df, categorical_info)
    
    # Get all column pairs (permutations)
    columns = df_processed.columns
    pairs = list(permutations(columns, 2))
    
    print(f"   🎨 Generating {len(pairs)} permutation plots (one per page)...")
    
    # Setup theme
    setup_theme(theme, use_tex)
    
    # Create PDF with all plots (one per page)
    with PdfPages(output_pdf_path) as pdf:
        for i, (ki, kj) in enumerate(pairs):
            # Create a new figure for each plot
            fig = plt.figure(figsize=(14, 8))
            
            # Create a grid for the page: title + main plot + side plots
            gs = plt.GridSpec(3, 2, figure=fig, height_ratios=[1, 4, 2], width_ratios=[3, 1])
            
            # Title section
            ax_title = fig.add_subplot(gs[0, :])
            ax_title.axis('off')
            
            # Main plot section
            ax_main = fig.add_subplot(gs[1, 0])
            
            # Side plot section (top right)
            ax_side1 = fig.add_subplot(gs[1, 1])
            
            # Bottom section for additional info
            ax_side2 = fig.add_subplot(gs[2, :])
            ax_side2.axis('off')
            
            try:
                # Set page/section title
                ki_type = "categorical" if ki in categorical_info else "numeric"
                kj_type = "categorical" if kj in categorical_info else "numeric"
                
                title_text = f"{ki} compared to {kj}\n{ki} ({ki_type}) vs {kj} ({kj_type})"
                
                ax_title.text(0.5, 0.5, title_text, 
                            transform=ax_title.transAxes, 
                            ha='center', va='center',
                            fontsize=18, fontweight='bold',
                            bbox=dict(boxstyle="round,pad=1", facecolor='lightgray', alpha=0.8))
                
                # Main scatter plot with theme colors
                cmap = mpl.colormaps[theme]
                scatter = ax_main.scatter(df_processed[ki], df_processed[kj], 
                                        c=range(len(df_processed)), cmap=cmap, 
                                        alpha=0.7, s=50, edgecolors='white', linewidth=0.5)
                ax_main.set_xlabel(ki, fontweight='bold', fontsize=12)
                ax_main.set_ylabel(kj, fontweight='bold', fontsize=12)
                ax_main.set_title('Scatter Plot', fontweight='bold', fontsize=14)
                ax_main.grid(True, alpha=0.3, linestyle='--')
                
                # Add trend line for numeric vs numeric
                if ki not in categorical_info and kj not in categorical_info:
                    try:
                        z = np.polyfit(df_processed[ki], df_processed[kj], 1)
                        p = np.poly1d(z)
                        ax_main.plot(df_processed[ki], p(df_processed[ki]), "r--", alpha=0.8, linewidth=2, label='Trend')
                        ax_main.legend()
                    except:
                        pass  # Skip trend line if cannot compute
                
                # Side plot 1: Distribution of first variable
                if ki in categorical_info:
                    # Show value distribution for categorical variable
                    value_counts = df[ki].value_counts().sort_index()
                    colors = [cmap(i/len(value_counts)) for i in range(len(value_counts))]
                    bars = ax_side1.bar(range(len(value_counts)), value_counts.values, color=colors)
                    ax_side1.set_xticks(range(len(value_counts)))
                    ax_side1.set_xticklabels([str(x) for x in value_counts.index], fontsize=8, rotation=45)
                    ax_side1.set_title(f'{ki} Distribution', fontsize=10, fontweight='bold')
                    ax_side1.set_ylabel('Count')
                else:
                    # Show histogram for numeric variable
                    ax_side1.hist(df_processed[ki], bins=15, color=cmap(0.5), alpha=0.7, edgecolor='black')
                    ax_side1.set_title(f'{ki} Distribution', fontsize=10, fontweight='bold')
                    ax_side1.set_ylabel('Frequency')
                
                # Bottom section: Statistics and information
                stats_text = []
                
                # Basic statistics
                if ki not in categorical_info:
                    stats_text.append(f"{ki} (numeric): min={df_processed[ki].min():.2f}, max={df_processed[ki].max():.2f}, mean={df_processed[ki].mean():.2f}")
                else:
                    stats_text.append(f"{ki} (categorical): {len(categorical_info[ki]['original_values'])} categories")
                
                if kj not in categorical_info:
                    stats_text.append(f"{kj} (numeric): min={df_processed[kj].min():.2f}, max={df_processed[kj].max():.2f}, mean={df_processed[kj].mean():.2f}")
                else:
                    stats_text.append(f"{kj} (categorical): {len(categorical_info[kj]['original_values'])} categories")
                
                # Correlation for numeric pairs
                if ki not in categorical_info and kj not in categorical_info:
                    corr = df_processed[ki].corr(df_processed[kj])
                    stats_text.append(f"Correlation: {corr:.3f}")
                
                stats_text = "\n".join(stats_text)
                ax_side2.text(0.02, 0.8, stats_text, transform=ax_side2.transAxes, 
                            fontsize=10, verticalalignment='top',
                            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.3))
                
                # Add page number
                fig.text(0.98, 0.02, f'Page {i+1}/{len(pairs)}', 
                        ha='right', va='bottom', fontsize=8, style='italic')
                
            except Exception as e:
                # Clear and create error message
                fig.clear()
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.text(0.5, 0.5, f'Error plotting {ki} vs {kj}\n\n{str(e)}', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title(f'ERROR: {ki} compared to {kj}', fontweight='bold')
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            if (i + 1) % 50 == 0:
                print(f"      📄 Generated {i + 1}/{len(pairs)} pages...")
    
    print(f"   ✅ All plots saved to: {output_pdf_path}")
    return pairs

def export_analysis_data(df, categorical_info, pairs, output_dir):
    """Export extra analysis data for each plot"""
    analysis_data = {}
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"   📊 Exporting extra analysis data for {len(pairs)} pairs...")
    
    for i, (ki, kj) in enumerate(pairs):
        plot_key = f"{ki}_vs_{kj}"
        analysis_data[plot_key] = {}
        
        try:
            # Basic statistics
            if ki in categorical_info:
                analysis_data[plot_key][f'{ki}_stats'] = {
                    'type': 'categorical',
                    'unique_values': categorical_info[ki]['total_unique'],
                    'value_distribution': categorical_info[ki]['value_counts'],
                    'mapping': categorical_info[ki]['mapping_to_numeric']
                }
            else:
                analysis_data[plot_key][f'{ki}_stats'] = {
                    'type': 'numeric',
                    'min': float(df[ki].min()),
                    'max': float(df[ki].max()),
                    'mean': float(df[ki].mean()),
                    'std': float(df[ki].std()),
                    'median': float(df[ki].median())
                }
            
            if kj in categorical_info:
                analysis_data[plot_key][f'{kj}_stats'] = {
                    'type': 'categorical',
                    'unique_values': categorical_info[kj]['total_unique'],
                    'value_distribution': categorical_info[kj]['value_counts'],
                    'mapping': categorical_info[kj]['mapping_to_numeric']
                }
            else:
                analysis_data[plot_key][f'{kj}_stats'] = {
                    'type': 'numeric',
                    'min': float(df[kj].min()),
                    'max': float(df[kj].max()),
                    'mean': float(df[kj].mean()),
                    'std': float(df[kj].std()),
                    'median': float(df[kj].median())
                }
            
            # Cross-analysis based on variable types
            df_processed = convert_categorical_to_numeric(df, categorical_info)
            
            if ki in categorical_info and kj not in categorical_info:
                # Categorical vs Numeric: group statistics
                grouped_stats = df.groupby(ki)[kj].agg(['mean', 'std', 'count', 'min', 'max']).round(3)
                analysis_data[plot_key]['cross_analysis'] = {
                    'type': 'categorical_vs_numeric',
                    'grouped_stats': grouped_stats.to_dict()
                }
            elif kj in categorical_info and ki not in categorical_info:
                # Numeric vs Categorical: group statistics
                grouped_stats = df.groupby(kj)[ki].agg(['mean', 'std', 'count', 'min', 'max']).round(3)
                analysis_data[plot_key]['cross_analysis'] = {
                    'type': 'numeric_vs_categorical',
                    'grouped_stats': grouped_stats.to_dict()
                }
            elif ki in categorical_info and kj in categorical_info:
                # Categorical vs Categorical: contingency table
                contingency = pd.crosstab(df[ki], df[kj]).to_dict()
                analysis_data[plot_key]['cross_analysis'] = {
                    'type': 'categorical_vs_categorical',
                    'contingency_table': contingency
                }
            else:
                # Numeric vs Numeric: correlation and regression
                correlation = float(df_processed[ki].corr(df_processed[kj]))
                analysis_data[plot_key]['cross_analysis'] = {
                    'type': 'numeric_vs_numeric',
                    'correlation': round(correlation, 4),
                    'covariance': float(df_processed[ki].cov(df_processed[kj]))
                }
                
        except Exception as e:
            analysis_data[plot_key]['error'] = str(e)
        
        if (i + 1) % 100 == 0:
            print(f"      📈 Processed {i + 1}/{len(pairs)} analysis pairs...")
    
    # Save analysis data
    analysis_file = os.path.join(output_dir, "detailed_analysis_data.json")
    with open(analysis_file, 'w') as f:
        json.dump(analysis_data, f, indent=2, default=str)
    
    # Also save a simplified version
    simplified_analysis = {}
    for plot_key, data in analysis_data.items():
        simplified_analysis[plot_key] = {
            'var1_type': data[f"{plot_key.split('_vs_')[0]}_stats"]['type'],
            'var2_type': data[f"{plot_key.split('_vs_')[1]}_stats"]['type'],
            'analysis_type': data.get('cross_analysis', {}).get('type', 'unknown'),
            'correlation' if 'correlation' in data.get('cross_analysis', {}) else 'key_metric': 
                data.get('cross_analysis', {}).get('correlation', 'N/A')
        }
    
    simplified_file = os.path.join(output_dir, "summary_analysis.json")
    with open(simplified_file, 'w') as f:
        json.dump(simplified_analysis, f, indent=2)
    
    print(f"   ✅ Extra analysis data saved to: {output_dir}/")
    return analysis_data

def generate_dataset_summary(df, categorical_info, output_dir):
    """Generate a comprehensive dataset summary"""
    print("   📋 Generating dataset summary...")
    
    summary = {
        'dataset_info': {
            'total_rows': df.shape[0],
            'total_columns': df.shape[1],
            'total_numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'total_categorical_columns': len(categorical_info)
        },
        'column_details': {},
        'categorical_variables': categorical_info
    }
    
    for column in df.columns:
        col_info = {
            'dtype': str(df[column].dtype),
            'non_null_count': int(df[column].count()),
            'null_count': int(df[column].isnull().sum()),
            'null_percentage': round(df[column].isnull().sum() / len(df) * 100, 2)
        }
        
        if column in categorical_info:
            col_info['type'] = 'categorical'
            col_info['unique_values'] = categorical_info[column]['total_unique']
        else:
            col_info['type'] = 'numeric'
            col_info.update({
                'min': float(df[column].min()),
                'max': float(df[column].max()),
                'mean': float(df[column].mean()),
                'std': float(df[column].std()),
                'median': float(df[column].median())
            })
        
        summary['column_details'][column] = col_info
    
    # Save summary
    summary_file = os.path.join(output_dir, "dataset_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Print key insights
    print(f"\n   🔍 DATASET INSIGHTS:")
    print(f"      • Total variables: {summary['dataset_info']['total_columns']}")
    print(f"      • Numeric variables: {summary['dataset_info']['total_numeric_columns']}")
    print(f"      • Categorical variables: {summary['dataset_info']['total_categorical_columns']}")
    print(f"      • Total observations: {summary['dataset_info']['total_rows']}")
    
    if categorical_info:
        print(f"      • Categorical variables found: {list(categorical_info.keys())}")
    
    return summary

def create_correlation_heatmap(df, output_dir, theme='viridis'):
    """Create a correlation heatmap for numeric variables"""
    numeric_df = df.select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) > 1:
        print("   🔥 Creating correlation heatmap...")
        
        plt.figure(figsize=(12, 10))
        correlation_matrix = numeric_df.corr()
        
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap=theme, 
                   center=0, fmt='.2f', linewidths=0.5, square=True,
                   cbar_kws={"shrink": .8})
        
        plt.title('Correlation Matrix - Numeric Variables', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        heatmap_path = os.path.join(output_dir, "correlation_heatmap.png")
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Correlation heatmap saved to: {heatmap_path}")

def main(csv_file_path, 
         pdf_output_name=None,
         analyze_categorical=True,
         export_extra_analysis=True,
         extra_analysis_dir="analysis_output",
         theme="viridis",
         use_tex=False,
         export_tex=False,
         tex_output_name=None):
    """
    Enhanced main function for permutation plot analysis
    
    Parameters:
    - csv_file_path: Path to the CSV file
    - pdf_output_name: Name for the output PDF file
    - analyze_categorical: Whether to analyze and report categorical variables
    - export_extra_analysis: Whether to export extra analysis data per plot
    - extra_analysis_dir: Directory for extra analysis output
    - theme: Matplotlib theme name (viridis, plasma, magma, inferno, coolwarm, etc.)
    - use_tex: Whether to use LaTeX for text rendering in plots
    - export_tex: Whether to export as LaTeX document
    - tex_output_name: Name for LaTeX output file
    """
    
    if not os.path.exists(csv_file_path):
        print(f"❌ Error: File '{csv_file_path}' not found.")
        return
    
    print("=" * 70)
    print("🏠 HOUSE PRICE PERMUTATION ANALYSIS TOOL")
    print("=" * 70)
    
    # Generate default output names if not provided
    base_name = os.path.splitext(os.path.basename(csv_file_path))[0]
    if pdf_output_name is None:
        pdf_output_name = f"{base_name}_permutation_analysis_{theme}.pdf"
    if extra_analysis_dir == "analysis_output":
        extra_analysis_dir = f"{base_name}_analysis_{theme}"
    
    print(f"📁 Dataset: {csv_file_path}")
    print(f"🎨 Theme: {theme}")
    print(f"📊 Output PDF: {pdf_output_name}")
    print(f"📈 Analysis Directory: {extra_analysis_dir}")
    
    # Load data
    df = load_and_preprocess_data(csv_file_path)
    
    # Analyze categorical variables
    categorical_info = {}
    if analyze_categorical:
        print("\n🔍 ANALYZING CATEGORICAL VARIABLES...")
        categorical_info = analyze_categorical_variables(df)
        
        if categorical_info:
            print(f"   ✅ Found {len(categorical_info)} categorical variables")
            # Save categorical mapping
            os.makedirs(extra_analysis_dir, exist_ok=True)
            mapping_file = os.path.join(extra_analysis_dir, "categorical_mapping.json")
            with open(mapping_file, 'w') as f:
                json.dump(categorical_info, f, indent=2, default=str)
        else:
            print("   ℹ️  No categorical variables found in the dataset")
    
    # Generate dataset summary
    dataset_summary = generate_dataset_summary(df, categorical_info, extra_analysis_dir)
    
    # Create correlation heatmap
    create_correlation_heatmap(df, extra_analysis_dir, theme)
    
    # Generate permutation plots
    print(f"\n🎨 GENERATING PERMUTATION PLOTS...")
    pairs = create_single_page_plots(df, categorical_info, pdf_output_name, theme, use_tex)
    
    # Export extra analysis data if requested
    if export_extra_analysis:
        print(f"\n📊 EXPORTING EXTRA ANALYSIS DATA...")
        analysis_data = export_analysis_data(df, categorical_info, pairs, extra_analysis_dir)
    
    print(f"\n" + "=" * 70)
    print("✅ ANALYSIS COMPLETED SUCCESSFULLY!")
    print(f"📁 Main Output: {pdf_output_name}")
    print(f"📊 Analysis Files: {extra_analysis_dir}/")
    print(f"📈 Total Plots Generated: {len(pairs)}")
    print("=" * 70)

# Example usage and theme testing
def run_comprehensive_analysis(csv_path, themes=None):
    """Run analysis with multiple themes"""
    if themes is None:
        themes = ['viridis', 'plasma', 'coolwarm', 'tab10']
    
    print("🚀 STARTING COMPREHENSIVE ANALYSIS WITH MULTIPLE THEMES")
    
    for theme in themes:
        print(f"\n{'='*50}")
        print(f"🎨 PROCESSING THEME: {theme.upper()}")
        print(f"{'='*50}")
        
        try:
            main(csv_path,
                 pdf_output_name=f"House_Price_Analysis_{theme.capitalize()}.pdf",
                 analyze_categorical=True,
                 export_extra_analysis=True,
                 extra_analysis_dir=f"House_Price_Analysis_{theme}",
                 theme=theme,
                 use_tex=False,
                 export_tex=False)
        except Exception as e:
            print(f"❌ Error with theme {theme}: {str(e)}")
            continue
    
    print(f"\n🎉 ALL THEMES COMPLETED!")
    print("📊 You now have multiple versions with different color schemes")

# Quick analysis function
def quick_analysis(csv_path, theme='viridis'):
    """Run a quick analysis with default settings"""
    print("⚡ RUNNING QUICK ANALYSIS...")
    main(csv_path,
         pdf_output_name=f"House_Price_Quick_Analysis.pdf",
         analyze_categorical=True,
         export_extra_analysis=False,
         theme=theme,
         use_tex=False)

if __name__ == "__main__":
    # Your dataset path
    csv_file = "House_PriceReg.csv"
    
    # Check if file exists
    if not os.path.exists(csv_file):
        print(f"❌ Error: File '{csv_file}' not found.")
        print("💡 Please make sure the CSV file is in the same directory as this script.")
        exit(1)
    
    print("🏠 HOUSE PRICE REGRESSION DATASET ANALYSIS")
    print("Choose an option:")
    print("1. Quick analysis (viridis theme, basic output)")
    print("2. Single theme analysis (choose your preferred theme)")
    print("3. Comprehensive analysis (multiple themes)")
    print("4. Custom analysis (full parameter control)")
    
    choice = input("Enter your choice (1-4): ").strip()
    
    if choice == "1":
        quick_analysis(csv_file)
        
    elif choice == "2":
        themes = ['viridis', 'plasma', 'magma', 'inferno', 'coolwarm', 'tab10']
        print("\nAvailable themes:")
        for i, theme in enumerate(themes, 1):
            print(f"  {i}. {theme}")
        theme_choice = input("Choose theme (1-6): ").strip()
        try:
            selected_theme = themes[int(theme_choice) - 1]
            main(csv_file, theme=selected_theme)
        except (ValueError, IndexError):
            print("Invalid choice, using default 'viridis' theme")
            main(csv_file, theme='viridis')
            
    elif choice == "3":
        run_comprehensive_analysis(csv_file)
        
    elif choice == "4":
        # Custom parameters
        theme = input("Enter theme (viridis/plasma/magma/inferno/coolwarm): ").strip() or 'viridis'
        use_tex = input("Use LaTeX rendering? (y/n): ").strip().lower() == 'y'
        export_extra = input("Export extra analysis data? (y/n): ").strip().lower() == 'y'
        
        main(csv_file,
             theme=theme,
             use_tex=use_tex,
             export_extra_analysis=export_extra)
    else:
        print("Invalid choice. Running quick analysis...")
        quick_analysis(csv_file)
