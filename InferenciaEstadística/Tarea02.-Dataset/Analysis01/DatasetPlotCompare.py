import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from itertools import permutations
import os
import json
from matplotlib import rcParams

def load_and_preprocess_data(csv_path):
    """Load CSV file and handle basic preprocessing"""
    df = pd.read_csv(csv_path)
    print(f"Loaded dataset with shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
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
            print(f"📊 Categorical column '{column}': {len(unique_values)} unique values")
    
    return categorical_info

def convert_categorical_to_numeric(df, categorical_info):
    """Convert categorical columns to numeric codes using provided mapping"""
    df_processed = df.copy()
    
    for column, info in categorical_info.items():
        mapping = info['mapping_to_numeric']
        df_processed[column] = df[column].map(mapping)
        print(f"🔢 Converted '{column}': {len(mapping)} categories → numeric codes")
    
    return df_processed

def export_analysis_data(df, categorical_info, pairs, output_dir):
    """Export extra analysis data for each plot"""
    analysis_data = {}
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for ki, kj in pairs:
        plot_key = f"{ki}_vs_{kj}"
        analysis_data[plot_key] = {}
        
        try:
            # Basic statistics
            if ki in categorical_info:
                analysis_data[plot_key][f'{ki}_stats'] = {
                    'type': 'categorical',
                    'unique_values': categorical_info[ki]['total_unique'],
                    'value_distribution': categorical_info[ki]['value_counts']
                }
            else:
                analysis_data[plot_key][f'{ki}_stats'] = {
                    'type': 'numeric',
                    'min': float(df[ki].min()),
                    'max': float(df[ki].max()),
                    'mean': float(df[ki].mean()),
                    'std': float(df[ki].std())
                }
            
            if kj in categorical_info:
                analysis_data[plot_key][f'{kj}_stats'] = {
                    'type': 'categorical',
                    'unique_values': categorical_info[kj]['total_unique'],
                    'value_distribution': categorical_info[kj]['value_counts']
                }
            else:
                analysis_data[plot_key][f'{kj}_stats'] = {
                    'type': 'numeric',
                    'min': float(df[kj].min()),
                    'max': float(df[kj].max()),
                    'mean': float(df[kj].mean()),
                    'std': float(df[kj].std())
                }
            
            # Cross-analysis
            if ki in categorical_info and kj not in categorical_info:
                # Categorical vs Numeric: group statistics
                grouped_stats = df.groupby(ki)[kj].agg(['mean', 'std', 'count']).to_dict()
                analysis_data[plot_key]['cross_analysis'] = {
                    'type': 'categorical_vs_numeric',
                    'grouped_stats': grouped_stats
                }
            elif kj in categorical_info and ki not in categorical_info:
                # Numeric vs Categorical: group statistics
                grouped_stats = df.groupby(kj)[ki].agg(['mean', 'std', 'count']).to_dict()
                analysis_data[plot_key]['cross_analysis'] = {
                    'type': 'numeric_vs_categorical',
                    'grouped_stats': grouped_stats
                }
            elif ki in categorical_info and kj in categorical_info:
                # Categorical vs Categorical: contingency table
                contingency = pd.crosstab(df[ki], df[kj]).to_dict()
                analysis_data[plot_key]['cross_analysis'] = {
                    'type': 'categorical_vs_categorical',
                    'contingency_table': contingency
                }
            else:
                # Numeric vs Numeric: correlation
                correlation = float(df[ki].corr(df[kj]))
                analysis_data[plot_key]['cross_analysis'] = {
                    'type': 'numeric_vs_numeric',
                    'correlation': correlation
                }
                
        except Exception as e:
            analysis_data[plot_key]['error'] = str(e)
    
    # Save analysis data
    analysis_file = os.path.join(output_dir, "plot_analysis_data.json")
    with open(analysis_file, 'w') as f:
        json.dump(analysis_data, f, indent=2, default=str)
    
    print(f"📈 Extra analysis data saved to: {analysis_file}")
    return analysis_data

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
            print("✅ LaTeX rendering enabled")
        except:
            print("❌ LaTeX not available, falling back to default rendering")
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
    
    print(f"🔄 Generating {len(pairs)} permutation plots (one per page)...")
    
    # Setup theme
    setup_theme(theme, use_tex)
    
    # Create PDF with all plots (one per page)
    with PdfPages(output_pdf_path) as pdf:
        for i, (ki, kj) in enumerate(pairs):
            # Create a new figure for each plot
            fig = plt.figure(figsize=(12, 8))
            
            # Create a grid for the page: title + main plot + optional side plots
            gs = plt.GridSpec(2, 2, figure=fig, height_ratios=[1, 4], width_ratios=[3, 1])
            
            # Title section
            ax_title = fig.add_subplot(gs[0, :])
            ax_title.axis('off')
            
            # Main plot section
            ax_main = fig.add_subplot(gs[1, 0])
            
            # Side plot section
            ax_side = fig.add_subplot(gs[1, 1])
            
            try:
                # Set page/section title
                ki_type = "categorical" if ki in categorical_info else "numeric"
                kj_type = "categorical" if kj in categorical_info else "numeric"
                
                if use_tex:
                    title_text = rf"\textbf{{{ki} vs {kj}}} \\ \small {ki} ({ki_type}) compared to {kj} ({kj_type})"
                else:
                    title_text = f"{ki} vs {kj}\n{ki} ({ki_type}) compared to {kj} ({kj_type})"
                
                ax_title.text(0.5, 0.5, title_text, 
                            transform=ax_title.transAxes, 
                            ha='center', va='center',
                            fontsize=16, fontweight='bold',
                            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.7))
                
                # Main scatter plot with theme colors
                colors = mpl.colormaps[theme](np.linspace(0, 1, len(df_processed)))
                scatter = ax_main.scatter(df_processed[ki], df_processed[kj], 
                                        c=colors, alpha=0.7, s=60, edgecolors='white', linewidth=0.5)
                ax_main.set_xlabel(ki, fontweight='bold')
                ax_main.set_ylabel(kj, fontweight='bold')
                ax_main.set_title('Scatter Plot', fontweight='bold')
                ax_main.grid(True, alpha=0.3, linestyle='--')
                
                # Add trend line for numeric vs numeric
                if ki not in categorical_info and kj not in categorical_info:
                    z = np.polyfit(df_processed[ki], df_processed[kj], 1)
                    p = np.poly1d(z)
                    ax_main.plot(df_processed[ki], p(df_processed[ki]), "r--", alpha=0.8, linewidth=2)
                
                # Side plot based on data types
                if ki in categorical_info:
                    # Show value distribution for categorical variable
                    value_counts = df[ki].value_counts()
                    bars = ax_side.barh(range(len(value_counts)), value_counts.values, 
                                      color=mpl.colormaps[theme](np.linspace(0, 1, len(value_counts))))
                    ax_side.set_yticks(range(len(value_counts)))
                    ax_side.set_yticklabels([str(x) for x in value_counts.index], fontsize=8)
                    ax_side.set_title(f'{ki} Distribution', fontsize=10, fontweight='bold')
                    ax_side.set_xlabel('Count')
                    
                elif kj in categorical_info:
                    # Show value distribution for categorical variable
                    value_counts = df[kj].value_counts()
                    bars = ax_side.barh(range(len(value_counts)), value_counts.values,
                                      color=mpl.colormaps[theme](np.linspace(0, 1, len(value_counts))))
                    ax_side.set_yticks(range(len(value_counts)))
                    ax_side.set_yticklabels([str(x) for x in value_counts.index], fontsize=8)
                    ax_side.set_title(f'{kj} Distribution', fontsize=10, fontweight='bold')
                    ax_side.set_xlabel('Count')
                    
                else:
                    # Both numeric - show histogram
                    ax_side.hist2d(df_processed[ki], df_processed[kj], bins=15, cmap=theme)
                    ax_side.set_xlabel(ki, fontsize=9)
                    ax_side.set_ylabel(kj, fontsize=9)
                    ax_side.set_title('2D Density', fontsize=10, fontweight='bold')
                
                # Add correlation or other metrics
                if ki not in categorical_info and kj not in categorical_info:
                    corr = df_processed[ki].corr(df_processed[kj])
                    fig.text(0.02, 0.02, f'Correlation: {corr:.3f}', 
                            fontsize=10, style='italic', 
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
                
            except Exception as e:
                # Clear and create error message
                fig.clear()
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.text(0.5, 0.5, f'Error plotting {ki} vs {kj}\n\n{str(e)}', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title(f'ERROR: {ki} vs {kj}', fontweight='bold')
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            if (i + 1) % 10 == 0:
                print(f"📄 Generated page {i + 1}/{len(pairs)}")
    
    print(f"✅ All plots saved to: {output_pdf_path}")
    return pairs

def export_to_tex(df, categorical_info, output_tex_path, theme='viridis'):
    """Export plots and analysis to LaTeX document"""
    
    print(f"📝 Generating LaTeX document: {output_tex_path}")
    
    columns = df.columns
    pairs = list(permutations(columns, 2))
    
    # Create plots directory
    plots_dir = os.path.splitext(output_tex_path)[0] + "_plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Generate individual plot images
    plot_files = []
    for i, (ki, kj) in enumerate(pairs):
        fig, ax = plt.subplots(figsize=(8, 6))
        try:
            df_processed = convert_categorical_to_numeric(df, categorical_info)
            ax.scatter(df_processed[ki], df_processed[kj], alpha=0.7, s=40, 
                     c=mpl.colormaps[theme](np.linspace(0, 1, len(df_processed))))
            ax.set_xlabel(ki)
            ax.set_ylabel(kj)
            ax.set_title(f'{ki} vs {kj}')
            ax.grid(True, alpha=0.3)
            
            plot_filename = f"plot_{i:03d}.png"
            plot_path = os.path.join(plots_dir, plot_filename)
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plot_files.append((ki, kj, plot_filename))
            
        except Exception as e:
            print(f"Error generating plot {i}: {e}")
        finally:
            plt.close()
    
    # Generate LaTeX document
    tex_content = r"""
\documentclass[11pt]{article}
\usepackage{graphicx}
\usepackage{geometry}
\usepackage{booktabs}
\usepackage{float}
\usepackage{subcaption}
\usepackage{hyperref}

\geometry{a4paper, margin=1in}

\title{Permutation Plot Analysis}
\author{Auto-generated Report}
\date{\today}

\begin{document}

\maketitle

\tableofcontents

\newpage
"""

    # Add sections for each plot
    for ki, kj, plot_file in plot_files:
        ki_type = "categorical" if ki in categorical_info else "numeric"
        kj_type = "categorical" if kj in categorical_info else "numeric"
        
        tex_content += f"""
\\section{{{ki} compared to {kj}}}

\\begin{{figure}}[H]
    \\centering
    \\includegraphics[width=0.8\\textwidth]{{{os.path.join(plots_dir, plot_file)}}}
    \\caption{{{ki} ({ki_type}) compared to {kj} ({kj_type})}}
    \\label{{fig:{ki}_{kj}}}
\\end{{figure}}

\\subsection*{{Variable Information}}
\\begin{{itemize}}
    \\item \\textbf{{{ki}}}: {ki_type} variable
    \\item \\textbf{{{kj}}}: {kj_type} variable
\\end{{itemize}}

\\newpage
"""
    
    tex_content += r"""
\end{document}
"""
    
    with open(output_tex_path, 'w') as f:
        f.write(tex_content)
    
    print(f"✅ LaTeX document saved to: {output_tex_path}")
    print(f"📊 Plot images saved to: {plots_dir}")
    print(f"🔧 Compile with: pdflatex {output_tex_path}")

def main(csv_file_path, 
         pdf_output_name="permutation_plots.pdf",
         analyze_categorical=True,
         export_extra_analysis=False,
         extra_analysis_dir="analysis_output",
         theme="viridis",
         use_tex=False,
         export_tex=False,
         tex_output_name=None):
    """
    Enhanced main function with theme support and LaTeX export
    
    Parameters:
    - csv_file_path: Path to the CSV file
    - pdf_output_name: Name for the output PDF file
    - analyze_categorical: Whether to analyze and report categorical variables
    - export_extra_analysis: Whether to export extra analysis data per plot
    - extra_analysis_dir: Directory for extra analysis output
    - theme: Matplotlib theme name (viridis, plasma, magma, inferno, etc.)
    - use_tex: Whether to use LaTeX for text rendering in plots
    - export_tex: Whether to export as LaTeX document
    - tex_output_name: Name for LaTeX output file
    """
    
    if not os.path.exists(csv_file_path):
        print(f"❌ Error: File '{csv_file_path}' not found.")
        return
    
    print("="*60)
    print("ENHANCED PERMUTATION PLOT GENERATOR")
    print("="*60)
    print(f"🎨 Theme: {theme}")
    print(f"📊 LaTeX rendering: {use_tex}")
    print(f"📝 LaTeX export: {export_tex}")
    
    # Load data
    df = load_and_preprocess_data(csv_file_path)
    
    # Analyze categorical variables
    categorical_info = {}
    if analyze_categorical:
        print("\n🔍 Analyzing categorical variables...")
        categorical_info = analyze_categorical_variables(df)
        
        if categorical_info:
            print(f"\n📋 Found {len(categorical_info)} categorical variables:")
            for col, info in categorical_info.items():
                print(f"   - {col}: {info['total_unique']} unique values")
                
            # Save categorical mapping
            mapping_file = os.path.join(extra_analysis_dir, "categorical_mapping.json")
            os.makedirs(extra_analysis_dir, exist_ok=True)
            with open(mapping_file, 'w') as f:
                json.dump(categorical_info, f, indent=2, default=str)
            print(f"💾 Categorical mapping saved to: {mapping_file}")
        else:
            print("📊 No categorical variables found in the dataset.")
    
    # Generate plots
    if export_tex:
        if tex_output_name is None:
            base_name = os.path.splitext(pdf_output_name)[0]
            tex_output_name = f"{base_name}.tex"
        export_to_tex(df, categorical_info, tex_output_name, theme)
    else:
        print(f"\n🎨 Generating permutation plots with '{theme}' theme...")
        pairs = create_single_page_plots(df, categorical_info, pdf_output_name, theme, use_tex)
    
    # Export extra analysis data if requested
    if export_extra_analysis and not export_tex:
        print(f"\n📊 Exporting extra analysis data...")
        pairs = list(permutations(df.columns, 2))
        analysis_data = export_analysis_data(df, categorical_info, pairs, extra_analysis_dir)
    
    print(f"\n✅ Task completed successfully!")
    if not export_tex:
        print(f"📁 Output PDF: {pdf_output_name}")
    if analyze_categorical and categorical_info:
        print(f"📊 Categorical analysis: Available in {extra_analysis_dir}/")
    if export_extra_analysis and not export_tex:
        print(f"📈 Extra analysis: Available in {extra_analysis_dir}/")


# Example usage with different configurations
if __name__ == "__main__":
    import sys
    # Replace with your CSV file path
    csv_path = sys.argv[1]
    pdf_path = sys.argv[2]
    
    # Example 3: Full analysis with extra data export
    main(csv_path,
         pdf_output_name=pdf_path,
         analyze_categorical=True,
         export_extra_analysis=True,
          theme='plasma'
    )
