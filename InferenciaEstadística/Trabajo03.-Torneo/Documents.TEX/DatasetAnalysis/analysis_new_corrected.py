import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import os

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Load the dataset
df = pd.read_csv('shuffle_01.csv')

# Create output directory
os.makedirs('plots', exist_ok=True)

# Create formatted labels for better readability
feature_labels = {
    'Marketing expense': 'Marketing Expense',
    'Production expense': 'Production Expense', 
    'Multiplex coverage': 'Multiplex Coverage',
    'Budget': 'Budget',
    'Movie_length': 'Movie Length (min)',
    'Lead_ Actor_Rating': 'Lead Actor Rating',
    'Lead_Actress_rating': 'Lead Actress Rating',
    'Director_rating': 'Director Rating',
    'Producer_rating': 'Producer Rating',
    'Critic_rating': 'Critic Rating',
    'Trailer_views': 'Trailer Views',
    'Time_taken': 'Production Time (days)',
    'Twitter_hastags': 'Twitter Hashtags',
    'Avg_age_actors': 'Avg Actor Age',
    'Num_multiplex': 'Number of Multiplexes',
    'Collection': 'Box Office Collection',
    'Start_Tech_Oscar': 'Tech Oscar Winner',
    'Genre': 'Genre',
    '3D_available': '3D Available'
}

# Features to plot against targets (exclude targets themselves)
features_to_plot = [col for col in df.columns if col not in ['Collection', 'Start_Tech_Oscar']]

# Thousands formatter function
def thousands(x, pos):
    return '%1.0fK' % (x * 1e-3)

# Data validation
print("Data validation:")
print(f"Dataset shape: {df.shape}")
print(f"Missing values:\n{df.isnull().sum()}")
print(f"Constant columns: {df.columns[df.nunique() == 1].tolist()}")

# Handle missing values
df_clean = df.dropna()
print(f"Shape after removing missing values: {df_clean.shape}")

# Create individual plots for each feature
for feature in features_to_plot:
    print(f"Creating plots for: {feature}")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left plot: Collection vs Feature
    if df_clean[feature].dtype in ['object', 'bool']:
        # Categorical feature - use box plot
        if feature == '3D_available':
            categories = ['No', 'Yes']
            data = [df_clean[df_clean[feature] == 'NO']['Collection'], 
                    df_clean[df_clean[feature] == 'YES']['Collection']]
        elif feature == 'Genre':
            categories = df_clean[feature].unique()
            data = [df_clean[df_clean[feature] == cat]['Collection'] for cat in categories]
        else:
            categories = df_clean[feature].unique()
            data = [df_clean[df_clean[feature] == cat]['Collection'] for cat in categories]
        
        box_plot = ax1.boxplot(data, tick_labels=categories, patch_artist=True)
        # Color the boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
        
    else:
        # Numerical feature - use scatter plot
        ax1.scatter(df_clean[feature], df_clean['Collection'], alpha=0.6, s=30, edgecolors='w', linewidth=0.5)
        
        # Add trend line for numerical features - WITH ERROR HANDLING
        if len(df_clean[feature].unique()) > 10:  # Only for continuous numerical features
            # Clean the data before fitting
            valid_data = df_clean[[feature, 'Collection']].dropna()
            x_vals = valid_data[feature]
            y_vals = valid_data['Collection']
            
            # Check if we have enough data points and variance
            if (len(x_vals) > 1 and 
                len(y_vals) > 1 and 
                x_vals.std() > 0 and 
                y_vals.std() > 0):
                try:
                    z = np.polyfit(x_vals, y_vals, 1)
                    p = np.poly1d(z)
                    ax1.plot(x_vals, p(x_vals), "r--", alpha=0.8, linewidth=1.5)
                except (np.linalg.LinAlgError, ValueError, TypeError):
                    # Skip trend line if fitting fails
                    print(f"  ⚠️  Could not fit trend line for {feature}")
    
    ax1.set_xlabel(feature_labels.get(feature, feature))
    ax1.set_ylabel(feature_labels['Collection'])
    ax1.set_title(f'Collection vs {feature_labels.get(feature, feature)}')
    ax1.grid(True, alpha=0.3)
    
    # Format axes for large numbers
    if feature in ['Budget', 'Marketing expense', 'Trailer_views', 'Num_multiplex']:
        ax1.yaxis.set_major_formatter(FuncFormatter(thousands))
    if feature in ['Budget', 'Marketing expense', 'Trailer_views']:
        ax1.xaxis.set_major_formatter(FuncFormatter(thousands))
    
    # Right plot: Oscar vs Feature
    if df_clean[feature].dtype in ['object', 'bool']:
        # Categorical feature - use grouped bar plot
        if feature == '3D_available':
            categories = ['No', 'Yes']
            oscar_rates = [df_clean[(df_clean[feature] == 'NO') & (df_clean['Start_Tech_Oscar'] == 1)].shape[0] / max(df_clean[df_clean[feature] == 'NO'].shape[0], 1) * 100,
                          df_clean[(df_clean[feature] == 'YES') & (df_clean['Start_Tech_Oscar'] == 1)].shape[0] / max(df_clean[df_clean[feature] == 'YES'].shape[0], 1) * 100]
        elif feature == 'Genre':
            categories = df_clean[feature].unique()
            oscar_rates = []
            for cat in categories:
                cat_data = df_clean[df_clean[feature] == cat]
                if len(cat_data) > 0:
                    rate = cat_data['Start_Tech_Oscar'].mean() * 100
                else:
                    rate = 0
                oscar_rates.append(rate)
        else:
            categories = df_clean[feature].unique()
            oscar_rates = [df_clean[(df_clean[feature] == cat) & (df_clean['Start_Tech_Oscar'] == 1)].shape[0] / max(df_clean[df_clean[feature] == cat].shape[0], 1) * 100 
                          for cat in categories]
        
        bars = ax2.bar(categories, oscar_rates, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax2.set_ylabel('Oscar Win Rate (%)')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom')
                    
    else:
        # Numerical feature - use box plot
        data = [df_clean[df_clean['Start_Tech_Oscar'] == 0][feature], 
                df_clean[df_clean['Start_Tech_Oscar'] == 1][feature]]
        box_plot = ax2.boxplot(data, tick_labels=['No', 'Yes'], patch_artist=True)
        
        # Color the boxes
        colors = ['lightblue', 'lightcoral']
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
        
        ax2.set_ylabel(feature_labels.get(feature, feature))
    
    ax2.set_xlabel(feature_labels['Start_Tech_Oscar'])
    ax2.set_title(f'Oscar Win vs {feature_labels.get(feature, feature)}')
    ax2.grid(True, alpha=0.3)
    
    # Format axes for large numbers on right plot
    if feature in ['Budget', 'Marketing expense', 'Trailer_views'] and df_clean[feature].dtype not in ['object', 'bool']:
        ax2.yaxis.set_major_formatter(FuncFormatter(thousands))
    
    plt.tight_layout()
    
    # Save as EPS and PNG
    filename = feature.replace(' ', '_').replace('/', '_')
    plt.savefig(f'plots/{filename}.eps', format='eps', dpi=300, bbox_inches='tight')
    plt.savefig(f'plots/{filename}.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()  # Close figure to free memory
    
    print(f"  → Saved: plots/{filename}.eps")

print(f"\n✅ All plots created successfully!")
print(f"📁 Location: ./plots/")
print(f"📊 Total plots: {len(features_to_plot)}")
print(f"🎯 Format: Each file contains Collection vs Feature (left) and Oscar vs Feature (right)")
