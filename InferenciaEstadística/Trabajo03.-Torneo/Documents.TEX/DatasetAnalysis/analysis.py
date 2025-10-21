import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Load the dataset
df = pd.read_csv('shuffle_01.csv')

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
    'Start_Tech_Oscar': 'Tech Oscar Winner'
}

# 1. Target Variable Histograms
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Collection histogram
ax1.hist(df['Collection'], bins=30, alpha=0.7, edgecolor='black', linewidth=0.5)
ax1.set_xlabel(feature_labels['Collection'])
ax1.set_ylabel('Frequency')
ax1.set_title('Distribution of Box Office Collection')
ax1.grid(True, alpha=0.3)

# Format x-axis for Collection
def thousands(x, pos):
    return '%1.0fK' % (x * 1e-3)
ax1.xaxis.set_major_formatter(FuncFormatter(thousands))

# Start_Tech_Oscar histogram
oscar_counts = df['Start_Tech_Oscar'].value_counts().sort_index()
bars = ax2.bar(['No', 'Yes'], oscar_counts.values, alpha=0.7, edgecolor='black', linewidth=0.5)
ax2.set_xlabel(feature_labels['Start_Tech_Oscar'])
ax2.set_ylabel('Count')
ax2.set_title('Distribution of Tech Oscar Winners')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('target_histograms.eps', format='eps', dpi=300, bbox_inches='tight')
plt.savefig('target_histograms.png', format='png', dpi=300, bbox_inches='tight')
plt.show()

# 2. Scatter plots for Collection vs key features
collection_features = ['Budget', 'Marketing expense', 'Trailer_views', 'Num_multiplex', 
                      'Critic_rating', 'Director_rating']

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for i, feature in enumerate(collection_features):
    axes[i].scatter(df[feature], df['Collection'], alpha=0.6, s=30, edgecolors='w', linewidth=0.5)
    axes[i].set_xlabel(feature_labels[feature])
    axes[i].set_ylabel(feature_labels['Collection'])
    axes[i].set_title(f'Collection vs {feature_labels[feature]}')
    axes[i].grid(True, alpha=0.3)
    
    # Format axes for large numbers
    if feature in ['Budget', 'Marketing expense', 'Trailer_views']:
        axes[i].xaxis.set_major_formatter(FuncFormatter(thousands))
    if i >= 3:  # Bottom row
        axes[i].yaxis.set_major_formatter(FuncFormatter(thousands))

plt.tight_layout()
plt.savefig('collection_scatters.eps', format='eps', dpi=300, bbox_inches='tight')
plt.savefig('collection_scatters.png', format='png', dpi=300, bbox_inches='tight')
plt.show()

# 3. Box plots for Start_Tech_Oscar vs key features
oscar_features = ['Budget', 'Marketing expense', 'Critic_rating', 'Director_rating', 
                 'Producer_rating', 'Trailer_views']

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for i, feature in enumerate(oscar_features):
    # Create box plot
    data = [df[df['Start_Tech_Oscar'] == 0][feature], 
            df[df['Start_Tech_Oscar'] == 1][feature]]
    box_plot = axes[i].boxplot(data, labels=['No', 'Yes'], patch_artist=True)
    
    # Color the boxes
    colors = ['lightblue', 'lightcoral']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
    
    axes[i].set_xlabel(feature_labels['Start_Tech_Oscar'])
    axes[i].set_ylabel(feature_labels[feature])
    axes[i].set_title(f'{feature_labels[feature]} by Oscar Win')
    axes[i].grid(True, alpha=0.3)
    
    # Format large numbers
    if feature in ['Budget', 'Marketing expense', 'Trailer_views']:
        axes[i].yaxis.set_major_formatter(FuncFormatter(thousands))

plt.tight_layout()
plt.savefig('oscar_boxplots.eps', format='eps', dpi=300, bbox_inches='tight')
plt.savefig('oscar_boxplots.png', format='png', dpi=300, bbox_inches='tight')
plt.show()

# 4. Genre analysis for both targets
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Collection by genre
genre_collection = df.groupby('Genre')['Collection'].mean().sort_values(ascending=False)
bars1 = ax1.bar(genre_collection.index, genre_collection.values, alpha=0.7, edgecolor='black', linewidth=0.5)
ax1.set_xlabel('Genre')
ax1.set_ylabel('Average Collection')
ax1.set_title('Average Collection by Genre')
ax1.tick_params(axis='x', rotation=45)
ax1.yaxis.set_major_formatter(FuncFormatter(thousands))

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{height/1000:.0f}K', ha='center', va='bottom')

# Oscar win rate by genre
genre_oscar = df.groupby('Genre')['Start_Tech_Oscar'].mean() * 100
bars2 = ax2.bar(genre_oscar.index, genre_oscar.values, alpha=0.7, edgecolor='black', linewidth=0.5)
ax2.set_xlabel('Genre')
ax2.set_ylabel('Oscar Win Rate (%)')
ax2.set_title('Tech Oscar Win Rate by Genre')
ax2.tick_params(axis='x', rotation=45)

# Add value labels
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('genre_analysis.eps', format='eps', dpi=300, bbox_inches='tight')
plt.savefig('genre_analysis.png', format='png', dpi=300, bbox_inches='tight')
plt.show()

# 5. 3D availability analysis
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Collection by 3D
d3d_collection = df.groupby('3D_available')['Collection'].mean()
bars1 = ax1.bar(['2D', '3D'], d3d_collection.values, alpha=0.7, edgecolor='black', linewidth=0.5)
ax1.set_xlabel('3D Format')
ax1.set_ylabel('Average Collection')
ax1.set_title('Average Collection by 3D Format')
ax1.yaxis.set_major_formatter(FuncFormatter(thousands))

for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{height/1000:.0f}K', ha='center', va='bottom')

# Oscar win rate by 3D
d3d_oscar = df.groupby('3D_available')['Start_Tech_Oscar'].mean() * 100
bars2 = ax2.bar(['2D', '3D'], d3d_oscar.values, alpha=0.7, edgecolor='black', linewidth=0.5)
ax2.set_xlabel('3D Format')
ax2.set_ylabel('Oscar Win Rate (%)')
ax2.set_title('Tech Oscar Win Rate by 3D Format')

for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('threed_analysis.eps', format='eps', dpi=300, bbox_inches='tight')
plt.savefig('threed_analysis.png', format='png', dpi=300, bbox_inches='tight')
plt.show()

print("All plots generated successfully!")
print("Files created:")
print("- target_histograms.eps/.png")
print("- collection_scatters.eps/.png") 
print("- oscar_boxplots.eps/.png")
print("- genre_analysis.eps/.png")
print("- threed_analysis.eps/.png")
