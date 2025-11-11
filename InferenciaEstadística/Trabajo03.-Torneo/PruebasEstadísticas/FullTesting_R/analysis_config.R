# Configuration file for statistical analysis

# Analysis parameters
ANALYSIS_CONFIG <- list(
  # Alpha grid parameters
  alpha_list = c(0.01, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 
                 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95),
  P = 0.5,
  k = 1,
  sample_size = 10,
  iterations = 500,
  
  # Plot settings
  plot_width = 800,
  plot_height = 600,
  
  # Significance level for single tests
  alpha_level = 0.05

  # Dataset names
  dataset1_name = "Brandon",
  dataset2_name = "Cindy"
)

# Dataset configuration
DATASETS <- list(
  primary = "test_results.csv",
  secondary = "F1_Score_cindy.xlsx",
  output_dir = "./analysis_results"
)

# Create output directory if it doesn't exist
if (!dir.exists(DATASETS$output_dir)) {
  dir.create(DATASETS$output_dir)
}
