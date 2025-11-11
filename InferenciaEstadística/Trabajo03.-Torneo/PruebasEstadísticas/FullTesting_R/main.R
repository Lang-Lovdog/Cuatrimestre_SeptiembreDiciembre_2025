source('install_packages.R')
source("statistical_analysis.R")
source("analysis_config.R")

# Use configured parameters
setwd(DATASETS$output_dir)
results <- perform_statistical_analysis(DATASETS$primary, DATASETS$secondary, ANALYSIS_CONFIG)
