library(car)
library(nortest)
library(dplyr)

# Function 1: Data Overview
data_overview <- function(data, data_name = "Dataset") {
  cat("\n", strrep("=", 50), "\n")
  cat("DATA OVERVIEW:", data_name, "\n")
  cat(strrep("=", 50), "\n")
  
  basic_stats <- list(
    n = length(data),
    mean = mean(data),
    median = median(data),
    sd = sd(data),
    variance = var(data),
    min = min(data),
    max = max(data),
    range = max(data) - min(data),
    iqr = IQR(data)
  )
  
  # Print basic statistics
  cat("Basic Statistics:\n")
  cat(sprintf("  Sample size (n): %d\n", basic_stats$n))
  cat(sprintf("  Mean: %.4f\n", basic_stats$mean))
  cat(sprintf("  Median: %.4f\n", basic_stats$median))
  cat(sprintf("  Standard Deviation: %.4f\n", basic_stats$sd))
  cat(sprintf("  Variance: %.4f\n", basic_stats$variance))
  cat(sprintf("  Minimum: %.4f\n", basic_stats$min))
  cat(sprintf("  Maximum: %.4f\n", basic_stats$max))
  cat(sprintf("  Range: %.4f\n", basic_stats$range))
  cat(sprintf("  IQR: %.4f\n", basic_stats$iqr))
  
  return(basic_stats)
}

# Function 2: Normality Test
normality_test <- function(data, data_name = "Dataset") {
  cat("\n", strrep("=", 50), "\n")
  cat("NORMALITY TEST:", data_name, "\n")
  cat(strrep("=", 50), "\n")
  
  # Perform normality tests
  shapiro_test <- shapiro.test(data)
  ad_test <- ad.test(data)
  
  # Determine normality
  is_normal <- shapiro_test$p.value > 0.05
  
  # Print results
  cat("Normality Test Results:\n")
  cat(sprintf("  Shapiro-Wilk Test:\n"))
  cat(sprintf("    W = %.4f, p-value = %.6f\n", 
              shapiro_test$statistic, shapiro_test$p.value))
  cat(sprintf("  Anderson-Darling Test:\n"))
  cat(sprintf("    A = %.4f, p-value = %.6f\n", 
              ad_test$statistic, ad_test$p.value))
  cat(sprintf("  Normally distributed: %s\n", 
              ifelse(is_normal, "YES", "NO")))
  
  return(list(
    shapiro = shapiro_test,
    anderson_darling = ad_test,
    is_normal = is_normal
  ))
}

# Function 3: Variance Test (for two datasets)
variance_test <- function(data1, data2, data1_name = "Dataset1", data2_name = "Dataset2") {
  cat("\n", strrep("=", 50), "\n")
  cat("VARIANCE TEST\n")
  cat(strrep("=", 50), "\n")
  
  # F-test for equal variance
  f_test <- var.test(data1, data2)
  equal_variance <- f_test$p.value > 0.05
  
  # Print results
  cat("Variance Test Results:\n")
  cat(sprintf("  F-test for equal variances:\n"))
  cat(sprintf("    F = %.4f, p-value = %.6f\n", 
              f_test$statistic, f_test$p.value))
  cat(sprintf("    Equal variance: %s\n", 
              ifelse(equal_variance, "YES", "NO")))
  
  return(list(
    f_test = f_test,
    equal_variance = equal_variance
  ))
}

# Function 4: Alpha Grid Optimization
alpha_grid_optimization <- function(data1, data2, config = NULL) {
  # Use configuration if provided, otherwise use defaults
  if (!is.null(config)) {
    alpha_list <- config$alpha_list
    P <- config$P
    k <- config$k
    sample_size <- config$sample_size
    iterations <- config$iterations
  } else {
    # Default values
    alpha_list <- c(0.01, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 
                   0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)
    P <- 0.4
    k <- 1
    sample_size <- 6
    iterations <- 500
  }

  cat("\n", strrep("=", 50), "\n")
  cat("ALPHA GRID OPTIMIZATION\n")
  cat(strrep("=", 50), "\n")
  
  beta_list <- c()
  power_list <- c()
  expected_losses_list <- c()
  
  for (alpha in alpha_list) {
    set.seed(23)
    rejection_count <- c()
    
    for (i in 1:iterations) {
      sample_x <- sample(data1, size = sample_size, replace = TRUE)
      sample_y <- sample(data2, size = sample_size, replace = TRUE)
      pvalue <- t.test(x = sample_x, y = sample_y)$p.value
      rejection_count <- append(rejection_count, ifelse(pvalue < alpha, 1, 0))
    }
    
    power <- sum(rejection_count) / length(rejection_count)
    beta <- 1 - power
    expected_losses <- (P * alpha) + ((1 - P) * k * beta)
    
    power_list <- append(power_list, power)
    beta_list <- append(beta_list, beta)
    expected_losses_list <- append(expected_losses_list, expected_losses)
  }
  
  solutions <- data.frame(
    alpha = alpha_list,
    power = power_list,
    beta = beta_list,
    expected_loss = expected_losses_list
  )
  
  optimal_solution <- solutions %>% filter(expected_loss == min(expected_loss))
  
  # Print results
  cat("Optimization Parameters:\n")
  cat(sprintf("  P: %.1f, k: %.1f, Sample size: %d, Iterations: %d\n", 
              P, k, sample_size, iterations))
  cat("\nOptimal Solution:\n")
  cat(sprintf("  Best alpha: %.3f\n", optimal_solution$alpha))
  cat(sprintf("  Power: %.4f\n", optimal_solution$power))
  cat(sprintf("  Beta: %.4f\n", optimal_solution$beta))
  cat(sprintf("  Expected loss: %.4f\n", optimal_solution$expected_loss))
  
  return(list(
    solutions = solutions,
    optimal = optimal_solution,
    params = list(P = P, k = k, sample_size = sample_size, iterations = iterations)
  ))
}

# Function 5: Difference Significance Test
difference_significance_test <- function(data1, data2, data1_name = "Dataset1", data2_name = "Dataset2", config = NULL) {
  cat("\n", strrep("=", 50), "\n")
  cat("DIFFERENCE SIGNIFICANCE TEST\n")
  cat(strrep("=", 50), "\n")
  
  # Check normality for both datasets
  norm1 <- normality_test(data1, data1_name)$is_normal
  norm2 <- normality_test(data2, data2_name)$is_normal
  
  test_results <- list()
  
  if (norm1 && norm2) {
    # Both normal - check variance
    var_test <- variance_test(data1, data2, data1_name, data2_name)
    
    if (var_test$equal_variance) {
      # Equal variance - use alpha grid optimization for T-test
      alpha_opt <- alpha_grid_optimization(data1, data2, config)
      optimal_alpha <- alpha_opt$optimal$alpha
      
      # Perform T-test with optimized alpha
      t_test <- t.test(data1, data2, var.equal = TRUE, conf.level = 1 - optimal_alpha)
      
      test_results <- list(
        test_type = "Student's t-test (equal variance)",
        test_result = t_test,
        alpha_used = optimal_alpha,
        significant = t_test$p.value < optimal_alpha,
        alpha_optimization = alpha_opt
      )
      
    } else {
      # Unequal variance - Welch T-test with default alpha
      welch_test <- t.test(data1, data2, var.equal = FALSE, conf.level = 0.95)
      
      test_results <- list(
        test_type = "Welch t-test (unequal variance)",
        test_result = welch_test,
        alpha_used = 0.05,
        significant = welch_test$p.value < 0.05
      )
    }
    
  } else {
    # At least one non-normal - Wilcoxon test
    wilcox_test <- wilcox.test(data1, data2, conf.int = TRUE, conf.level = 0.95)
    
    test_results <- list(
      test_type = "Wilcoxon rank sum test",
      test_result = wilcox_test,
      alpha_used = 0.05,
      significant = wilcox_test$p.value < 0.05
    )
  }
  
  # Print test results
  cat("Test Results:\n")
  cat(sprintf("  Test used: %s\n", test_results$test_type))
  cat(sprintf("  Alpha level: %.3f\n", test_results$alpha_used))
  
  if (test_results$test_type %in% c("Student's t-test (equal variance)", "Welch t-test (unequal variance)")) {
    cat(sprintf("  t-statistic: %.4f\n", test_results$test_result$statistic))
    cat(sprintf("  Degrees of freedom: %.1f\n", test_results$test_result$parameter))
    cat(sprintf("  p-value: %.6f\n", test_results$test_result$p.value))
    cat(sprintf("  95%% CI: [%.4f, %.4f]\n", 
                test_results$test_result$conf.int[1], 
                test_results$test_result$conf.int[2]))
  } else {
    cat(sprintf("  W-statistic: %.4f\n", test_results$test_result$statistic))
    cat(sprintf("  p-value: %.6f\n", test_results$test_result$p.value))
  }
  
  cat(sprintf("  Statistically significant: %s\n", 
              ifelse(test_results$significant, "YES", "NO")))
  
  return(test_results)
}

# Plotting functions
generate_distribution_plot <- function(data, data_name) {
  png(paste0(data_name, "_distribution.png"), width = 800, height = 600)
  par(mfrow = c(1, 2))
  
  # Histogram with density
  hist(data, breaks = 30, col = "lightblue", 
       main = paste("Distribution of", data_name),
       xlab = "Values", probability = TRUE)
  lines(density(data), col = "red", lwd = 2)
  abline(v = mean(data), col = "blue", lwd = 2, lty = 2)
  
  # Boxplot
  boxplot(data, main = paste("Boxplot of", data_name),
          ylab = "Values", col = "lightgreen")
  
  dev.off()
}

generate_qq_plot <- function(data, data_name) {
  png(paste0(data_name, "_qqplot.png"), width = 600, height = 600)
  qqnorm(data, main = paste("QQ-Plot of", data_name))
  qqline(data, col = "red")
  dev.off()
}

# Function 6: Main analysis function
perform_statistical_analysis <- function(csv_file_path, two_sample_file = NULL, config = NULL) {
  
  # Read the main CSV file
  if (!file.exists(csv_file_path)) {
    stop("CSV file not found: ", csv_file_path)
  }
  
  data <- read.csv(csv_file_path)
  
  # Check if f1_score column exists
  if (!"f1_score" %in% names(data)) {
    stop("Column 'f1_score' not found in the CSV file")
  }
  
  f1_scores <- data$f1_score
  
  cat("F1 SCORES STATISTICAL ANALYSIS\n")
  cat("===============================\n")
  cat("File:", csv_file_path, "\n")
  cat("Sample size:", length(f1_scores), "\n\n")
  
  # Step 1: Data Overview
  overview <- data_overview(f1_scores, "F1_Scores")
  
  # Generate plots
  generate_distribution_plot(f1_scores, "F1_Scores")
  generate_qq_plot(f1_scores, "F1_Scores")
  
  if (is.null(two_sample_file)) {
    # Single dataset - only normality test
    cat("\n", strrep("=", 50), "\n")
    cat("SINGLE DATASET ANALYSIS - NORMALITY CHECK ONLY\n")
    cat(strrep("=", 50), "\n")
    
    # Step 2: Normality Test
    normality <- normality_test(f1_scores, "F1_Scores")
    
    # Generate report for single dataset
    results <- generate_single_dataset_report(overview, normality)
    
  } else {
    # Two datasets - full validation process
    if (!file.exists(two_sample_file)) {
      stop("Second CSV file not found: ", two_sample_file)
    }
    
    data2 <- read.csv(two_sample_file)
    
    # Check if f1_score column exists in second file
    if (!"f1_score" %in% names(data2)) {
      stop("Column 'f1_score' not found in the second CSV file")
    }
    
    f1_scores2 <- data2$f1_score
    
    cat("\n", strrep("=", 50), "\n")
    cat("TWO DATASETS ANALYSIS - FULL VALIDATION\n")
    cat(strrep("=", 50), "\n")
    
    # Overview for both datasets
    overview1 <- data_overview(f1_scores, "Dataset1_F1_Scores")
    overview2 <- data_overview(f1_scores2, "Dataset2_F1_Scores")
    
    # Generate plots for second dataset
    generate_distribution_plot(f1_scores2, "Dataset2_F1_Scores")
    generate_qq_plot(f1_scores2, "Dataset2_F1_Scores")
    
    # Step 2: Normality Tests
    normality1 <- normality_test(f1_scores, "Dataset1_F1_Scores")
    normality2 <- normality_test(f1_scores2, "Dataset2_F1_Scores")
    
    # Step 3: Variance Test
    variance <- variance_test(f1_scores, f1_scores2, "Dataset1", "Dataset2")
    
    # Step 4: Difference Significance Test
    diff_test <- difference_significance_test(f1_scores, f1_scores2, config$dataset1_name, config$dataset2_name, config)
    
    # Generate comprehensive report
    results <- generate_two_dataset_report(overview1, overview2, normality1, normality2, variance, diff_test)
    
    # Export alpha optimization data if available
    if (!is.null(diff_test$alpha_optimization)) {
      write.csv(diff_test$alpha_optimization$solutions, "alpha_optimization_grid.csv", row.names = FALSE)
      write.csv(diff_test$alpha_optimization$optimal, "alpha_optimization_optimal.csv", row.names = FALSE)
    }
  }
  
  cat("\n", strrep("=", 50), "\n")
  cat("ANALYSIS COMPLETE!\n")
  cat(strrep("=", 50), "\n")
  
  return(results)
}

# Function to generate single dataset report
generate_single_dataset_report <- function(overview, normality) {
  sink("analysis_report.txt")
  
  cat("SINGLE DATASET ANALYSIS REPORT\n")
  cat("==============================\n\n")
  
  cat("DATA OVERVIEW:\n")
  cat("-------------\n")
  cat(sprintf("Sample size: %d\n", overview$n))
  cat(sprintf("Mean: %.4f\n", overview$mean))
  cat(sprintf("Median: %.4f\n", overview$median))
  cat(sprintf("Standard Deviation: %.4f\n", overview$sd))
  cat(sprintf("Variance: %.4f\n", overview$variance))
  cat(sprintf("Range: [%.4f, %.4f]\n", overview$min, overview$max))
  cat(sprintf("IQR: %.4f\n", overview$iqr))
  
  cat("\nNORMALITY ASSESSMENT:\n")
  cat("--------------------\n")
  cat(sprintf("Shapiro-Wilk Test: W = %.4f, p = %.6f\n", 
              normality$shapiro$statistic, normality$shapiro$p.value))
  cat(sprintf("Anderson-Darling Test: A = %.4f, p = %.6f\n", 
              normality$anderson_darling$statistic, normality$anderson_darling$p.value))
  cat(sprintf("Normally Distributed: %s\n", 
              ifelse(normality$is_normal, "YES", "NO")))
  
  cat("\nCONCLUSION:\n")
  cat("-----------\n")
  if (normality$is_normal) {
    cat("The dataset appears to be normally distributed based on statistical tests.\n")
  } else {
    cat("The dataset does not appear to be normally distributed based on statistical tests.\n")
  }
  
  sink()
  
  return(list(
    overview = overview,
    normality = normality
  ))
}

# Function to generate two dataset report
generate_two_dataset_report <- function(overview1, overview2, normality1, normality2, variance, diff_test) {
  sink("analysis_report.txt")
  
  cat("TWO DATASETS COMPARISON REPORT\n")
  cat("==============================\n\n")
  
  # Dataset 1 information
  cat("DATASET 1:\n")
  cat("----------\n")
  cat(sprintf("Sample size: %d\n", overview1$n))
  cat(sprintf("Mean: %.4f\n", overview1$mean))
  cat(sprintf("Variance: %.4f\n", overview1$variance))
  cat(sprintf("Normally distributed: %s\n", 
              ifelse(normality1$is_normal, "YES", "NO")))
  cat(sprintf("Shapiro-Wilk p-value: %.6f\n", normality1$shapiro$p.value))
  cat("\n")
  
  # Dataset 2 information
  cat("DATASET 2:\n")
  cat("----------\n")
  cat(sprintf("Sample size: %d\n", overview2$n))
  cat(sprintf("Mean: %.4f\n", overview2$mean))
  cat(sprintf("Variance: %.4f\n", overview2$variance))
  cat(sprintf("Normally distributed: %s\n", 
              ifelse(normality2$is_normal, "YES", "NO")))
  cat(sprintf("Shapiro-Wilk p-value: %.6f\n", normality2$shapiro$p.value))
  cat("\n")
  
  # Variance test results
  cat("VARIANCE COMPARISON:\n")
  cat("--------------------\n")
  cat(sprintf("F-test for equal variances: F = %.4f, p = %.6f\n", 
              variance$f_test$statistic, variance$f_test$p.value))
  cat(sprintf("Equal variance: %s\n", 
              ifelse(variance$equal_variance, "YES", "NO")))
  cat("\n")
  
  # Difference test results
  cat("DIFFERENCE SIGNIFICANCE TEST:\n")
  cat("------------------------------\n")
  cat(sprintf("Test used: %s\n", diff_test$test_type))
  cat(sprintf("Alpha level: %.3f\n", diff_test$alpha_used))
  
  if (diff_test$test_type %in% c("Student's t-test (equal variance)", "Welch t-test (unequal variance)")) {
    cat(sprintf("t-statistic: %.4f\n", diff_test$test_result$statistic))
    cat(sprintf("Degrees of freedom: %.1f\n", diff_test$test_result$parameter))
    cat(sprintf("p-value: %.6f\n", diff_test$test_result$p.value))
    cat(sprintf("95%% CI: [%.4f, %.4f]\n", 
                diff_test$test_result$conf.int[1], 
                diff_test$test_result$conf.int[2]))
  } else {
    cat(sprintf("W-statistic: %.4f\n", diff_test$test_result$statistic))
    cat(sprintf("p-value: %.6f\n", diff_test$test_result$p.value))
  }
  
  cat(sprintf("Statistically significant difference: %s\n", 
              ifelse(diff_test$significant, "YES", "NO")))
  
  # Alpha optimization results if available
  if (!is.null(diff_test$alpha_optimization)) {
    cat("\nALPHA OPTIMIZATION DETAILS:\n")
    cat("---------------------------\n")
    cat(sprintf("Optimal alpha: %.3f\n", diff_test$alpha_optimization$optimal$alpha))
    cat(sprintf("Statistical power: %.4f\n", diff_test$alpha_optimization$optimal$power))
    cat(sprintf("Type II error rate (beta): %.4f\n", diff_test$alpha_optimization$optimal$beta))
    cat(sprintf("Expected loss: %.4f\n", diff_test$alpha_optimization$optimal$expected_loss))
  }
  
  cat("\nCONCLUSION:\n")
  cat("-----------\n")
  if (diff_test$significant) {
    cat("There is a statistically significant difference between the two datasets.\n")
  } else {
    cat("There is no statistically significant difference between the two datasets.\n")
  }
  
  sink()
  
  return(list(
    overview1 = overview1,
    overview2 = overview2,
    normality1 = normality1,
    normality2 = normality2,
    variance = variance,
    diff_test = diff_test
  ))
}
