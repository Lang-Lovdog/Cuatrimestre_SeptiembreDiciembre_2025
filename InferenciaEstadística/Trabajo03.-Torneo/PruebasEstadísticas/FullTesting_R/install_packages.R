# robust_install.R
# Enhanced package installation with error handling
options(repos=c(CRAN="https://cran.r-project.org"))

required_packages <- c("car", "nortest", "dplyr")

install_packages_safely <- function(packages) {
  for (pkg in packages) {
    tryCatch({
      if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
        cat("Installing:", pkg, "...")
        install.packages(pkg, dependencies = TRUE, quiet = TRUE)
        
        # Verify installation
        if (require(pkg, character.only = TRUE, quietly = TRUE)) {
          cat(" OK\n")
        } else {
          cat(" FAILED\n")
          warning(paste("Failed to install package:", pkg))
        }
      } else {
        cat(pkg, "already installed\n")
      }
    }, error = function(e) {
      cat("Error installing", pkg, ":", e$message, "\n")
    })
  }
}

# Check if packages are available from CRAN
check_cran_availability <- function(packages) {
  cat("Checking package availability on CRAN...\n")
  available <- available.packages()
  for (pkg in packages) {
    if (pkg %in% rownames(available)) {
      cat(pkg, ": Available\n")
    } else {
      cat(pkg, ": NOT available on CRAN\n")
    }
  }
}

# Main installation process
cat("Statistical Analysis Package Installation\n")
cat("=========================================\n\n")

# Check availability
check_cran_availability(required_packages)

cat("\nStarting installation...\n")
install_packages_safely(required_packages)

# Final verification
cat("\nVerifying installations...\n")
for (pkg in required_packages) {
  if (require(pkg, character.only = TRUE, quietly = TRUE)) {
    cat("✓", pkg, "loaded successfully\n")
  } else {
    cat("✗", pkg, "failed to load\n")
  }
}

cat("\nInstallation complete!\n")
