# Create the covariates data frame
library('simsurv')

# Define N, maxt
N <- 100
maxt <- 100

# Make covariates
covs <- data.frame(id = 1:N,
                   x1 = rbinom(N, 1, 0.5),
                   x2 = rbinom(N, 1, 0.5),
                   x3 = rbinom(N, 1, 0.5),
                   x4 = rnorm(N, mean = 10, sd = 2),
                   x5 = rnorm(N, mean = 0, sd = 1))

# Make time-independent effects datasets
# Define the distributions and the parameters
dists <- c("exp", "weibull", "gompertz")
lambdas <- 0.005
gammas <- 1.25
betas <- c(x1 = 0.8, x2 = 0.5, x3 = 0.25, x4 = 0.1)  # Baseline effects

# Loop through each distribution
for (dist in dists) {
  if (dist == "exp") {
    simdat <- simsurv(
      dist = dist,
      lambdas = lambdas,
      betas = betas,
      x = covs,
      maxt = maxt
    )
  } else {
    simdat <- simsurv(
      dist = dist,
      lambdas = lambdas,
      gammas = gammas,
      betas = betas,
      x = covs,
      maxt = maxt
    )
  }

  # Round event times
  simdat$eventtime <- round(simdat$eventtime)

  # Merge with covariates
  simdat <- merge(simdat, covs)

  # Create file name
  file_name <- paste0("data/", dist, "_time_indep.csv")
 
  # Save to CSV
  write.csv(simdat, file = file_name, row.names = FALSE)
}

# Make time-dependent effects datasets
tdefunction <- function(t) {
  if (t < 25) {
    c(x1 = 1, x2 = 0.5, x3 = 0.2, x4 = 0.1)  # Early: x1 > x2 > x3 > x4
  } else if (t < 50) {
    c(x1 = 0.5, x2 = 1, x3 = 0.75, x4 = 0.25)  # Mid: x2 > x3 > x4 > x1
  } else if (t < 75) {
    c(x1 = 0.25, x2 = 0.75, x3 = 1, x4 = 0.5)  # Mid-late: x2 > x3 > x4 > x1
  } else {
    c(x1 = 0.1, x2 = 0.25, x3 = 0.5, x4 = 1)  # Late: x4 > x3 > x2 > x1
  }
}

# Loop through each distribution
for (dist in dists) {
  if (dist == "exp") {
    simdat <- simsurv(
      dist = dist,
      lambdas = lambdas,
      betas = betas,
      x = covs,
      tde = tde,
      tdefunction = tdefunction,
      maxt = maxt
    )
  } else {
    simdat <- simsurv(
      dist = dist,
      lambdas = lambdas,
      gammas = gammas,
      betas = betas,
      x = covs,
      tde = tde,
      tdefunction = tdefunction,
      maxt = maxt
    )
  }

  # Round event times
  simdat$eventtime <- round(simdat$eventtime)

  # Merge with covariates
  simdat <- merge(simdat, covs)

  # Create file name
  file_name <- paste0("data/", dist, "_time_dep.csv")

  # Save to CSV
  write.csv(simdat, file = file_name, row.names = FALSE)
}