# Create the covariates data frame
library('simsurv')

N <- 500
covs <- data.frame(id = 1:N,
                   A = rbinom(N, 1, 0.5),
                   B = rbinom(N, 1, 0.5),
                   C = rbinom(N, 1, 0.5),
                   D = rnorm(N, mean = 10, sd = 2),
                   E = rnorm(N, mean = 0, sd = 1))

# Simulate the survival data
simdat <- simsurv(
  dist = "weibull",
  lambdas = 0.005,
  gammas = 1.25,
  betas = c(A = -0.3, B = -0.2, C = -0.1, D = 0.01), # Baseline effects
  x = covs,
  tde = c(A = 0.1, B = -0.1, C = 0.2),    # Time-dependent effects
  tdefunction = function(t) {
    ifelse(t < 25, c(A = 1, B = 0.5, C = 0.2),  # Early: A > B > C
           ifelse(t < 50, c(A = 0.2, B = 1, C = 0.5),  # Mid: B > C > A
                  c(A = 0.1, B = 0.5, C = 1)))  # Late: C > B > A
  },
  maxt = 100
)

# Make integer
simdat$eventtime <- round(simdat$eventtime)

# Merge the simulated data with the covariates
simdat <- merge(simdat, covs)

print(head(simdat))

# Save file
write.csv(simdat, file = "data/data.csv", row.names = FALSE)
