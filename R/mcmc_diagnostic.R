gelman.rubin <- function(param) {
  # mcmc information
  n <- nrow(param) # number of iterations
  m <- ncol(param) # number of chains
  
  # calculate the mean of the means
  theta.bar.bar <- mean(colMeans(param))
  
  # within chain variance
  W <- mean(apply(param, 2, var))
  
  # between chain variance
  B <- n / (m - 1) * sum((colMeans(param) - theta.bar.bar) ^ 2)
  
  # variance of stationary distribution
  theta.var.hat <- (1 - 1 / n) * W + 1 / n * B
  
  # Potential Scale Reduction Factor (PSRF)
  R.hat <- sqrt(theta.var.hat / W)
  
  return(R.hat)
}