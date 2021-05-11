\name{MCMCirt}
\alias{MCMCirt}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
 MCMC Algorithms for Roll Call Vote Data
}
\description{
 }
\usage{
MCMCirt(X, burnin = 0L, gibbs = 1L, thin = 1L, chains = 2L, verbose = TRUE)
}
 \arguments{
  \item{X}{
 }
  \item{burnin}{
 }
  \item{gibbs}{
 }
  \item{thin}{
 }
  \item{chains}{
 }
  \item{verbose}{
 }
}
\details{
 }
\value{
 
}
\references{
 }
\author{
Lukas F. Stoetzer 
}
\note{

}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{

}
\examples{
load("../../in/cces2008_rollvote.Rdata")
X <- as.matrix(na.omit(hou110[,grep("rv",colnames(hou110))]))
r <- MCMCirt(X,gibbs = 10000,burnin = 500, thin=1, chains=1)

}
