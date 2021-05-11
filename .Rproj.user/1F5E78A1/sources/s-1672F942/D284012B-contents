#include <Rcpp.h>
#include "rtn.h"
#include "mvrnorm.h"
using namespace Rcpp;


// Update Z latent  
void update_Z_g(const arma::mat& Y, arma::mat& Z, const arma::mat X, 
                arma::vec& gamma, arma::vec& theta, 
                arma::vec& alpha, arma::vec& beta, 
                unsigned int J, unsigned int N){
  
  
  for(unsigned  int i = 0; i<N; ++i){
    for(unsigned  int j = 0; j<J; ++j){
      
      double s = as_scalar(exp((X.row(i))*gamma));
      double Z_mean =  (beta(j)*theta(i) - alpha(j)); // Latent trait
      
      if (Y(i,j) == 0.0){
          Z(i,j) = rnorm_trunc(Z_mean,s,R_NegInf,0.0); // truncated normal distribution
      } else  if (Y(i,j) == 1.0) {
        Z(i,j) = rnorm_trunc(Z_mean,s,0.0,R_PosInf) ; // truncated normal distribution
      } else {
        Z(i,j) = R::rnorm(Z_mean,s); // normal distribution
      }
      
    }
  }
}

// Update Z latent  
void update_Z_g2(const arma::mat& Y, arma::mat& Z, 
                arma::vec& s, arma::vec& theta, 
                arma::vec& alpha, arma::vec& beta, 
                unsigned int J, unsigned int N){
  
  
  for(unsigned  int i = 0; i<N; ++i){
    for(unsigned  int j = 0; j<J; ++j){
      
      double Z_mean =  (beta(j)*theta(i) - alpha(j)); // Latent trait
      
      if (Y(i,j) == 0.0){
        Z(i,j) = rnorm_trunc(Z_mean,s(i),R_NegInf,0.0); // truncated normal distribution
      } else  if (Y(i,j) == 1.0) {
        Z(i,j) = rnorm_trunc(Z_mean,s(i),0.0,R_PosInf) ; // truncated normal distribution
      } else {
        Z(i,j) = R::rnorm(Z_mean,s(i)); // normal distribution
      }
      
    }
  }
}

// Update theta; latent traits
void update_theta_g(arma::mat& Z,const arma::mat X, 
                    arma::vec& theta, arma::vec& gamma , 
                    arma::vec& alpha, arma::vec& beta, 
                    arma::mat& T0, arma::vec& t0,
                    unsigned  int N
){
  
  for(unsigned  int i = 0; i<N; ++i){
    
    double s = as_scalar(exp((X.row(i))*gamma));
    double ivs2 = 1/pow(s,2);
    arma::mat post_var = inv(ivs2*trans(beta) * beta + T0);
    arma::vec wj = trans(Z.row(i)) + alpha;
    arma::mat post_mean_theta = post_var*(ivs2*trans(beta) * wj + T0*t0);
    theta(i) = R::rnorm(post_mean_theta(0),post_var(0));
  }
  

}

// Update theta; latent traits
void update_theta_g2(arma::mat& Z,
                    arma::vec& theta, arma::vec& s , 
                    arma::vec& alpha, arma::vec& beta, 
                    arma::mat& T0, arma::vec& t0,
                    unsigned  int N
){
  
  for(unsigned  int i = 0; i<N; ++i){
    
    double ivs2 = 1/s(i);
    arma::mat post_var = inv(ivs2*trans(beta) * beta + T0);
    arma::vec wj = trans(Z.row(i)) + alpha;
    arma::mat post_mean_theta = post_var*(ivs2*trans(beta) * wj + T0*t0);
    theta(i) = R::rnorm(post_mean_theta(0),post_var(0));
  }
  
  
}


// Update eta; alpha & beta paramter
void update_eta_g(arma::mat& Z, 
                  arma::vec& theta, arma::vec& alpha, arma::vec& beta, 
                  arma::mat& B0, arma::vec& b0,
                  unsigned  int N, unsigned  int J
){
  
  
  arma::mat theta_star = join_rows<arma::mat>(theta,-1*ones<arma::vec>(N));
  arma::mat post_var = inv(trans(theta_star) * theta_star + B0);
  
  for (unsigned  int j=0; j<J; ++j){
    arma::vec post_mean = post_var*(trans(theta_star) * Z.col(j) + B0*b0);
    arma::mat eta_new = mvrnorm(1,post_mean,post_var);
    beta(j) = eta_new(0);
    alpha(j) = eta_new(1);
  }
  
}


// Sample Sigma 
// [[Rcpp::export]]
double gamma_conditional(const arma::mat Y, const arma::mat X,
                         arma::vec& theta, arma::vec& gamma,
                         double prior_var_gamma, 
                         arma::vec& alpha, arma::vec& beta, 
                         unsigned int J, unsigned int N, unsigned int K){
  
  double lik =0.0;
  
  // Liklihood Model
  for (unsigned int i=0; i<N; ++i){
    for (unsigned int j=0; j<J; ++j){
      double sd = as_scalar(exp((X.row(i))*gamma));
      // Rprintf(" s %f ", s);
      double lp = (theta(i) * beta(j) - alpha(j))/sd ; // linear predictor
      // Rprintf(" lp %f ", lp);
      double p = arma::normcdf(lp) ; // probability
      // Rprintf(" p %f ", p);
      if (p<0.00000000000001)  p = 0.0000000000001;
      if (p>0.99999999999999)  p = 0.9999999999999;
      double lik_contr =  Y(i,j) * log(p) + (1.0 - Y(i,j)) * log(1.0 - p); // log Lik
      if (arma::is_finite(lik_contr)) lik += lik_contr ;
    } }
  
  // Prior
  arma::mat prior_var = prior_var_gamma*eye<arma::mat>(K,K);
  double val = lik - 0.5* as_scalar(trans(gamma) *prior_var * gamma);

  
  return val;
}


// [[Rcpp::export]]
void update_gamma(const arma::mat Y,const arma::mat X, 
                  arma::vec& thetal, arma::vec& gamma, 
                  double prior_var_gamma, const double proposal_var,
                  arma::vec& alpha, arma::vec& beta, 
                  unsigned int J, unsigned int N, unsigned int K,
                  double& a){
  
  // Proposal Value
  arma::mat var_prop = proposal_var*eye<arma::mat>(K,K);
  arma::vec cand = arma::vectorise(mvrnormArma(1,gamma,var_prop));

  // Calculate acceptance ratio
  double uni = ::Rf_runif(0., 1.);

  double accept =  gamma_conditional(Y,X,thetal,cand,prior_var_gamma,alpha,beta,J,N,K) - gamma_conditional(Y,X,thetal,gamma,prior_var_gamma,alpha,beta,J,N,K);

  // Accept or reject proposal
  if (log(uni) < accept) {
    gamma = cand;
    a += 1;
  } 
  
} 

// Update Z latent
void update_Z(const arma::mat& X, arma::mat& Z, arma::vec& theta, arma::vec& alpha, arma::vec& beta){
  
  // Constants
  unsigned int K = Z.n_cols; // Number of Proposals
  unsigned int N = Z.n_rows; // Number of Respondents
  
  for(unsigned  int i = 0; i<N; ++i){
    for(unsigned  int k = 0; k<K; ++k){
      
      double Z_mean =  beta(k)*theta(i) - alpha(k); // Latent trait
      
      if (X(i,k) == 0.0){
        Z(i,k) = rnorm_trunc(Z_mean,1.0,R_NegInf,0.0); // truncated normal distribution
      } else  if (X(i,k) == 1.0) {
        Z(i,k) = rnorm_trunc(Z_mean,1.0,0.0,R_PosInf) ; // truncated normal distribution
      } else {
        Z(i,k) = R::rnorm(Z_mean,1.0); // normal distribution
      }
      
    }
  }
  
}


// Update eta; alpha & beta paramter
void update_eta(arma::mat& Z, arma::vec& theta, arma::vec& alpha, arma::vec& beta, arma::mat& B0, arma::vec& b0){
  
  unsigned int K = Z.n_cols; // Number of Proposals
  unsigned int N = Z.n_rows; // Number of Respondents
  
  arma::mat theta_star = join_rows<arma::mat>(theta,-1*ones<arma::vec>(N));
  arma::mat post_var = inv(trans(theta_star) * theta_star + B0);
  
  for (unsigned  int k=0; k<K; ++k){
    arma::vec post_mean = post_var*(trans(theta_star) * Z.col(k) + B0*b0);
    arma::mat eta_new = mvrnorm(1,post_mean,post_var);
    beta(k) = eta_new(0);
    alpha(k) = eta_new(1);
  }
  
}


// Update theta; latent traits
void update_theta(arma::mat& Z, arma::vec& theta, arma::vec& alpha, arma::vec& beta, arma::mat& T0, arma::vec& t0){
  
  unsigned int N = Z.n_rows; // Number of Respondents
  
  arma::mat post_var = inv(trans(beta) * beta + T0);
  
  for(unsigned  int i = 0; i<N; ++i){
    arma::mat post_mean_theta = post_var*(trans(beta) * (trans(Z.row(i)) + alpha) + T0*t0);
    theta(i) = R::rnorm(post_mean_theta(0),post_var(0));
  }
  
}

