// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppProgress)]]

#include <RcppArmadillo.h>
#include <omp.h>

#include <progress.hpp>
#include <progress_bar.hpp>

#include "MCMC_update_steps.h"

using namespace Rcpp;



// [[Rcpp::export]]
  Rcpp::List MCMCirt(const arma::mat X,  // Data arma::matrix
                     int burnin= 0, int gibbs = 1, int thin = 1, int chains =2, bool verbose = true // Control Gibbs Sampler
                    ) {
  
    // Constants
    unsigned int K = X.n_cols; // Number of Proposals
    unsigned int N = X.n_rows; // Number of Respondents
  
    // Storage arma::matrix
    arma::cube theta_store(gibbs/thin,N,chains,fill::zeros);
    arma::cube alpha_store(gibbs/thin,K,chains,fill::zeros);
    arma::cube beta_store(gibbs/thin,K,chains,fill::zeros);
  
    // Prior
    arma::vec b0 = zeros(2);
    arma::mat B0 = eye<arma::mat>(2,2);
    arma::vec t0 = zeros(1);
    arma::mat T0 = eye<arma::mat>(1,1);
    
    // Chains
    #ifdef _OPENMP
      if (chains > 0 )
        omp_set_num_threads( chains );
      REprintf("Number of Chains=%i\n", omp_get_max_threads());
    #endif
    
    Progress p(chains*(burnin + gibbs), verbose);
    
    // Set up Multiple Chains 
    #pragma omp parallel for num_threads(chains)
    for (int chain = 0; chain < chains; ++chain) {
      
      // Starting values
      arma::mat Z(N,K,fill::zeros);
      arma::vec alpha(K,fill::randn);
      arma::vec beta(K,fill::randn);
      arma::vec theta(N,fill::randn);
      
      // Gibbs Sampler
      int count = 0;
      
  
      for(int iter = 0; iter< burnin + gibbs; ++iter){
        
        if (! Progress::check_abort()) {
          
        // Print progress
        p.increment();
          
        // Sample Z
        update_Z(X, Z, theta, alpha, beta);
    
        // Sample (alpha_j, beta_j)
        update_eta(Z, theta, alpha, beta, B0, b0);
    
        // Sample theta
        update_theta(Z, theta, alpha, beta, T0, t0);
    
        
        // Store results
        if ((iter >= burnin) && ((iter%thin)==0)){
          theta_store.slice(chain).row(count) = trans(theta);
          alpha_store.slice(chain).row(count) = trans(alpha);
          beta_store.slice(chain).row(count) = trans(beta);
          ++count;
        }
    
      }}
      
    }
    
    // Return Draws
    return Rcpp::List::create(Rcpp::Named("theta") = theta_store,
                              Rcpp::Named("beta") =  beta_store,
                              Rcpp::Named("alpha") = alpha_store);
  }

  
// [[Rcpp::export]]
Rcpp::List MCMCirt_var( const arma::mat Y, // Data Respondents
                        arma::vec alpha, // Fixed Item Paramter alpha
                        arma::vec beta, // Fixed Item Paramter alpha
                        const arma::mat X, // Covariate
                        int burnin= 0, int gibbs = 1, int thin = 1, int chains=2, bool verbose = true, // Control Gibbs Sampler
                        double prior_var_gamma = 1, double proposal_var = 0.0001
    ) {

      // Constants
      unsigned int N = Y.n_rows; // Number of Respondents
      unsigned int J = Y.n_cols; // Number of Respondents

      // Number of Respondents Covariates
      unsigned int K = X.n_cols;

      // Storage arma::matrix
      arma::cube theta_store(gibbs/thin,N,chains,fill::zeros); // Ideal-Points Respondents
      arma::cube gamma_store(gibbs/thin,K,chains,fill::zeros);

      // Prior
      arma::vec t0 = zeros(1);
      arma::mat T0 = eye<arma::mat>(1,1);

      // Chains
#ifdef _OPENMP
      if (chains > 0 )
        omp_set_num_threads( chains );
      REprintf("Number of Chains=%i\n", omp_get_max_threads());
#endif

      // Progess Bar
      Progress p(chains*(burnin + gibbs), verbose);

      // Gibbs Sampler
      double a = 0;

      // Set up Multiple Chains
#pragma omp parallel for num_threads(chains)
      for (int chain = 0; chain < chains; ++chain) {

        // Starting values
        arma::mat Z(N,J,fill::zeros);
        arma::vec theta(N,fill::randn);
        arma::vec gamma(K,fill::randn);

        // Gibbs Sampler
        int count = 0;

        for(int iter = 0; iter< burnin + gibbs; ++iter){

          if (! Progress::check_abort()) {

            // Print progress
            p.increment();

            // Sample Z
            update_Z_g(Y,Z,X,gamma,theta, alpha, beta, J, N);

            // Sample theta
            update_theta_g(Z,X, theta, gamma, alpha, beta, T0, t0, N);

            // Sample Sigma
            update_gamma(Y,X, theta, gamma, prior_var_gamma, proposal_var, alpha, beta, J, N,K, a);

            // Store results
            if ((iter >= burnin) && ((iter%thin)==0)){
              theta_store.slice(chain).row(count) = trans(theta);
              gamma_store.slice(chain).row(count) = trans(gamma);
              ++count;
            }

          }}

      }


      // Return Draws
      return Rcpp::List::create(Rcpp::Named("theta_leg") = theta_store,
                                Rcpp::Named("theta_cit") = theta_store,
                                Rcpp::Named("gamma") = gamma_store,
                                Rcpp::Named("accept_rate") = a / (gibbs+burnin)
      );
    }

  
  
  
  // [[Rcpp::export]]
  Rcpp::List MCMCirtG(const arma::mat Yl, // Data arma::matrix Legislator
                      const arma::mat Yc, // Data arma::matrix Citzins
                      const arma::mat X, // Covariate
                      int burnin= 0, int gibbs = 1, int thin = 1, int chains=2, bool verbose = true, // Control Gibbs Sampler
                      double prior_var_gamma = 100, const double proposal_var = 0.0001
  ) {

    // Constants
    unsigned int J = Yl.n_cols; // Number of Items
    unsigned int Nl = Yl.n_rows; // Number of Legislators
    unsigned int Nc = Yc.n_rows; // Number of Citzins
    
    // Number of Citzins Covariates
    unsigned int K = X.n_cols;
    
    // Storage arma::matrix
    arma::cube theta_l_store(gibbs/thin,Nl,chains,fill::zeros); // Ideal-Points Legislators
    arma::cube theta_c_store(gibbs/thin,Nc,chains,fill::zeros); // Ideal-Points Citzins
    arma::cube alpha_store(gibbs/thin,J,chains,fill::zeros);  
    arma::cube beta_store(gibbs/thin,J,chains,fill::zeros);
    arma::cube gamma_store(gibbs/thin,K,chains,fill::zeros);
    
    // Prior
    arma::vec b0 = zeros(2);
    arma::mat B0 = eye<arma::mat>(2,2);
    arma::vec t0 = zeros(1);
    arma::mat T0 = eye<arma::mat>(1,1);
    
    // Chains
#ifdef _OPENMP
    if (chains > 0 )
      omp_set_num_threads( chains );
    REprintf("Number of Chains=%i\n", omp_get_max_threads());
#endif
    
    // Progess Bar
    Progress p(chains*(burnin + gibbs), verbose);
    
    // Gibbs Sampler
    double a = 0;
    
    // Set up Multiple Chains 
#pragma omp parallel for num_threads(chains)
    for (int chain = 0; chain < chains; ++chain) {
      
      // Starting values 
      arma::mat Zl(Nl,J,fill::zeros);
      arma::mat Zc(Nc,J,fill::zeros);
      arma::vec thetal(Nl,fill::randn);
      arma::vec thetac(Nc,fill::randn);
      arma::vec gammac(K,fill::zeros);
      arma::vec gammal(K,fill::zeros);
      arma::vec alpha(J,fill::zeros);
      arma::vec beta(J,fill::zeros);
      
      // Gibbs Sampler
      int count = 0; 
      
      for(int iter = 0; iter< burnin + gibbs; ++iter){
        
        if (! Progress::check_abort()) {
          
          // Print progress
          p.increment();
          
          
          // Legislator Model
          
          // Sample Z 
          update_Z_g(Yl,Zl,X,gammal,thetal, alpha, beta, J, Nl);
          
          // Sample alpha_j and beta_j
          update_eta_g(Zl,thetal, alpha, beta,B0, b0,Nl,J);
          
          // Sample theta 
          update_theta_g(Zl,X, thetal, gammal, alpha, beta, T0, t0, Nl);
          
          // Citzins Model
          
          // Sample Z
          update_Z_g(Yc,Zc,X,gammac,thetac, alpha, beta, J, Nc);
          
          // Sample theta 
          update_theta_g(Zc,X, thetac, gammac, alpha, beta, T0, t0, Nc);
          
          // Sample Sigma 
          update_gamma(Yc,X, thetac, gammac, prior_var_gamma, proposal_var, alpha, beta, J, Nc,K, a);
          
          // Store results
          if ((iter >= burnin) && ((iter%thin)==0)){
            theta_l_store.slice(chain).row(count) = trans(thetal);
            theta_c_store.slice(chain).row(count) = trans(thetac);
            alpha_store.slice(chain).row(count) = trans(alpha);
            beta_store.slice(chain).row(count) = trans(beta);      
            gamma_store.slice(chain).row(count) = trans(gammac);
            ++count;
          }
          
        }}

    }

    // Return Draws
    return Rcpp::List::create(Rcpp::Named("theta_leg") = theta_l_store,
                              Rcpp::Named("theta_cit") = theta_c_store,
                              Rcpp::Named("alpha") = alpha_store,
                              Rcpp::Named("beta") = beta_store,
                              Rcpp::Named("gamma") = gamma_store,
                              Rcpp::Named("accept_rate") = a / (gibbs+burnin)
    );
  }
  
  // [[Rcpp::export]]
  Rcpp::List MCMCirtG2(const arma::mat Yl, // Data arma::matrix Legislator
                      const arma::mat Yc, // Data arma::matrix Citzins
                      const arma::mat X, // Covariate
                      int burnin= 0, int gibbs = 1, int thin = 1, int chains=2, bool verbose = true, // Control Gibbs Sampler
                      double prior_var_gamma = 100, const double proposal_var = 0.0001
  ) {
    
    // Joined Data
    const arma::mat Y = arma::join_cols(Yl,Yc);
    
    // Constants
    unsigned int J = Yl.n_cols; // Number of Items
    unsigned int Nl = Yl.n_rows; // Number of Legislators
    unsigned int Nc = Yc.n_rows; // Number of Citzins
    unsigned int N = Nl + Nc; // Total Number of cases
    
    // Number of Citzins Covariates
    unsigned int K = X.n_cols;
    
    // Storage arma::matrix
    arma::cube theta_l_store(gibbs/thin,Nl,chains,fill::zeros); // Ideal-Points Legislators
    arma::cube theta_c_store(gibbs/thin,Nc,chains,fill::zeros); // Ideal-Points Citzins
    arma::cube alpha_store(gibbs/thin,J,chains,fill::zeros);  
    arma::cube beta_store(gibbs/thin,J,chains,fill::zeros);
    arma::cube gamma_store(gibbs/thin,K,chains,fill::zeros);
    
    // Prior
    arma::vec b0 = zeros(2);
    arma::mat B0 = eye<arma::mat>(2,2);
    arma::vec t0 = zeros(1);
    arma::mat T0 = eye<arma::mat>(1,1);
    
    // Chains
#ifdef _OPENMP
    if (chains > 0 )
      omp_set_num_threads( chains );
    REprintf("Number of Chains=%i\n", omp_get_max_threads());
#endif
    
    // Progess Bar
    Progress p(chains*(burnin + gibbs), verbose);
    
    // Gibbs Sampler
    double a = 0;
    
    // Set up Multiple Chains 
#pragma omp parallel for num_threads(chains)
    for (int chain = 0; chain < chains; ++chain) {
      
      // Starting values 
      arma::mat Zl(Nl,J,fill::zeros);
      arma::mat Zc(Nc,J,fill::zeros);
      arma::mat Z(N,J,fill::zeros);
      
      arma::vec thetal(Nl,fill::randn);
      arma::vec thetac(Nc,fill::randn);
      
      arma::vec theta(N,fill::randn); // Theta for all 
      
      arma::vec gammac(K,fill::zeros);
      arma::vec gammal(K,fill::zeros); // Zero and not updated => s=1 for all leg
      
      arma::vec sl(N,fill::ones); // Error variance legislators
      arma::vec sc(N,fill::ones); // Error variance citzins
      arma::vec s = arma::join_cols(sl,sc);
      
      arma::vec alpha(J,fill::zeros);
      arma::vec beta(J,fill::zeros);
      
      // Gibbs Sampler
      int count = 0; 
      
      for(int iter = 0; iter< burnin + gibbs; ++iter){
        
        if (! Progress::check_abort()) {
          
          // Print progress
          p.increment();
          
          // Legislator Model
          
          // Combine ideal points
          theta  = arma::join_cols(thetal,thetac); // Join legislators and citzins
          // Rcout << "Z legislator rows: " << Zl.n_rows << "\n";
          
          // Sample Z 
          update_Z_g2(Y,Z,s,theta, alpha, beta, J, Nl);

          // Sample alpha_j and beta_j
          Zl = Z.rows(0,Nl-1);
          // Rcout << "Z legislator rows: " << Zl.n_rows << "\n";
          update_eta_g(Zl,thetal, alpha, beta,B0, b0,Nl,J);
          
          // Sample theta 
          update_theta_g2(Z,theta, s, alpha, beta, T0, t0, Nl);
          
          thetac = theta.rows(Nl,N-1);
          thetal = theta.rows(0,Nl-1);
          
          // Sample Citzin Error Variance
          update_gamma(Yc,X, thetac, gammac, prior_var_gamma, proposal_var, alpha, beta, J, Nc,K, a);
          
          // Update error variances
          sc = exp(X * gammac);
          
          // Rcout << "sc: " << sc.rows(0,10) << "\n";
          s = arma::join_cols(sl,sc);
    
          // Store results
          if ((iter >= burnin) && ((iter%thin)==0)){
            theta_l_store.slice(chain).row(count) = trans(thetal);
            theta_c_store.slice(chain).row(count) = trans(thetac);
            alpha_store.slice(chain).row(count) = trans(alpha);
            beta_store.slice(chain).row(count) = trans(beta);      
            gamma_store.slice(chain).row(count) = trans(gammac);
            ++count;
          }
          
        }}
      
    }
    
    // Return Draws
    return Rcpp::List::create(Rcpp::Named("theta_leg") = theta_l_store,
                              Rcpp::Named("theta_cit") = theta_c_store,
                              Rcpp::Named("alpha") = alpha_store,
                              Rcpp::Named("beta") = beta_store,
                              Rcpp::Named("gamma") = gamma_store,
                              Rcpp::Named("accept_rate") = a / (gibbs+burnin)
    );
  }
  
  
  // [[Rcpp::export]]
  Rcpp::List MCMCirtG3(const arma::mat Yl, // Data arma::matrix Legislator
                       const arma::mat Yc, // Data arma::matrix Citzins
                       const arma::mat X, // Covariate
                       int burnin= 0, int gibbs = 1, int thin = 1, int chains=2, bool verbose = true, // Control Gibbs Sampler
                       double prior_var_gamma = 100, const double proposal_var = 0.0001
  ) {
    
    // Joined Data
    const arma::mat Y = arma::join_cols(Yl,Yc);
    
    // Constants
    unsigned int J = Yl.n_cols; // Number of Items
    unsigned int Nl = Yl.n_rows; // Number of Legislators
    unsigned int Nc = Yc.n_rows; // Number of Citzins
    unsigned int N = Nl + Nc; // Total Number of cases
    
    // Number of Citzins Covariates
    unsigned int K = X.n_cols;
    
    // Storage arma::matrix
    arma::cube theta_l_store(gibbs/thin,Nl,chains,fill::zeros); // Ideal-Points Legislators
    arma::cube theta_c_store(gibbs/thin,Nc,chains,fill::zeros); // Ideal-Points Citzins
    arma::cube alpha_store(gibbs/thin,J,chains,fill::zeros);  
    arma::cube beta_store(gibbs/thin,J,chains,fill::zeros);
    arma::cube gamma_store(gibbs/thin,K,chains,fill::zeros);
    
    // Prior
    arma::vec b0 = zeros(2);
    arma::mat B0 = eye<arma::mat>(2,2);
    arma::vec t0 = zeros(1);
    arma::mat T0 = eye<arma::mat>(1,1);
    
    // Chains
#ifdef _OPENMP
    if (chains > 0 )
      omp_set_num_threads( chains );
    REprintf("Number of Chains=%i\n", omp_get_max_threads());
#endif
    
    // Progess Bar
    Progress p(chains*(burnin + gibbs), verbose);
    
    // Gibbs Sampler
    double a = 0;
    
    // Set up Multiple Chains 
#pragma omp parallel for num_threads(chains)
    for (int chain = 0; chain < chains; ++chain) {
      
      // Starting values 
      arma::mat Zl(Nl,J,fill::zeros);
      arma::mat Zc(Nc,J,fill::zeros);
      arma::mat Z(N,J,fill::zeros);
      
      arma::vec thetal(Nl,fill::randn);
      arma::vec thetac(Nc,fill::randn);
      
      arma::vec theta(N,fill::randn); // Theta for all 
      
      arma::vec gammac(K,fill::zeros);
      arma::vec gammal(K,fill::zeros); // Zero and not updated => s=1 for all leg
      
      arma::vec sl(N,fill::ones); // Error variance legislators
      arma::vec sc(N,fill::ones); // Error variance citzins
      arma::vec s = arma::join_cols(sl,sc);
      
      arma::vec alpha(J,fill::zeros);
      arma::vec beta(J,fill::zeros);
      
      // Gibbs Sampler
      int count = 0; 
      
      for(int iter = 0; iter< burnin + gibbs; ++iter){
        
        if (! Progress::check_abort()) {
          
          // Print progress
          p.increment();
          
          // Legislator Model
          
          // Combine ideal points
          theta  = arma::join_cols(thetal,thetac); // Join legislators and citzins
          // Rcout << "Z legislator rows: " << Zl.n_rows << "\n";
          
          // Sample Z 
          update_Z_g2(Y,Z,s,theta, alpha, beta, J, Nl);
          
          // Sample alpha_j and beta_j
          Zl = Z.rows(0,Nl-1);
          // Rcout << "Z legislator rows: " << Zl.n_rows << "\n";
          update_eta_g(Zl,thetal, alpha, beta,B0, b0,Nl,J);
          
          // Sample theta 
          update_theta_g2(Z,theta, s, alpha, beta, T0, t0, Nl);
          
          thetac = theta.rows(Nl,N-1);
          thetal = theta.rows(0,Nl-1);
          
          // Sample Citzin Error Variance
          update_gamma(Y,X, theta, gammac, prior_var_gamma, proposal_var, alpha, beta, J, Nc,K, a);
          
          // Update error variances
          s = exp(X * gammac);
          
          // Rcout << "sc: " << sc.rows(0,10) << "\n";
          // s = arma::join_cols(sl,sc);
          
          // Store results
          if ((iter >= burnin) && ((iter%thin)==0)){
            theta_l_store.slice(chain).row(count) = trans(thetal);
            theta_c_store.slice(chain).row(count) = trans(thetac);
            alpha_store.slice(chain).row(count) = trans(alpha);
            beta_store.slice(chain).row(count) = trans(beta);      
            gamma_store.slice(chain).row(count) = trans(gammac);
            ++count;
          }
          
        }}
      
    }
    
    // Return Draws
    return Rcpp::List::create(Rcpp::Named("theta_leg") = theta_l_store,
                              Rcpp::Named("theta_cit") = theta_c_store,
                              Rcpp::Named("alpha") = alpha_store,
                              Rcpp::Named("beta") = beta_store,
                              Rcpp::Named("gamma") = gamma_store,
                              Rcpp::Named("accept_rate") = a / (gibbs+burnin)
    );
  }
  
