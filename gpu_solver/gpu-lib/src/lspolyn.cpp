#include "gpusollib.h"

/*-------------------------------------------*
 *        L-S POLY Precon Operation
 *-------------------------------------------*/
void polyapprox(matrix_t *mat, precon_t *prec,
options_t *opts, double *d_x, double *d_b) {
/*-------------------------------------*/
  int n,k,deg;
  double *alpha,*beta,*gamma,*v1,*v0,*v;
  lspoly_t *lspol;
/*-------------------------------------*/
  lspol = prec->lspoly;
  n = mat->n;
  deg = opts->lspoly_opt->deg;
  alpha = lspol->alpha;
  beta  = lspol->beta;
  gamma = lspol->gamma;
  v1 = lspol->d_v1;
  v0 = lspol->d_v0;
  v  = lspol->d_v;
/*-----------------------------------*/
  // d_x = M^{-1} * d_b
  // assuume x0 = 0
  // r0 = b - A * x0 = b
  // v1 = r0 / beta(1) = b / beta(1)
  cublasDcopy(n, d_b, 1, v1, 1);
  cublasDscal(n, 1.0/beta[0], v1, 1);

  // x  = x0 + gamma(1)*v1 = gamm(1)*v1;
  cuda_memset(d_x, 0, n*sizeof(double));
  cublasDaxpy(n, gamma[0], v1, 1, d_x, 1);

  //v0 = zeros(n,1);
  cuda_memset(v0, 0, n*sizeof(double));

  double bet = 0.0;

  for (k=0; k<deg; k++) {
    //v = A*v1;
    (*(mat->spmv))(mat, v1, v, 0);

    //v  = v - alpha(k)*v1 - bet*v0; 
    cublasDaxpy(n, -alpha[k], v1, 1, v, 1);

    if (k>0) cublasDaxpy(n, -bet, v0, 1, v, 1);

    //v0 = v1;
    cublasDcopy(n, v1, 1, v0, 1);

    //v1 = v/beta(k+1);
    cuda_memset(v1, 0, n*sizeof(double));
    cublasDaxpy(n, 1.0/beta[k+1], v, 1, v1, 1);

    //x = x + gamma(k+1)*v1;
    cublasDaxpy(n, gamma[k+1], v1, 1, d_x, 1);

    bet = beta[k+1];
  }
}

/*--------------------------------------------*/
void SetupLSPOLY(matrix_t *mat, precon_t *prec,
options_t *opts) {
/*---------------------*/
  int deg, nlan,n;
  double maxev, minev;
/*---------------------------------*/
  n = mat->n;
  deg = opts->lspoly_opt->deg;
  nlan = opts->lspoly_opt->nlan;
  Calloc(prec->lspoly, 1, lspoly_t);
/*---------------------------------*/
  Malloc(prec->lspoly->alpha, deg, double);
  Malloc(prec->lspoly->beta, deg+1, double);
  Malloc(prec->lspoly->gamma, deg+1, double);
  prec->lspoly->d_v1 = 
  (double*) cuda_malloc(n*sizeof(double));
  prec->lspoly->d_v0 = 
  (double*) cuda_malloc(n*sizeof(double));
  prec->lspoly->d_v = 
  (double*) cuda_malloc(n*sizeof(double));
/*-------------------------------------*/
/*  Lanczos alg for eig val estimation */
/*-------------------------------------*/
  lanczos(mat, nlan, &maxev, &minev);
  //printf("min-max-eigv=%e %e\n", minev, maxev);
/*---------------------------------*/
/* Stieltjes Procedure to compute  */
/* coeff of L-S Polynomial         */
/*---------------------------------*/
  double intv[2];
  intv[0] = minev;
  intv[1] = maxev;
  lspol(deg, 1, intv, prec->lspoly->alpha, 
  prec->lspoly->beta,prec->lspoly->gamma);
}

