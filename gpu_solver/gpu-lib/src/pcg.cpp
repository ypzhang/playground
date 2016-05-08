#include "gpusollib.h"

int pcg(matrix_t *mat, precon_t *prec,
options_t *opts, double *d_x, double *d_b) {
/*----------------------------------------------*/
  int n, its, maxits;
  double tol,*d_r,*d_z,*d_p,*d_ap,
  ro1,ro,tol1,alp,bet;
/*----------------------------------------*/
  n = mat->n;
  maxits = opts->maxits;  tol = opts->tol;
/*----------------------------------------*/
  printf("begin PCG ... \n");
  d_r = (double*) cuda_malloc(n*sizeof(double));
  d_z = (double*) cuda_malloc(n*sizeof(double));
  d_p = (double*) cuda_malloc(n*sizeof(double));
  d_ap= (double*) cuda_malloc(n*sizeof(double));
/*--------- r = b - Ax */
  (*(mat->spmv))(mat, d_x, d_r, 1);
  CUAXPY(n, 1.0, d_b, 1, d_r, 1);
/*--------- z = M^{-1}*r */
  (*(prec->op))(mat, prec, opts, d_z, d_r);
/*--------- p = z; */
  memcpyd2d(d_p, d_z, n*sizeof(double));
/*--------- ro1 = z'*r */
  ro1 = CUDOT(n, d_z, 1, d_r, 1);
  double res = CUDOT(n, d_r, 1, d_r, 1);
  tol1 = tol*tol*res;
  its = 0;
  //printf( "init res = %E ... ", sqrt(res) );
  opts->result.rnorm0 = sqrt(res);
/*----------------------------------*/
  while (its < maxits && res > tol1) {
    its ++;
    ro = ro1;
/*------- ap = A * p */
    (*(mat->spmv))(mat, d_p, d_ap, 0);
    alp = ro / CUDOT(n, d_ap, 1, d_p, 1);
/*------- x = x + alp * p */
    CUAXPY(n, alp, d_p, 1, d_x, 1);
/*------- r = r - alp * ap */
    CUAXPY(n, -alp, d_ap, 1, d_r, 1);
    res = CUDOT(n,d_r,1,d_r,1);
/*------- z = M^{-1}*r */
    (*(prec->op))(mat, prec, opts, d_z, d_r);
/*------- ro1 = z' *r */
    ro1 = CUDOT(n, d_z, 1, d_r, 1);
    bet = ro1/ro;
/*------- p = z + bet *p */
    CUAXPY(n, bet, d_p, 1, d_z, 1);
    memcpyd2d(d_p, d_z, n*sizeof(double));
  }
/*-----------------------------*/
  opts->result.niters = its;
  //printf( "ro1=%E\n", sqrt(res));
/*----------------*/
  cuda_free(d_r);
  cuda_free(d_z);
  cuda_free(d_p);
  cuda_free(d_ap);
  return its;
}

