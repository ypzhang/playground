#include "gpusollib.h"

#define SEED 100

/*---------------------------------------------*/
void lanczos(matrix_t *mat, int msteps, 
             double *maxev, double *minev) {
/*---------------------------------------------*/
  int i,m,n,n2;
  double *h_v,*d_w,*d_u,*VV,t,beta,alpha,*d,*e,*z,*work,
  orthTol,wn;
/*---------------------------------------------*/
  srand(SEED);
  n = mat->n;
/*----- ALIGNMENT OF DEVICE MEM */
  m  = ALIGNMENT/sizeof(double);
  n2 = (n+m-1)/m*m;
/*----- random vector */
  Malloc(h_v, n, double);
  for (i=0; i<n; i++)
    h_v[i] = rand()/((double)RAND_MAX + 1);
/*----- device mem */
  VV = (double*) cuda_malloc((msteps+1)*n2*sizeof(double));
  cuda_memset(VV, 0, (msteps+1)*n2*sizeof(double));
  d_w = (double*) cuda_malloc(n*sizeof(double));
  d_u = (double*) cuda_malloc(msteps*sizeof(double));
/*------ VV[:,1] = v */
  memcpyh2d(VV, h_v, n*sizeof(double));
  t = CUNRM2(n, VV, 1);
  CUSCAL(n, 1.0/t, VV, 1);
  beta = 0.0;
  orthTol = 1.0e-8;
  wn = 0.0;
/*--------------------------------*/
/*        Tridiagonal matrix      */
/*  saved in d(diag), e(off-diag) */
/*--------------------------------*/
  Malloc(d, msteps, double);
  Malloc(e, msteps-1, double);
/*-------- main loop */
  printf("Lanczos Alg %d begins ...\n", msteps);
  for (i=0; i<msteps; i++) {
    (*(mat->spmv))(mat, &VV[i*n2], d_w, 0);
 
    if (i == 0)
       CUAXPY(n, -beta, VV, 1, d_w, 1);
     else
       CUAXPY(n, -beta, &VV[(i-1)*n2], 1, d_w, 1);

    alpha = CUDOT(n, d_w, 1, &VV[i*n2], 1);
    wn += alpha*alpha;
    d[i] = alpha;
    CUAXPY(n, -alpha, &VV[i*n2], 1, d_w, 1);
    // u = V'*w
    CUGEMV('t', n, i+1, 1.0, VV, n2, d_w, 1, 0.0, d_u, 1);
    // w = V*u - w
    CUGEMV('n', n, i+1, -1.0, VV, n2, d_u, 1, 1.0, d_w, 1);

    beta = CUDOT(n, d_w, 1, d_w, 1);

    if (beta*(i+1) < orthTol*wn)
      break;

    wn += 2.0 * beta;
    beta = sqrt(beta);

    CUAXPY(n, 1.0/beta, d_w, 1, &VV[(i+1)*n2], 1);

    if (i < msteps-1) {
      e[i] = beta;
    }
  }
  printf("  done\n");
/*--------------------------------------*
 *   compute eigenvalue of T and
 *   eigen-vec associated with largest 
 *   eigen-val of TT by LAPACK
 *--------------------------------------*/
  printf("Lapack STEQR begins ...\n");
  Malloc(work, 2*msteps-2, double);
  Calloc(z, msteps*msteps, double);
  int info;
  char compz = 'I';
  STEQR(&compz, &msteps, d, e, z, &msteps, work, &info);
  if (info != 0) {
    printf("LAPACK: FAILED TO FIND EIGENVALUES !!!\n");
    exit(-1);
  }
  printf("  done\n");
  *maxev = d[msteps-1];
  *minev = d[0];
/*------- add safeguard term */
  double sg = fabs(beta*z[msteps*msteps-1]);
  *maxev += sg;
/*
  printf("eigen-value:[%e, %e]  zn = %e\n",
  *minev, *maxev, sg);
*/
/*------- done, free */
  free(h_v);
  cuda_free(VV);
  cuda_free(d_w);
  cuda_free(d_u);
  free(d);
  free(e);
  free(work);
  free(z);
}

