#include "gpusollib.h"

void Spmatvec_csr2(int n, int nrow, int *d_ia, int *d_ja, 
                  double *d_a, double *d_y, double alpha, 
                  double *, double*);

/*-----------------------------------------------*
 *       y[i] += d[i] * x[i]
 *-----------------------------------------------*/
__global__
void scal_k(int n, double *d, double *x, double *y) {
  int nt  = gridDim.x * blockDim.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int i;
  for (i=idx; i<n; i+=nt)
    y[i] += d[i]*x[i];
}

/*-----------------------------------------------*
 *       y[i] = d[i] * x[i]
 *-----------------------------------------------*/
__global__
void scal_k2(int n, double *d, double *x, double *y) {
  int nt  = gridDim.x * blockDim.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int i;
  for (i=idx; i<n; i+=nt)
    y[i] = d[i]*x[i];
}

/*--------------------------------------------*
             SSOR iters (omega == 1)
 *--------------------------------------------*/
void ssor(matrix_t *mat, precon_t *prec, 
options_t *opts, double *d_y, double *d_x) {
/*--------------------------------------------*/
  double *d_w,*d_diag;
  int k,ncol,i,j,j1,n1,n2,n,gDim;
  mcsor_prec_t *mcsor;
  csr_t *d_csr;
/*--------------------------------------------*/
  n = mat->n;
  k = opts->mcsor_opt->k;
  mcsor = prec->mcsor;
  ncol = mcsor->ncol;
  d_w = mcsor->d_w;
  d_diag = mcsor->d_diag;
  d_csr = mat->d_csr;
/*--------------------------------------------*/
  gDim = (n + BLOCKDIM -1)/BLOCKDIM;
  scal_k2<<<gDim, BLOCKDIM>>>
  (n, d_diag, d_x, d_y);
/*--------------------- SSOR(k) */
  for (i=0; i<k; i++)
    for (j=0; j<2*ncol; j++) {
/*----- forward SOR */
      if (j < ncol)
        j1 = j;
      else {
/*----- backward SOR */
        if (j == ncol) continue;
        if (i>0 && j==0) continue;
        j1 = 2*ncol-1-j;
      }
/*----- offset */
      n1 = mcsor->il[j1]-1;
/*----- size */
      n2 = mcsor->il[j1+1] - mcsor->il[j1];
/*----- w = -omega*(A*y-x) */
      spmv_sor(n, n2, d_csr->ia+n1, d_csr->ja,
      d_csr->a, d_y, d_x+n1, d_w+n1);
/*----- y := D^{-1}*w + y, diag(i) = 1/D(i) */
      gDim = (n2 + BLOCKDIM -1)/BLOCKDIM;
      scal_k<<<gDim, BLOCKDIM>>>
      (n2, d_diag+n1, d_w+n1, d_y+n1);
    }
}

