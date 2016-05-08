#include "gpusollib.h"

/*--------------------------------------------*/
void SetupMCSOR(matrix_t *mat, precon_t *prec,
options_t *opts) {
/*--------------------------------------------*/
  csr_t *csr = mat->h_csr;
  int n,i,j;
  double *diag,d;
/*--------------------------------------------*/
  n = mat->n;
  assert(prec->mcsor);
  prec->mcsor->d_w = 
  (double*) cuda_malloc(n*sizeof(double));
  prec->mcsor->d_diag = 
  (double*) cuda_malloc(n*sizeof(double));
/*-------- extract diag */
  Calloc(diag, n, double);
  for (i=0; i<n; i++) {
    d = 0.0;
    for (j=csr->ia[i]; j<csr->ia[i+1]; j++) {
      if (i == csr->ja[j-1]-1) {
        d = csr->a[j-1];
        break;
      }
    }
    if (d == 0.0) {
      printf("zero diag[%d]\n",i);
      d = 1e-4;
    }
    diag[i] = 1.0/d;
  }
/*--------- copy to device */
  memcpyh2d(prec->mcsor->d_diag, diag, 
  n*sizeof(double));
  free(diag);
}

