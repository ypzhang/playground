#include "gpusollib.h"

/*----------------------------------------------------------*
 *      Block ILU solv with level scheduling kernel         *
 *      half-warp per row
 *----------------------------------------------------------*/
__global__
void BLU_LEV(double *d_x, double *d_y, double *d_a, int *d_ja,
             int *d_ia, int nziv, int *d_jlev, int *d_ilev, 
             int *d_nlev, int n, int *noffs, int *nrows) {
  int i,j,k,noff,nrow,nnzl,*ja,*ia,nlev,*jlev,
      *ilev;
  double *a,*b,*x;

  noff = noffs[blockIdx.x];
  nrow = nrows[blockIdx.x];

  // thread lane in each half warp
  int lane = threadIdx.x & (HALFWARP-1);
  // half warp lane in the block
  int hwlane = threadIdx.x/HALFWARP;
  // shared memory for patial result
  volatile __shared__ double r[BLOCKDIM2+8];

   a = &d_a[blockIdx.x*nziv];
  ja = &d_ja[blockIdx.x*nziv];
  ia = &d_ia[2*(blockIdx.x+noff)];

  nlev = d_nlev[blockIdx.x];
  jlev = &d_jlev[2*noff];
  ilev = &d_ilev[2*(blockIdx.x+noff)];

  b = &d_x[noff];
  x = &d_y[noff];

  /* Forward solve. Solve L*x = b */
  for (i=0; i<nlev; i++) {
    int p = ilev[i];
    int q = ilev[i+1];
    for (j=p+hwlane; j<q; j+=NHW2) {
      int jj = jlev[j-1]-1;
      int p1 = ia[jj];
      int q1 = ia[jj+1];

      double sum = 0.0;
      for (k=p1+lane; k<q1; k+=HALFWARP)
        sum += a[k-1]*x[ja[k-1]-1];

      // parallel reduction
      r[threadIdx.x] = sum;
      r[threadIdx.x] = sum = sum + r[threadIdx.x+8];
      r[threadIdx.x] = sum = sum + r[threadIdx.x+4];
      r[threadIdx.x] = sum = sum + r[threadIdx.x+2];
      r[threadIdx.x] = sum = sum + r[threadIdx.x+1];

      if (lane == 0)
        x[jj] = b[jj] - r[threadIdx.x];
    }

    __syncthreads();
  }

  nnzl = ia[nrow]-1;
   a = &a[nnzl];
  ja = &ja[nnzl];
  ia = &ia[nrow+1];
  nlev = d_nlev[gridDim.x+blockIdx.x];
  jlev = &jlev[nrow];
  ilev = &ilev[nrow+1];

  /* Backward slove. Solve x = U^{-1}*x */
  for (i=0; i<nlev; i++) {
    int p = ilev[i];
    int q = ilev[i+1];
    for (j=p+hwlane; j<q; j+=NHW2) {
      int jj = jlev[j-1]-1;
      int p1 = ia[jj];
      int q1 = ia[jj+1];

      double sum = 0.0;
      for (k=p1+1+lane; k<q1; k+=HALFWARP)
        sum += a[k-1]*x[ja[k-1]-1];

      // parallel reduction
      r[threadIdx.x] = sum;
      r[threadIdx.x] = sum = sum + r[threadIdx.x+8];
      r[threadIdx.x] = sum = sum + r[threadIdx.x+4];
      r[threadIdx.x] = sum = sum + r[threadIdx.x+2];
      r[threadIdx.x] = sum = sum + r[threadIdx.x+1];

      if (lane == 0) {
        double t = a[p1-1];
        x[jj] = t*(x[jj]-r[threadIdx.x]);
      }
    }

    __syncthreads();
  }
}

/*---------------------------------------------------*/
void luSolCPU2(int n, double *b, double *x, double *la, 
int *lja, int *lia, double *ua, int *uja, int *uia) {
/*---------------------------------------------------*/
  int i,k,i1,i2;

  /* Forward solve. Solve L*x = b */
  for (i=0; i<n; i++) {
    x[i] = b[i];
    i1 = lia[i];
    i2 = lia[i+1];
    for (k=i1; k<i2; k++)
      x[i] -= la[k-1]*x[lja[k-1]-1];
  }
  
  /* Backward slove. Solve x = U^{-1}*x */
  for (i=n-1; i>=0; i--) {
    double t = ua[uia[i]-1];
    i1 = uia[i]+1;
    i2 = uia[i+1];
    for (k=i1; k<i2; k++)
      x[i] -= ua[k-1]*x[uja[k-1]-1];

    x[i] = t*x[i];
  }
}

/*--------------------------------------------------*/
void BluSolCPU(int n, int bn, bilu_host_t *h_bilu,
double *d_y, double *d_x, double *h_b, double *h_x) {
/*--------------------------------------------------*/
  int i,nrow,noff,*lja,*uja,*lia,*uia;
  double *la,*ua;
  memcpyd2h(h_b, d_x, n*sizeof(double));  
  for (i=0; i<bn; i++) {
    nrow = h_bilu->nrow[i];
    noff = h_bilu->noff[i];
    la = h_bilu->blu[i].l->a;
    ua = h_bilu->blu[i].u->a;
    lja = h_bilu->blu[i].l->ja;
    uja = h_bilu->blu[i].u->ja;
    lia = h_bilu->blu[i].l->ia;
    uia = h_bilu->blu[i].u->ia;
    luSolCPU2(nrow, h_b+noff, h_x+noff, 
              la, lja, lia, ua, uja, uia);
  }
  memcpyh2d(d_y, h_x, n*sizeof(double));  
}

/*---------------------------------------*/
void blusol(matrix_t *mat, precon_t *prec, 
options_t *opts, double *d_y, double *d_x) {
/*--------------------------*/
  int nb,n;
  bilu_dev_t *bilu;
/*--------------------------*/
  n = mat->n;
  nb = opts->bilu_opt->bn;
  bilu = prec->bilu->dev;
/*--------------------------*/
  if (opts->lusolgpu == 1)
    BLU_LEV<<<nb, BLOCKDIM2>>>
    (d_x, d_y, bilu->a, bilu->ja, bilu->ia, 
    bilu->nzinterval, bilu->jlev, bilu->ilev, 
    bilu->nlev, n, bilu->noff, bilu->nrow);
  else
    BluSolCPU(n, nb, prec->bilu->host, d_y, d_x, 
    prec->bilu->h_b, prec->bilu->h_x);
}

