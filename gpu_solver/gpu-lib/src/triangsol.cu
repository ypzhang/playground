#include "gpusollib.h"

/*-------------------------------------------*/
/*     level scheduling kernel               */
/*-------------------------------------------*/
__global__
void L_SOL_LEV(double *b, double *x, double *la, 
               int *lja, int *lia,
               int *jlevL, int l1, int l2) {
  int i,k,jj;

  // num of half-warps
  int nhw = gridDim.x*BLOCKDIM/HALFWARP;
  // half warp id
  int hwid = (blockIdx.x*BLOCKDIM+threadIdx.x)/HALFWARP;
  // thread lane in each half warp
  int lane = threadIdx.x & (HALFWARP-1);
  // shared memory for patial result
  volatile __shared__ double r[BLOCKDIM+8];

  for (i=l1+hwid; i<l2; i+=nhw) {
    jj = jlevL[i-1]-1;
    int p1 = lia[jj];
    int q1 = lia[jj+1];

    double sum = 0.0;
    for (k=p1+lane; k<q1; k+=HALFWARP)
      sum += la[k-1]*x[lja[k-1]-1];

    // parallel reduction
    r[threadIdx.x] = sum;
    r[threadIdx.x] = sum = sum + r[threadIdx.x+8];
    r[threadIdx.x] = sum = sum + r[threadIdx.x+4];
    r[threadIdx.x] = sum = sum + r[threadIdx.x+2];
    r[threadIdx.x] = sum = sum + r[threadIdx.x+1];

    if (lane == 0)
      x[jj] = b[jj] - r[threadIdx.x];
  }
}

/*----------------- x = U^{-1}*x -----------*/
__global__
void U_SOL_LEV(double *x, double *ua, int *uja, 
int *uia, int *jlevU, int l1, int l2) {
/*------------------------------------------*/
  int i,k,jj;

  // num of half-warps
  int nhw = gridDim.x*BLOCKDIM/HALFWARP;
  // half warp id
  int hwid = (blockIdx.x*BLOCKDIM+threadIdx.x)/HALFWARP;
  // thread lane in each half warp
  int lane = threadIdx.x & (HALFWARP-1);
  // shared memory for patial result
  volatile __shared__ double r[BLOCKDIM+8];

  for (i=l1+hwid; i<l2; i+=nhw) {
    jj = jlevU[i-1]-1;
    int p1 = uia[jj];
    int q1 = uia[jj+1];

    double sum = 0.0;
    for (k=p1+1+lane; k<q1; k+=HALFWARP)
      sum += ua[k-1]*x[uja[k-1]-1];

    // parallel reduction
    r[threadIdx.x] = sum;
    r[threadIdx.x] = sum = sum + r[threadIdx.x+8];
    r[threadIdx.x] = sum = sum + r[threadIdx.x+4];
    r[threadIdx.x] = sum = sum + r[threadIdx.x+2];
    r[threadIdx.x] = sum = sum + r[threadIdx.x+1];

    if (lane == 0) {
      double t = ua[p1-1];
      x[jj] = t*(x[jj]-r[threadIdx.x]);
    }
  }
}

/*---------------------------------------*/
void lu_solv_lev(ilu_prec_t *ilu, 
double *d_x, double *d_b) {
/*---------------------------------------*/
  int i,nlev,l1,l2,*ilev,*jlev,*ia,*ja;
  int nthreads, gDim, bDim;
  lu_t *d_lu;
  level_t *h_lev, *d_lev;
  double *a;
/*---------------------------*/
  d_lu  = ilu->d_lu;
  h_lev = ilu->h_lev;
  d_lev = ilu->d_lev;
/*----------- L-solve -------*/
  nlev = h_lev->nlevL;
  ilev = h_lev->ilevL;
  jlev = d_lev->jlevL;
  a  = d_lu->l->a;
  ja = d_lu->l->ja;
  ia = d_lu->l->ia;
/*---------------------------*/
  for (i=0; i<nlev; i++) {
    l1 = ilev[i];
    l2 = ilev[i+1];
    nthreads = min((l2-l1)*HALFWARP, MAXTHREADS);
    gDim = (nthreads+BLOCKDIM-1)/BLOCKDIM;
    bDim = BLOCKDIM;
    L_SOL_LEV<<<gDim, bDim>>>
    (d_b, d_x, a, ja, ia, jlev, l1, l2);
  }
/*----------- U-solve ------*/
  nlev = h_lev->nlevU;
  ilev = h_lev->ilevU;
  jlev = d_lev->jlevU;
  a  = d_lu->u->a;
  ja = d_lu->u->ja;
  ia = d_lu->u->ia;
/*---------------------------*/
  for (i=0; i<nlev; i++) {
    l1 = ilev[i];
    l2 = ilev[i+1];
    nthreads = min((l2-l1)*HALFWARP, MAXTHREADS);
    gDim = (nthreads+BLOCKDIM-1)/BLOCKDIM;
    bDim = BLOCKDIM;
    U_SOL_LEV<<<gDim, bDim>>>
    (d_x, a, ja, ia, jlev, l1,l2);
  }
}

/*------------------------------*/
void lu_sol_cpu(ilu_prec_t *ilu, 
     double *d_x, double *d_b) { 
/*------------------------------*/
  double *a,*h_b,*h_x;
  int *ja,*ia,i,k,i1,i2,n;
  lu_t *h_lu;
/*------------------------------*/
  h_x = ilu->h_x;
  h_b = ilu->h_b;
  h_lu = ilu->h_lu;
  n = h_lu->l->n;
  memcpyd2h(h_b, d_b, n*sizeof(double));
/*----- Forward solve */
  a  = h_lu->l->a;
  ia = h_lu->l->ia;
  ja = h_lu->l->ja;
/*----- Solve L*x = b */
  for (i=0; i<n; i++) {
    h_x[i] = h_b[i];
    i1 = ia[i];
    i2 = ia[i+1];
    for (k=i1; k<i2; k++)
      h_x[i] -= a[k-1]*h_x[ja[k-1]-1];
  }  
/*----- Backward slove */
  a  = h_lu->u->a;
  ia = h_lu->u->ia;
  ja = h_lu->u->ja;
/*----- Solve x = U^{-1}*x */
  for (i=n-1; i>=0; i--) {
    double t = a[ia[i]-1];
    i1 = ia[i]+1;
    i2 = ia[i+1];
    for (k=i1; k<i2; k++)
      h_x[i] -= a[k-1]*h_x[ja[k-1]-1];
    h_x[i] = t * h_x[i];
  }
  memcpyh2d(d_x, h_x, n*sizeof(double));
}

/*--------------------------------------*/
void lusol(matrix_t *mat, precon_t *prec, 
options_t *opts, double *d_x, double *d_b) {
/*--------------------------------------*/
  if (opts->lusolgpu == 1)
    lu_solv_lev(prec->ilu, d_x, d_b);
  else
    lu_sol_cpu(prec->ilu, d_x, d_b);
}

/*------------------*/
/*   L*L^{T} solve  */
/*------------------*/
/*--------------------------*/
void lltsol_cpu(ic_prec_t *ic, 
double *d_x, double *d_b) {
/*----------------------*/
  int n,i,j,*ia,*ja;
  double *a,*h_b,*h_x,*diag;
/*----------------------*/
  n = ic->LT->n;
  ia = ic->LT->ia;
  ja = ic->LT->ja;
  a  = ic->LT->a;
  h_b = ic->h_b;
  h_x = ic->h_x;
  diag = ic->D;
/*--------------------*/
  memcpyd2h(h_b, d_b, n*sizeof(double));
  for (i=0; i<n; i++) {
    h_b[i] /= diag[i];
    for (j=ia[i]; j<ia[i+1]; j++)
      h_b[ja[j-1]-1] -= h_b[i]*a[j-1];
  }
  for (i=n-1; i>=0; i--) {
    h_x[i] = h_b[i];
    for (j=ia[i]; j<ia[i+1]; j++)
      h_x[i] -= a[j-1] * h_x[ja[j-1]-1];
    h_x[i] /= diag[i];
  }
  memcpyh2d(d_x, h_x, n*sizeof(double));
}

/*---------------------------*/
void lltsol_gpu(ic_prec_t *ic, 
double *d_x, double *d_b) {
/*---------------------------*/
  ilu_prec_t ilu;
  ilu.d_lu = ic->d_lu;
  ilu.h_lev = ic->h_lev;
  ilu.d_lev = ic->d_lev;
  lu_solv_lev(&ilu, d_x, d_b);
}

/*-----------------------------------------*/
void lltsol(matrix_t *mat, precon_t *prec, 
options_t *opts, double *d_x, double *d_b) {
/*-----------------------------------------*/
  if (opts->lusolgpu == 1)
    lltsol_gpu(prec->ic, d_x, d_b);
  else
    lltsol_cpu(prec->ic, d_x, d_b);
}
