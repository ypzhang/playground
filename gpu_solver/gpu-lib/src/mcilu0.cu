#include "gpusollib.h"

__global__
void scal_k2(int n, double *d, double *x, double *y);

/*------------------------------------------*/
/*  NOTE: we assume that A(i,:) is ordered  */
/*  by increasing col index                 */
/*------------------------------------------*/
int ildu0(csr_t *csr, lu_t *lu, double *d) {
  int i,j,k,n,nnz,*ia,*ja,*iw,row,col,pos;
  int ctrL, ctrU;
  double *a,fact,lxu;
  csr_t *L,*U;
/*-------------------*/
  n = csr->n;
  nnz = csr->nnz;
  ia = csr->ia;
  ja = csr->ja;
  a = csr->a;
/*-----------------------*/
  malloc_lu(n, nnz, nnz, lu);
  L = lu->l;
  U = lu->u;
  Malloc(iw, n, int);
  for (i=0; i<n; i++)
    iw[i] = -1;
  ctrL = ctrU = 0;
  (L->ia)[0] = (U->ia)[0] = 1;
/*------- main loop for each row */
  for (i=0; i<n; i++) {
    d[i] = 0.0;
/*------- unpack row i to L/U */
    for (j=ia[i]; j<ia[i+1]; j++) {
      col = ja[j-1]-1;
      if (col < i) {
/*------- L part */
        iw[col] = ctrL;
        L->ja[ctrL] = col+1;
        L->a[ctrL] = a[j-1];
        ctrL++;
      } else if (col > i){
/*------- U part */
        iw[col] = ctrU;
        U->ja[ctrU] = col+1;
        U->a[ctrU] = a[j-1];
        ctrU++;
      } else {
/*------ diag entry */
        d[i] = a[j-1];
      }
    }
/*------ next row ptr */
    L->ia[i+1] = ctrL+1;
    U->ia[i+1] = ctrU+1;
/*------ elimination of prev rows */
    for (j=L->ia[i]; j<L->ia[i+1]; j++) {
      row = L->ja[j-1]-1;
      //if (j>L->ia[i]) assert(row > L->ja[L->ia[i]-1]-1);
/*------ get the multiplier */
      fact = L->a[j-1] * d[row];
      L->a[j-1] = fact;
/*------ combine prev row */
      for (k=U->ia[row]; k<U->ia[row+1]; k++) {
        col = U->ja[k-1]-1;
        lxu = -fact * U->a[k-1];
        if (col < i) {
/*------ L part */
          pos = iw[col];
          if (pos != -1)
            L->a[pos] += lxu;
        } else if (col > i) {
/*------ U part */
          pos = iw[col];
          if (pos != -1)
            U->a[pos] += lxu;
        } else {
/*------ diag entry */
          d[i] += lxu;
        }
      }
    }
/*----------*/
    if (d[i] == 0.0) {
      printf("zero diag(%d)\n", i);
      d[i] = 1e-6;
    }
    d[i] = 1.0 / d[i];
/*------ reset iw */
    for (j=L->ia[i]; j<L->ia[i+1]; j++)
      iw[L->ja[j-1]-1] = -1;
    for (j=U->ia[i]; j<U->ia[i+1]; j++)
      iw[U->ja[j-1]-1] = -1;
  }
/*------- resize L/U */
  realloc_csr(L, ctrL);
  realloc_csr(U, ctrU);
/*------ done, free */
  free(iw);
  return 0;
}

/*--------------------------------------------*/
void SetupMCILU0a(matrix_t *mat, precon_t *prec,
options_t *opts) {
/*--------------------------------------------*/
  int n,nnzl,nnzu,err;
  double *diag;
  csr_t *csr;
  lu_t *h_lu,*d_lu;
/*-------------------------------------------*/
  n = mat->n;
  csr = mat->h_csr;
  assert(prec->mcilu0);
/*-------------*/
  //check_mc(csr, prec->mcilu0->ncol,\
  prec->mcilu0->il);
/*-------------*/
  Calloc(prec->mcilu0->h_lu, 1, lu_t);
  h_lu = prec->mcilu0->h_lu;
  Malloc(diag, n, double);
/*--------- ilu0 ---------------*/
  //sortrow(csr);
  printf("starting ildu0 ...\n");
  fflush(stdout);
  err=ildu0(csr, h_lu, diag);
  printf("  Done\n");
  if (err != 0) { 
    printf("ildu0 err:%d\n", err);
    exit(-1);
  }
  prec->mcilu0->h_diag = diag;
/*-------- copy to device */
  Calloc(prec->mcilu0->d_lu, 1, lu_t);
  d_lu = prec->mcilu0->d_lu;
  nnzl = h_lu->l->nnz;
  nnzu = h_lu->u->nnz;
  cuda_malloc_lu(n, nnzl, nnzu, d_lu);
  copy_lu_h2d(h_lu, d_lu);
/*------------------*/
  prec->mcilu0->d_w =
  (double*) cuda_malloc(n*sizeof(double));
  prec->mcilu0->d_diag = 
  (double*) cuda_malloc(n*sizeof(double));
  memcpyh2d(prec->mcilu0->d_diag, diag,
  n*sizeof(double));
/*-----------------------------*/
}

/*--------------------------------------------*/
void SetupMCILU0(matrix_t *mat, precon_t *prec,
options_t *opts) {
/*--------------------------------------------*/
  int n,nnzl,nnzu,err,sp;
  double *diag;
  csr_t *csr,*B;
  lu_t *h_lu,*d_lu;
/*-------------------------------------------*/
  n = mat->n;
  csr = mat->h_csr;
  sp = opts->mcilu0_opt->sp;
  assert(prec->mcilu0);
/*-------------*/
  //check_mc(csr, prec->mcilu0->ncol,\
  prec->mcilu0->il);
/*-------------*/
  Calloc(prec->mcilu0->h_lu, 1, lu_t);
  h_lu = prec->mcilu0->h_lu;
  Malloc(diag, n, double);
/*----- drop entries in diag blocks*/
  if (sp) 
    B = mcprune_csr(csr, prec->mcilu0);
  else
    B = csr;
/*---------- ilu0 ------------*/
  printf("starting ildu0 ...\n");
  fflush(stdout);
/*-------- factorize B */
  err=ildu0(B, h_lu, diag);
  printf("  Done\n");
  if (err != 0) { 
    printf("ildu0 err:%d\n", err);
    exit(-1);
  }
  prec->mcilu0->h_diag = diag;
/*----------------------------*/
/*------ drop entries in diag blocks */
//  mcprune_lu(h_lu, prec->mcilu0);
/*-------- copy to device */
  Calloc(prec->mcilu0->d_lu, 1, lu_t);
  d_lu = prec->mcilu0->d_lu;
  nnzl = h_lu->l->nnz;
  nnzu = h_lu->u->nnz;
  cuda_malloc_lu(n, nnzl, nnzu, d_lu);
  copy_lu_h2d(h_lu, d_lu);
/*------------------*/
  prec->mcilu0->d_w =
  (double*) cuda_malloc(n*sizeof(double));
  prec->mcilu0->d_diag = 
  (double*) cuda_malloc(n*sizeof(double));
  memcpyh2d(prec->mcilu0->d_diag, diag,
  n*sizeof(double));
/*-----------------------------*/
}

/*--------------------------------------------*/
void SetupMCILU0c(matrix_t *mat, precon_t *prec,
options_t *opts) {
/*--------------------------------------------*/
  int n,nnzl,nnzu,err,sp;
  double *diag;
  csr_t *csr;
  lu_t *h_lu,*d_lu;
/*-------------------------------------------*/
  n = mat->n;
  csr = mat->h_csr;
  sp = opts->mcilu0_opt->sp;
  assert(prec->mcilu0);
/*-------------*/
  //check_mc(csr, prec->mcilu0->ncol,\
  prec->mcilu0->il);
/*-------------*/
  Calloc(prec->mcilu0->h_lu, 1, lu_t);
  h_lu = prec->mcilu0->h_lu;
  Malloc(diag, n, double);
/*--------- ilu0 ---------------*/
  printf("starting ildu0 ...\n");
  fflush(stdout);
  err=ildu0(csr, h_lu, diag);
  printf("  Done\n");
  if (err != 0) { 
    printf("ildu0 err:%d\n", err);
    exit(-1);
  }
  prec->mcilu0->h_diag = diag;
/*------ drop entries in diag blocks */
  if (sp) mcprune_lu(h_lu, prec->mcilu0);
/*-------- copy to device */
  Calloc(prec->mcilu0->d_lu, 1, lu_t);
  d_lu = prec->mcilu0->d_lu;
  nnzl = h_lu->l->nnz;
  nnzu = h_lu->u->nnz;
  cuda_malloc_lu(n, nnzl, nnzu, d_lu);
  copy_lu_h2d(h_lu, d_lu);
/*------------------*/
  prec->mcilu0->d_w =
  (double*) cuda_malloc(n*sizeof(double));
  prec->mcilu0->d_diag = 
  (double*) cuda_malloc(n*sizeof(double));
  memcpyh2d(prec->mcilu0->d_diag, diag,
  n*sizeof(double));
/*-----------------------------*/
}

__global__
void scal_k3(int n, double *d, double *x, double *y) {
  int nt  = gridDim.x * blockDim.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int i;
  for (i=idx; i<n; i+=nt)
    y[i] = d[i]*(y[i]+x[i]);
}

/*---------------------------------------*/
void mcilu0op(matrix_t *mat, precon_t *prec, 
options_t *opts, double *d_y, double *d_x) {
/*--------------------------------------*/
  int n, ncol, *il, i, n1, n2, gDim;
  double *diag, *d_w;
  lu_t *d_lu;
  csr_t *L, *U;
/*--------------------------------------*/
  n = mat->n;
  ncol = prec->mcilu0->ncol;
  il = prec->mcilu0->il;
  d_lu = prec->mcilu0->d_lu;
  L = d_lu->l;
  U = d_lu->u;
  diag = prec->mcilu0->d_diag;
  d_w = prec->mcilu0->d_w;
  cublasDcopy(n, d_x, 1, d_y, 1);
/*--------- L-solve */
  for (i=1; i<ncol; i++) {
    n1 = il[i]-1;
    n2 = il[i+1] - il[i];
    spmv_ilu0_1(n1, n2, L->ia+n1, 
    L->ja, L->a, d_y);
  }
/*--------- U-solve */
  n1 = il[ncol-1]-1;
  n2 = il[ncol] - il[ncol-1];
  gDim = (n2 + BLOCKDIM - 1)/BLOCKDIM;
  scal_k2<<<gDim, BLOCKDIM>>>
  (n2, diag+n1, d_y+n1, d_y+n1);
/*------------------------------*/
  for (i=ncol-2; i>=0; i--) {
    n1 = il[i]-1;
    n2 = il[i+1] - il[i];
    spmv_ilu0_2(n, n2, U->ia+n1, 
    U->ja, U->a, d_y, d_w+n1);
    gDim = (n2 + BLOCKDIM - 1)/BLOCKDIM;
    scal_k3<<<gDim, BLOCKDIM>>>
    (n2, diag+n1, d_w+n1, d_y+n1);
  }
}

