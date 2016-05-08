#include "gpusollib.h"
#define MAXLOOP 5
#define MAXTOL 0.02

/*---------------------------------------*/
void reorder_mc(csr_t *A, precon_t *prec,
options_t *opts, int **perm) {
/*---------------------------------------*/
  PREC_TYPE ptype;
  int n, sp;
/*---------------------------------------*/
  ptype = opts->prectype;
/*---------------------------------------*/
  if (ptype != MCSOR && ptype != MCILU0) {
    *perm = NULL;
    return;
  }
/*--------- reordering -----------*/
/*--------- by multi-color */
  if (ptype == MCSOR)
    sp = opts->mcsor_opt->sp;
  else
    sp = opts->mcilu0_opt->sp;
/*------------------------*/
  n = A->n;
  Malloc(*perm, n, int);
  if (sp)
    mulcol_sp(A,opts,prec,*perm);
  else
    mulcol(A,opts,prec,*perm);
/*------------------*/
  if (ptype == MCSOR)
    opts->result.ncol = prec->mcsor->ncol;
  else
    opts->result.ncol = prec->mcilu0->ncol;
}

/*---------------------------------------*/
void mulcol(csr_t *A, options_t *opts, 
precon_t *prec, int *perm) {
/*---------------------------------------*/
  PREC_TYPE ptype;
  int n,i,maxcol,ncol,*kolrs,*il,err;
  csr_t *C;
/*---------------------------------------*/
  n = A->n;
  ptype = opts->prectype;
  printf("begin MULTI-COLOR ...\n");
/*------- symetrize matrix */
  Calloc(C, 1, csr_t);
  symmgraph(A, C);
/*-------- multi-color reordering */
  if (ptype == MCSOR)
    maxcol = opts->mcsor_opt->maxcol;
  else
    maxcol = opts->mcilu0_opt->maxcol;
/*---------------------*/
  Calloc(kolrs, n, int);
  Calloc(il, maxcol+1, int);
/*-------- input node order */
  for (i=0; i<n; i++) perm[i] = i+1;
/*-------- multi-coloring, greedy alg */
  multic_(&n, C->ja, C->ia, &ncol, kolrs,
  il, perm, &maxcol, &err);
  if (err != 0) {
    printf("exceed max num of colors\n");
    exit(-1);
  }
  printf("  done, %d colors\n", ncol);
/*-----------------------*/
  if (ptype == MCSOR) {
    Calloc(prec->mcsor, 1, mcsor_prec_t);
    prec->mcsor->ncol = ncol;
    prec->mcsor->kolrs = kolrs;
    prec->mcsor->il = il;
  } else {
    Calloc(prec->mcilu0, 1, mcilu0_prec_t);
    prec->mcilu0->ncol = ncol;
    prec->mcilu0->kolrs = kolrs;
    prec->mcilu0->il = il;
  }
/*----------------- done */
  free(C->ja);
  free(C->ia);
  free(C);
}

/*--------------------------------------*/
void mulcol_sp(csr_t *A, options_t *opts,
precon_t *prec, int *perm) {
/*---------------------------------------*/
  PREC_TYPE ptype;
  int n,i,maxcol,ncol,*kolrs,*il,err,k;
  double tol;
  csr_t *C,*B;
/*---------------------------------------*/
  n = A->n;
  ptype = opts->prectype;
  if (ptype == MCSOR) {
    maxcol = opts->mcsor_opt->maxcol;
    tol = opts->mcsor_opt->tol;
  }
  else {
    maxcol = opts->mcilu0_opt->maxcol;
    tol = opts->mcilu0_opt->tol;
  }
  printf("begin Sparsified MULTI-COLOR ...\n");
/*-------------------------------------*/
  Calloc(kolrs, n, int);
  Calloc(il, maxcol+1, int);
/*------------- filter out small terms */
/*------------- B: sparsified matrix */
  Calloc(B, 1, csr_t);
  Filter(A, B, tol);
/*--------------------*/
  Calloc(C, 1, csr_t);
  for (k=0; k<MAXLOOP; k++) {
/*----- symetrize matrix B */
    symmgraph(B, C);
/*----- init ordering */
    for (i=0; i<n; i++)
      perm[i] = i+1;
/*----- multi-color */
    multic_(&n, C->ja, C->ia, &ncol, kolrs,
    il, perm, &maxcol, &err);
    printf("  # of colors: %d\n", ncol);
    if (err != 0) {
      printf("exceed max num of colors\n");
      exit(-1);
    }
/*-------------------------*/
    free(C->ja);  free(C->ia);
/*-------------------------------*/
    if (tol*2 > MAXTOL || ncol < 5)
      break;
    tol = tol*2;
/*------- further filtering */
/*---- drop small entries in long rows */
    Filter2(B, tol, ncol);
  }
/*----------------------------------*/
  if (ptype == MCSOR) {
    Calloc(prec->mcsor, 1, mcsor_prec_t);
    prec->mcsor->ncol = ncol;
    prec->mcsor->kolrs = kolrs;
    prec->mcsor->il = il;
  } else {
    Calloc(prec->mcilu0, 1, mcilu0_prec_t);
    prec->mcilu0->ncol = ncol;
    prec->mcilu0->kolrs = kolrs;
    prec->mcilu0->il = il;
  }
/*----------------- done */
  free_csr(B);
  free(C);
}

/*---------------------------------------------*/
void Filter(csr_t *A, csr_t *B, double drptol) {
/*---------------------------------------------*/
  int n,nnz,job,err;
/*---------------------------------------------*/
  n = A->n;
  nnz = A->nnz;
  malloc_csr(n, nnz, B);
  job = 2;
  filter_(&n, &job, &drptol, A->a, A->ja, A->ia,
          B->a, B->ja, B->ia, &nnz, &err);
  if (err != 0) {
    printf("SPARSKIT filter error![%d]\n", err);
    exit(-1);
  }
  nnz = B->ia[n]-1;
/*------ resize B*/
  realloc_csr(B, nnz);
}

/*----------------------------------------------*/
void Filter2(csr_t *A, double drptol, int ncol) {
/*----------------------------------------------*/
  int i,j,k,n,*ia,*ja,j1,j2,deg;
  double *a,norm,toli;
/*----------------------------------------------*/
  n = A->n;
  ia = A->ia;
  ja = A->ja;
  a = A->a;
  k = 0;
  for (i=0; i<n; i++) {
    j1 = ia[i];
    j2 = ia[i+1];
/*------- row pointer */
    ia[i] = k+1;
/*--- deg of node i */
    deg = 0;
    for (j=j1; j<j2; j++)
      if (ja[j-1]-1 != i)
        deg ++;
/*--- if deg+1 is large */
    if (deg+1 >= ncol) {
/*----- row i's 2-norm */
      norm = 0.0;
      for (j=j1; j<j2; j++)
        norm += a[j-1]*a[j-1];
      norm = sqrt(norm);
/*----- scan row i, drop small entries */
      toli = norm * drptol;
      for (j=j1; j<j2; j++)
        if (fabs(a[j-1]) > toli) {
          a[k] = a[j-1];
          ja[k] = ja[j-1];
          k++;
        }
    } else {
/*-------- short row */
      for (j=j1; j<j2; j++) {
        a[k] = a[j-1];
        ja[k] = ja[j-1];
        k++;
      }
    }
  }
  ia[n] = k+1;
/*------ resize A */
  realloc_csr(A, k);
}

