#include "gpusollib.h"

void genrcm ( int, int, int*, int*, int*);
void genmmd(int, int *, int *, int *, int *,
	    int, int *, int *, int *, int *,
	    int, int *);

/*-----------------------------------*/
void symmgraph(csr_t *A, csr_t *C) {
  int n = A->n;
  int nnz = A->nnz;

  csr_t AT;
  Malloc(AT.ja, nnz, int);
  Malloc(AT.ia, n+1, int);

/*------- A': by csr2csc */
/*------- job == 0 only pattern needed */
  csrcsc(n, n, 0, 1, NULL, A->ja, 
         A->ia, NULL, AT.ja, AT.ia);
// pattern of C = A+A'
  int nzmax = 2*nnz;
  Malloc(C->ja, nzmax, int);
  Malloc(C->ia, n+1, int);
// only the pattern is computed
  int job = 0; 
  int *iw = (int *) malloc(n*sizeof(int));
  int err = -1;
  aplb_(&n, &n, &job, NULL, A->ja, A->ia, NULL, 
        AT.ja, AT.ia, NULL, C->ja, C->ia, 
        &nzmax, iw, &err);
  if (err != 0) {
    printf("SPARSKIT aplb error\n");
    exit(-1);
  }
  C->n = n;
  C->nnz = C->ia[n]-1;

  free(iw);
  free(AT.ja);
  free(AT.ia);
}

/*----------------------------------*/
void remove_diag(csr_t *C) {
  int i,j,k,n,nnz,*ja2,*ia2;
/*----------------------------------*/
  n = C->n;  nnz = C->nnz;
/*----------------------------------*/
  Malloc(ja2, nnz, int);
  Malloc(ia2, n+1, int);
  k = 0;    
  ia2[0] = 1;
  for (i=0; i<n; i++) {
    for (j=C->ia[i]; j<C->ia[i+1]; j++)
      if (C->ja[j-1]-1 != i)
	ja2[k++] = C->ja[j-1];	  
    ia2[i+1] = k+1;
  }
  free(C->ia);  free(C->ja);
/*----------------------------------*/
  Realloc(ja2, k, int);
  C->ia = ia2;
  C->ja = ja2;
}

/*---------------------------------------*/
/* Reorder CSR matrix h_csr by RCM  */
void reorder_rcm(csr_t *A, int *perm) {
  int *iperm,nnz,n,*ia,*ja,i;
/*---------------------------------------*/
  n = A->n;
  ia = A->ia; ja = A->ja;
  nnz = ia[n]-1;
  Malloc(iperm, n, int);
  genrcm(n, nnz, ia, ja, iperm);
  for (i=0; i<n; i++)
    perm[iperm[i]-1] = i+1;
  free(iperm);
}

/*---------------------------------------*/
/* Reorder CSR matrix h_csr by ND metis  */
void reorder_nd(csr_t *A, int *perm) {
  int *iperm,n,*ia,*ja;
/*---------------------------------------*/
  n = A->n;
  ia = A->ia;  ja = A->ja;
  Malloc(iperm, n, int);
  int nf = 1;
  int opt = 0;
  METIS_NodeND(&n,ia,ja,&nf,&opt,iperm,perm);
  free(iperm);
}

/*--------------------------------------*/
/*        Multiple Minimum Degree       */
/*--------------------------------------*/
void reorder_mmd(csr_t *A, int *perm) {
  int *iperm,mxint,n,*ia,*ja;
/*------------------------------------------*/
  n = A->n;
  ia = A->ia;  ja = A->ja;
  mxint = (1<<(8*sizeof(int)-2));
  Malloc(iperm, n, int);
  int delta,*head,*qsize,*Nlist,*marker,nzout;
  delta = 1;
  Malloc(head, n, int);
  Malloc(qsize, n, int);
  Malloc(Nlist, n, int);
  Malloc(marker, n, int);
  genmmd(n, ia, ja, perm, iperm, delta, head,
         qsize, Nlist, marker, mxint, &nzout);
  free(iperm);
  free(head);
  free(qsize);
  free(Nlist);
  free(marker);
}

/*-----------------------------------*/
void reorder_sym(csr_t *A, options_t *opts, 
                 int **perm) {
/*-----------------------------------*/
  REORDER_TYPE reord;
  int n;
  csr_t *C;
/*-----------------------------------*/
  reord = opts->reord;
/*-----------------------------------*/
  if (reord == NONE) {
    *perm = NULL;
    return;
  }
/*--------- reordering - ----------*/
/*--------- by RCM MMD ND or METIS */
  n = A->n;
  Malloc(*perm, n, int);
/*--------------- symmetrize A */
  Malloc(C, 1, csr_t);
  symmgraph(A, C);
/*---------- remove self loop */
  remove_diag(C);
/*-------------------------------*/
  if (reord == RCM)
    reorder_rcm(C, *perm);
  else if (reord == ND)
    reorder_nd (C, *perm);
  else if (reord == MMD)
    reorder_mmd(C, *perm);
/*-------------------------------*/
  free(C->ia); free(C->ja);
  free(C);
}

/*----------------------------------------*/
void perm_mat_sym(csr_t *A, int *perm) {
  if (perm == NULL) return;
  int job = 1;
  int n = A->n;
  int nnz = A->nnz;
  double *ao;
  Malloc(ao, nnz, double);
  int *jao, *iao;
  Malloc(jao, nnz, int);
  Malloc(iao, n+1, int);
  dperm_(&n, A->a, A->ja, A->ia, 
             ao, jao, iao, perm, NULL, &job);
  free(A->a);
  free(A->ja);
  free(A->ia);
  A->a = ao;
  A->ia = iao;
  A->ja = jao;
}

/*---------------------------------------*/
void perm_vec(int n, double *x, int *perm) {
  if (perm == NULL) return;
  vperm_(&n, x, perm);
}

/*---------------------------------------*/
void iperm_vec(int n, double *x, int *perm) {
  int i, *iperm;
/*---------------------------------------*/
  if (perm == NULL) return;
/*------ inv perm array */
  Malloc(iperm, n, int);
  for (i=0; i<n; i++)
    iperm[perm[i]-1] = i+1;
/*------- perm */
  vperm_(&n, x, iperm);
  free(iperm);
}
