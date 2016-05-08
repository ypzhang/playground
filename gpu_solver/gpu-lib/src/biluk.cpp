#include "gpusollib.h"

/*-----------------------------------------*/  
void Partition1D(int len, int pnum, int idx, 
int &j1, int &j2) {
/*-----------------------------------------*
   Partition of 1D array
   Input:
   len:  length of the array
   pnum: partition number
   idx:  index of a partition
   Output:
   j1,j2: partition [j1, j2)
 *-----------------------------------------*/
  int size = (len+pnum-1)/pnum;
  if (idx < (size*pnum - len))
    j1 = idx * (--size);
  else
    j1 = len - size*(pnum-idx);
  j2 = j1 + size;
}

/*---------------------------------------*/
void biluk(csr_t *csr, bilu_host_t *h_bilu,
int nb, int kk, double *avgfil) {
/*---------------------------------------*/
  int i,j,k,j1,j2;
  int *ia = csr->ia;
  int *ja = csr->ja;
  double *a = csr->a;
/*---------------------------------------*/
/*---------- diagonal blocks */
  Calloc(h_bilu->bdiag, nb, csr_t);
/*------ nnz in all diag blocks */
  int nnzds = 0;
/*------ for each block diag */
  for (i=0; i<nb; i++) {
    int nnzd = 0;
    j1 = h_bilu->noff[i];
    j2 = h_bilu->noff[i] + h_bilu->nrow[i];
/*---------- nnz of block diag i */
    for (j=j1; j<j2; j++)
      for (k=ia[j]; k<ia[j+1]; k++)
        if (ja[k-1] > j1 && ja[k-1] <= j2)
          nnzd ++;
/*---------- alloc diag block csr */
    nnzds += nnzd;
    malloc_csr(j2-j1, nnzd, h_bilu->bdiag+i);
    int *ia2   = h_bilu->bdiag[i].ia;
    int *ja2   = h_bilu->bdiag[i].ja;
    double *a2 = h_bilu->bdiag[i].a;
/*------------- build block diag i */
    ia2[0] = 1;
    //ptr for ja, a
    int p = 0; 
    for (j=j1; j<j2; j++) {
      int rownnz = 0;
      for (k=ia[j]; k<ia[j+1]; k++)
        if (ja[k-1] > j1 && ja[k-1] <= j2) {
          rownnz ++;
          a2[p] = a[k-1];
          ja2[p] = ja[k-1] - j1;
          p++;
        }
      ia2[j+1-j1] = ia2[j-j1] + rownnz;
    }
  }
  printf("number of blocks %d\n", nb);
  printf("  %f in diag blocks\n",\
  (double)nnzds/(double)csr->nnz);
/*------- iluk for each block diag */
  printf("begin biluk(%d) ...\n", kk);
  Calloc(h_bilu->blu, nb, lu_t);
  double *fil;
  Malloc(fil, nb, double);
  for (i=0; i<nb; i++) {
    if (iluk(&h_bilu->bdiag[i], &h_bilu->blu[i],
        kk)) {
       printf("BILUK error in block %d\n", i);
       exit(-1);
    }
    int nnzl = h_bilu->blu[i].l->nnz;
    int nnzu = h_bilu->blu[i].u->nnz;
    int nnzdi = h_bilu->bdiag[i].nnz;
    fil[i] = (nnzl+nnzu) / (double)(nnzdi);
  }
/*-------- average fill factor */
  (*avgfil) = 0.0;
  for (i=0; i<nb; i++)
    (*avgfil) += fil[i];
  (*avgfil) /= nb;
  printf("  done, avg fillfactor %f\n",*avgfil);
  free(fil);
}

/*------------------------------------------*/
void SetupBILUK(matrix_t *mat, precon_t *prec,
options_t *opts) {
/*------------------------------------------*/
  int n,bn,i,j1,j2,k,dd,lusolgpu,*nrow,*noff;
  double avgfil;
/*------------------------------------------*/
  n = mat->n;
  bn = opts->bilu_opt->bn;
  k = opts->bilu_opt->lfil;
  lusolgpu = opts->lusolgpu;
  dd = opts->bilu_opt->dd;
/*------------------------------------*/
  if (dd) {
    assert(prec->bilu);
    assert(prec->bilu->host);
  } else {
/*---- no domain domcop */
    assert(prec->bilu == NULL);
    Calloc(prec->bilu, 1, bilu_prec_t);
    Calloc(prec->bilu->host, 1, bilu_host_t);
    Malloc(nrow, bn, int);
    Malloc(noff, bn, int);
    for (i=0; i<bn; i++) {
      Partition1D(n, bn, i, j1, j2);
      noff[i] = j1;
      nrow[i] = j2 - j1;
    }
    prec->bilu->host->nrow = nrow;
    prec->bilu->host->noff = noff;
  }
/*--------------------*/
  prec->bilu->nb = bn;
/*------------ iluk fact for all blocks*/
  biluk(mat->h_csr, prec->bilu->host, 
  bn, k, &avgfil);
  opts->result.bfilfact = avgfil;
/*----------------------------------------*/
  if (lusolgpu == 0) {
/*---------- cpu solve */
    prec->bilu->h_x = 
    (double*)cuda_malloc_host(n*sizeof(double));
    prec->bilu->h_b = 
    (double*)cuda_malloc_host(n*sizeof(double));
    return;
  }
/*---------- gpu solve */
/*---------- level scheduling */
  Malloc(prec->bilu->host->blev, bn, level_t);
  for (i=0; i<bn; i++)
    make_level(prec->bilu->host->blu+i, 
              prec->bilu->host->blev+i);
/*---------- copy to device */
  Calloc(prec->bilu->dev, 1, bilu_dev_t);
  blu_h2d(prec->bilu->host, prec->bilu->dev, n, bn);
  printf("%d block ilu[%d] prec ... done\n", bn, k);
}

/*-------------------------------------*/
void blu_h2d(bilu_host_t *h_bilu, 
bilu_dev_t *d_bilu, int n, int bn) {
/*-------------------------------------*/
  int i,j,p,q,*ia,*ja,*jlev,*ilev,*nlev;
  double *a;
/*-------------- nrow, noff */
  d_bilu->nrow = (int*) cuda_malloc(bn*sizeof(int));
  d_bilu->noff = (int*) cuda_malloc(bn*sizeof(int));
  memcpyh2d(d_bilu->nrow, h_bilu->nrow, bn*sizeof(int));
  memcpyh2d(d_bilu->noff, h_bilu->noff, bn*sizeof(int));
/*-------------- block ilu csr */
  j = 0; //max nnz in all block ilu
  for (i=0; i<bn; i++) {
    int nnzl = h_bilu->blu[i].l->nnz;
    int nnzu = h_bilu->blu[i].u->nnz;
    j = max(j, (nnzl+nnzu));
  }
  d_bilu->nzinterval = j;
/*-------------------------------------------*/
  d_bilu->a = (double*) cuda_malloc(bn*j*sizeof(double));
  d_bilu->ja = (int*) cuda_malloc(bn*j*sizeof(int));
  d_bilu->ia = (int*) cuda_malloc(2*(n+bn)*sizeof(int));
  Calloc(a, bn*j, double);
  Calloc(ja, bn*j, int); // ja
  Calloc(ia, 2*(n+bn), int);
  p = q = 0;
  for (i=0; i<bn; i++) {
    assert(q == 2*(h_bilu->noff[i]+i));
    // l
    memcpy(&a[p], h_bilu->blu[i].l->a, 
           h_bilu->blu[i].l->nnz*sizeof(double));
    memcpy(&ja[p], h_bilu->blu[i].l->ja,
           h_bilu->blu[i].l->nnz*sizeof(int));
    memcpy(&ia[q], h_bilu->blu[i].l->ia, 
          (h_bilu->blu[i].l->n+1)*sizeof(int));
    p += h_bilu->blu[i].l->nnz;
    q += h_bilu->blu[i].l->n + 1;
    // u
    memcpy(&a[p], h_bilu->blu[i].u->a, 
           h_bilu->blu[i].u->nnz*sizeof(double));
    memcpy(&ja[p], h_bilu->blu[i].u->ja, 
           h_bilu->blu[i].u->nnz*sizeof(int));
    memcpy(&ia[q], h_bilu->blu[i].u->ia, 
          (h_bilu->blu[i].u->n+1)*sizeof(int));
    q += (h_bilu->blu[i].u->n + 1);
    assert((i+1)*j >= p+h_bilu->blu[i].u->nnz);
    p = (i+1)*j;
  }
  assert(q == 2*(n+bn));
/*------------- copy a, ia, ja to device */
  memcpyh2d(d_bilu->a,  a,  bn*j*sizeof(double));
  memcpyh2d(d_bilu->ja, ja, bn*j*sizeof(int));
  memcpyh2d(d_bilu->ia, ia, 2*(n+bn)*sizeof(int));
/*----------*/
  free(a);  free(ia);  free(ja);
/*------- level of block iu */
  Calloc(jlev, 2*n, int);
  Calloc(ilev, 2*(n+bn), int);
  Calloc(nlev, 2*bn, int);
  p = q = 0;
  for (i=0; i<bn; i++) {
    nlev[i] = h_bilu->blev[i].nlevL;
    nlev[bn+i] = h_bilu->blev[i].nlevU;
    assert(nlev[i] <= h_bilu->nrow[i]);
    assert(nlev[bn+i] <= h_bilu->nrow[i]);
    // l
    memcpy(&jlev[p], h_bilu->blev[i].jlevL, 
           h_bilu->nrow[i]*sizeof(int));
    memcpy(&ilev[q], h_bilu->blev[i].ilevL, 
           (nlev[i]+1)*sizeof(int));
    p += h_bilu->nrow[i];
    q += h_bilu->nrow[i]+1;
    // u
    memcpy(&jlev[p], h_bilu->blev[i].jlevU, 
           h_bilu->nrow[i]*sizeof(int));
    memcpy(&ilev[q], h_bilu->blev[i].ilevU, 
          (nlev[bn+i]+1)*sizeof(int));
    p += h_bilu->nrow[i];
    q += h_bilu->nrow[i]+1;
  }
/*---- copy jlev ilev nlev to device */
  d_bilu->jlev = (int*) cuda_malloc(2*n*sizeof(int));
  d_bilu->ilev = (int*) cuda_malloc(2*(n+bn)*sizeof(int));
  d_bilu->nlev = (int*) cuda_malloc(2*bn*sizeof(int));

  memcpyh2d(d_bilu->jlev, jlev, 2*n*sizeof(int));
  memcpyh2d(d_bilu->ilev, ilev, 2*(n+bn)*sizeof(int));
  memcpyh2d(d_bilu->nlev, nlev, 2*bn*sizeof(int));
  free(jlev);
  free(ilev);
  free(nlev);
/*-------- check for error so far */
  cuda_check_err();
}

