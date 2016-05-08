#include "gpusollib.h"

/*---------------------------------------*/
void bilut(csr_t *csr, bilu_host_t *h_bilu,
int nb, int p, double tol, double *avgfil) {
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
  printf("begin bilut(%d,%.2e) ...\n", p, tol);
  Calloc(h_bilu->blu, nb, lu_t);
  double *fil;
  Malloc(fil, nb, double);
  for (i=0; i<nb; i++) {
    if (ilut(&h_bilu->bdiag[i], &h_bilu->blu[i],
        tol, p)) {
       printf("BILUT error in block %d\n", i);
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
void SetupBILUT(matrix_t *mat, precon_t *prec,
options_t *opts) {
/*------------------------------------------*/
  int n,bn,i,j1,j2,p,dd,lusolgpu,*nrow,*noff;
  double tol,avgfil;
/*------------------------------------------*/
  n = mat->n;
  bn = opts->bilu_opt->bn;
  p = opts->bilu_opt->lfil;
  tol = opts->bilu_opt->tol;
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
  bilut(mat->h_csr, prec->bilu->host, 
  bn, p, tol, &avgfil);
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
  printf("%d block ilut prec ... done\n", bn);
}

