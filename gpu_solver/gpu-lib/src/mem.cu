#include "gpusollib.h"

/*-----------------------------------*/
void *cuda_malloc(int size) {
  void *p = NULL;
  CUDA_SAFE_CALL(cudaMalloc(&p, size));
  return(p);
}

/*---------------------------------------*/
void *cuda_malloc_host(int size) {
  void *p = NULL;
  CUDA_SAFE_CALL(cudaMallocHost(&p, size));
  return (p);
}

/*---------------------------------------------*/
void memcpyh2d(void *dest, void* src, int size) {
  CUDA_SAFE_CALL(cudaMemcpy(dest, src, size, 
  cudaMemcpyHostToDevice));
}

/*---------------------------------------------*/
void memcpyd2h(void *dest, void *src, int size) {
  CUDA_SAFE_CALL(cudaMemcpy(dest, src, size,
  cudaMemcpyDeviceToHost));
}

/*---------------------------------------------*/
void memcpyd2d(void *dest, void *src, int size) {
  CUDA_SAFE_CALL(cudaMemcpy(dest, src, size,
  cudaMemcpyDeviceToDevice));
}

/*---------------------------------------------*/
void cuda_memset(void *addr, int val, int size) {
  CUDA_SAFE_CALL(cudaMemset(addr, val, size));
}

/*------------------------------------------*/
void malloc_csr(int n, int nnz, csr_t *csr) {
  csr->n = n;
  csr->nnz = nnz;
  Malloc(csr->ia, n+1, int);
  Malloc(csr->ja, nnz, int);
  Malloc(csr->a,  nnz, double);
}

/*-----------------------------------------*/
void realloc_csr(csr_t *csr, int nnz) {
  csr->nnz = nnz;
  Realloc(csr->ja, nnz, int);
  Realloc(csr->a, nnz, double);
}

/*-----------------------------------------*/
void cuda_malloc_csr(int n, int nnz, 
                     csr_t *d_csr) {
  d_csr->ia = (int*)cuda_malloc((n+1)*sizeof(int));
  d_csr->ja = (int*)cuda_malloc(nnz*sizeof(int));
  d_csr->a = (double*)cuda_malloc(nnz*sizeof(double));
}

/*-------------------------------------------*/
void copy_csr_h2d(csr_t *h_csr, csr_t *d_csr) {
  int n = h_csr->n;
  int nnz = h_csr->nnz;
  d_csr->n = n;
  d_csr->nnz = nnz;
  memcpyh2d(d_csr->ia, h_csr->ia, (n+1)*sizeof(int));
  memcpyh2d(d_csr->ja, h_csr->ja, nnz*sizeof(int));
  memcpyh2d(d_csr->a,  h_csr->a,  nnz*sizeof(double));
}

/*--------------------------------------------*/
void copy_csr_h2h(csr_t *csr1, csr_t *csr2) {
  int n = csr1->n;
  int nnz = csr1->nnz;
  csr2->n = n;
  csr2->nnz = nnz;
  memcpy(csr2->ia, csr1->ia, (n+1)*sizeof(int));
  memcpy(csr2->ja, csr1->ja, nnz*sizeof(int));
  memcpy(csr2->a,  csr1->a,  nnz*sizeof(double));
}

/*------------------------------------------*/
void cuda_malloc_jad(int n, int njad, 
                     int nnz, jad_t *d_jad) {
  d_jad->ia = (int*)cuda_malloc((njad+1)*sizeof(int));
  d_jad->ja = (int*)cuda_malloc(nnz*sizeof(int));
  d_jad->a  = (double*)cuda_malloc(nnz*sizeof(double));
  d_jad->perm = (int*)cuda_malloc(n*sizeof(int));
  d_jad->w = (double*)cuda_malloc(n*sizeof(double));
}

/*--------------------------------------------*/
void copy_jad_h2d(jad_t *h_jad, jad_t *d_jad) {
  int njad = h_jad->njad;
  int nnz = h_jad->nnz;
  int n = h_jad->n;
  d_jad->n = n;
  d_jad->nnz = nnz;
  d_jad->njad = njad;
  memcpyh2d(d_jad->ia, h_jad->ia, (njad+1)*sizeof(int));
  memcpyh2d(d_jad->ja, h_jad->ja, nnz*sizeof(int));
  memcpyh2d(d_jad->a, h_jad->a, nnz*sizeof(double));
  memcpyh2d(d_jad->perm, h_jad->perm, n*sizeof(int));
}

/*-------------------------------------------*/
void cuda_malloc_dia(int nd, int strd, 
                     dia_t *d_dia) {
  d_dia->diags = 
  (double*) cuda_malloc(nd*strd*sizeof(double));
  d_dia->ioff = 
  (int*) cuda_malloc(nd*sizeof(int));
}

/*--------------------------------------------*/
void copy_dia_h2d(dia_t *h_dia, dia_t *d_dia) {
  int nd = h_dia->ndiags;
  int strd = h_dia->stride;
  d_dia->n = h_dia->n;
  d_dia->nnz = h_dia->nnz;
  d_dia->ndiags = nd;
  d_dia->stride = strd;
  memcpyh2d(d_dia->diags, h_dia->diags, 
            nd*strd*sizeof(double));
  memcpyh2d(d_dia->ioff, h_dia->ioff, nd*sizeof(int)); 
}

/*--------------------------------------------*/
void malloc_lu(int n, int nnzl, int nnzu, 
               lu_t *lu) {
  Malloc(lu->l, 1, csr_t);
  malloc_csr(n, nnzl, lu->l);
  Malloc(lu->u, 1, csr_t);
  malloc_csr(n, nnzu, lu->u);
}
/*--------------------------------------------*/
void cuda_malloc_lu(int n, int nnzl, int nnzu, 
                    lu_t *d_lu) {
  Malloc(d_lu->l, 1, csr_t);
  cuda_malloc_csr(n, nnzl, d_lu->l);
  Malloc(d_lu->u, 1, csr_t);
  cuda_malloc_csr(n, nnzu, d_lu->u);
}

/*---------------------------------------*/
void copy_lu_h2d(lu_t *h_lu, lu_t *d_lu) {
/*------- L */
  copy_csr_h2d(h_lu->l, d_lu->l);
/*------- U */
  copy_csr_h2d(h_lu->u, d_lu->u);
}

/*-----------------*/
void Free(void *p) {
  if (p) free(p);
}

/*---------------------------*/
void cuda_free(void *p) {
  if (p == NULL) return;
  CUDA_SAFE_CALL(cudaFree(p));
}

/*------------------------------*/
void cuda_free_host(void *p) {
  if (p == NULL) return;
  CUDA_SAFE_CALL(cudaFreeHost(p));
}

/*---------------------------*/
void free_coo(coo_t *coo) {
  if (coo == NULL) return;
  free(coo->ir);
  free(coo->jc);
  free(coo->val);
  free(coo);
}

/*------------------------*/
void free_csr(csr_t *csr) {
  if (csr == NULL) return;
  free(csr->a);
  free(csr->ia);
  free(csr->ja);
  free(csr);
}

/*------------------------------*/
void cuda_free_csr(csr_t *d_csr) {
  if (d_csr == NULL) return;
  cuda_free(d_csr->ia);
  cuda_free(d_csr->ja);
  cuda_free(d_csr->a);
  free(d_csr);
}

/*------------------------*/
void free_jad(jad_t *jad) {
  if (jad == NULL) return;
  free(jad->ia);
  free(jad->ja);
  free(jad->a);
  free(jad->perm);
  free(jad);
}

/*-------------------------------*/
void cuda_free_jad(jad_t *d_jad) {
  if (d_jad == NULL) return;
  cuda_free(d_jad->ia);
  cuda_free(d_jad->ja);
  cuda_free(d_jad->a);
  cuda_free(d_jad->perm);
  cuda_free(d_jad->w);
  free(d_jad);
}

/*------------------------*/
void free_dia(dia_t *dia) {
  if (dia == NULL) return;
  free(dia->diags);
  free(dia->ioff);
  free(dia);
}

/*------------------------------*/
void cuda_free_dia(dia_t *d_dia) {
  if (d_dia == NULL) return;
  cuda_free(d_dia->diags);
  cuda_free(d_dia->ioff);
  free(d_dia);
}

/*------------------------*/
void free_lu(lu_t *h_lu) {
  if (h_lu == NULL) return;
  free_csr(h_lu->l);
  free_csr(h_lu->u);
  free(h_lu);
}

/*---------------------------*/
void cuda_free_lu(lu_t *d_lu) {
  if (d_lu == NULL) return;
  cuda_free_csr(d_lu->l);
  cuda_free_csr(d_lu->u);
  free(d_lu);
}

/*---------------------------*/
void free_lev(level_t *h_lev) {
  if (h_lev == NULL) return;
  free(h_lev->jlevL);
  free(h_lev->ilevL);
  free(h_lev->jlevU);
  free(h_lev->ilevU);
  free(h_lev);
}

/*--------------------------------*/
void cuda_free_lev(level_t *d_lev) {
  if (d_lev == NULL) return;
  cuda_free(d_lev->jlevL);
  cuda_free(d_lev->ilevL);
  cuda_free(d_lev->jlevU);
  cuda_free(d_lev->ilevU);
  free(d_lev);
}

/*-----------------------------*/
void free_matrix(matrix_t *mat) {
  free_csr(mat->h_csr);
  cuda_free_csr(mat->d_csr);
  free_jad(mat->h_jad);
  cuda_free_jad(mat->d_jad);
  free_dia(mat->h_dia);
  cuda_free_dia(mat->d_dia);
  free(mat);
}

/*----------------------------*/
void free_ilu(ilu_prec_t *ilu) {
  if (ilu == NULL) return;
  free_lu(ilu->h_lu);
  cuda_free_lu(ilu->d_lu);
  cuda_free_host(ilu->h_x);
  cuda_free_host(ilu->h_b);
  free_lev(ilu->h_lev);
  cuda_free_lev(ilu->d_lev);
  free(ilu);
}

/*----------------------------------*/
void free_mcsor(mcsor_prec_t *mcsor) {
  if (mcsor == NULL) return;
  free(mcsor->kolrs);
  free(mcsor->il);
  cuda_free(mcsor->d_w);
  cuda_free(mcsor->d_diag);
  free(mcsor);
}

/*-------------------------------------*/
void free_mcilu0(mcilu0_prec_t *mcilu0) {
  if (mcilu0 == NULL) return;
  free(mcilu0->kolrs);
  free(mcilu0->il);
  cuda_free(mcilu0->d_w);
  free(mcilu0->h_diag);
  cuda_free(mcilu0->d_diag);
  free_lu(mcilu0->h_lu);
  cuda_free_lu(mcilu0->d_lu);
  free(mcilu0);
}

/*-------------------------------*/
void free_bilu(bilu_prec_t *bilu) {
  int nb,i;
  if (bilu == NULL) return;
  free(bilu->host->nrow);
  free(bilu->host->noff);
  nb = bilu->nb;
  for (i=0; i<nb; i++) {
    free(bilu->host->bdiag[i].ia);
    free(bilu->host->bdiag[i].ja);
    free(bilu->host->bdiag[i].a);
  }
  free(bilu->host->bdiag);
  
  for (i=0; i<nb; i++) {
    free_csr(bilu->host->blu[i].l);
    free_csr(bilu->host->blu[i].u);
  }
  free(bilu->host->blu);

  if (bilu->host->blev) {
    for (i=0; i<nb; i++) {
      free(bilu->host->blev[i].jlevL);
      free(bilu->host->blev[i].ilevL);
      free(bilu->host->blev[i].jlevU);
      free(bilu->host->blev[i].ilevU);
    }
    free(bilu->host->blev);
  }
  free(bilu->host);

  if (bilu->dev) {
    cuda_free(bilu->dev->a);
    cuda_free(bilu->dev->ja);
    cuda_free(bilu->dev->ia);
    cuda_free(bilu->dev->nrow);
    cuda_free(bilu->dev->noff);
    cuda_free(bilu->dev->jlev);
    cuda_free(bilu->dev->ilev);
    cuda_free(bilu->dev->nlev);
    free(bilu->dev);
  }

  cuda_free_host(bilu->h_x);
  cuda_free_host(bilu->h_b);
  free(bilu);
}

void free_ic(ic_prec_t *ic) {
  if (ic == NULL) return;
  free_csr(ic->LT);
  free(ic->D);
  cuda_free_host(ic->h_x);
  cuda_free_host(ic->h_b);
  free_lu(ic->h_lu);
  cuda_free_lu(ic->d_lu);
  free_lev(ic->h_lev);
  cuda_free_lev(ic->d_lev);
  free(ic);
}

void free_lspoly(lspoly_t *lspoly) {
  if (lspoly == NULL) return;
  free(lspoly->alpha);
  free(lspoly->beta);
  free(lspoly->gamma);
  cuda_free(lspoly->d_v1);
  cuda_free(lspoly->d_v0);
  cuda_free(lspoly->d_v);
  free(lspoly);
}
/*-------------------------------*/
void free_precon(precon_t *prec) {
  free_ilu(prec->ilu);
  free_mcsor(prec->mcsor);
  free_mcilu0(prec->mcilu0);
  free_bilu(prec->bilu);
  free_ic(prec->ic);
  free_lspoly(prec->lspoly);
  free(prec);
}

/*------------------------------*/
void free_opts(options_t *opts) {
  Free(opts->ilut_opt);
  Free(opts->iluk_opt);
  Free(opts->mcsor_opt);
  Free(opts->mcilu0_opt);
  Free(opts->bilu_opt);
  Free(opts->ic_opt);
  Free(opts->lspoly_opt);
  free(opts);
}

