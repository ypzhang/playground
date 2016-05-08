#include "gpusollib.h"
#define EXPAND_FACT 1.5

/*------------------------------------------*/
int search_col_index(int j, int lenl, int *iL,
int *llev, int *iw) {
/*------------------------------------------*/
/* Linear Search for the smallest row index */
/*------------------------------------------*/
  int k, irow, ipos;
  irow = iL[j];
  ipos = j;
/*---------- determine smallest col index */
  for(k=j+1; k<lenl; k++) {
    if( iL[k] < irow ) {
      irow = iL[k];
      ipos = k;
    }
  }
  if( ipos != j ) {
/*-------------------- exchange entries */
    int row = iL[j];
    iL[j] = iL[ipos];
    iL[ipos] = row;
    int t = llev[j];
    llev[j] = llev[ipos];
    llev[ipos] = t;
    iw[irow] = j;
    iw[row]  = ipos;
  }
  return (irow);
}

/*--------------------------------------------*/
int symblc(int lfil, csr_t *csmat, lu_t *lu) {
  int n = csmat->n;
  int i,j,k,k1,k2,jlev,lev1,row,col,ipos,lenl,lenu;
  int *ja, *ia, *iU, *iL, *iw;
  csr_t *L, *U;
  int *lev, *ulev, *llev, ctrU, ctrL;
/*--------------------------------------------*/
  malloc_lu(n, 2*n, 2*n, lu);
  L = lu->l;
  U = lu->u;
  Malloc(lev, 2*n, int);
  Malloc(iw, n, int);
  Malloc(iL, n, int);
  ja = csmat->ja; 
  ia = csmat->ia;
  ctrL = ctrU = 0;
  (L->ia)[0] = (U->ia)[0] = 1;
  Malloc(llev, n, int);
  for (i=0; i<n; i++)
    iw[i] = -1;
/*----------- main loop */
  for (i=0; i<n; i++) {
    iU = iL+i;
    ulev = llev+i;
    lenl = lenu = 0;
/*---------- unpack the ith row */
    k1 = ia[i]; k2 = ia[i+1];
    for (j=k1; j<k2; j++) {
      col = ja[j-1]-1;
      if (col > i) {
/*------ u part */
        iw[col] = lenu;
        iU[lenu] = col;
        ulev[lenu] = 0;
        lenu++;
      } else if (col < i) {
/*------ l part */
        iw[col] = lenl;
        iL[lenl] = col;
        llev[lenl] = 0;
        lenl++;
      }
    }
/*---------  eliminate rows */
    j = -1;
    while (++j < lenl) {
      row = search_col_index(j, lenl, iL, llev, iw);
      jlev = llev[j];
      k1 = (U->ia)[row]+1;
      k2 = (U->ia)[row+1];
      for (k=k1; k<k2; k++) {
        col = (U->ja)[k-1]-1;
        lev1 = jlev + lev[k-1] + 1;
        if (lev1 > lfil) continue;
        ipos = iw[col];
        if (ipos == -1) {
/*-------- fill-in */
          if (col > i) {
/*-------- U part */
            iw[col] = lenu;
            iU[lenu] = col;
            ulev[lenu] = lev1;
            lenu++;
          } else if (col < i) {
/*-------- L part */
            iw[col] = lenl;
            iL[lenl] = col;
            llev[lenl] = lev1;
            lenl++;
          }
        }
        else {
/*--------- not a fill-in */
            if (col > i) 
              ulev[ipos] = min(ulev[ipos], lev1);
            else if (col < i)
              llev[ipos] = min(llev[ipos], lev1);
        }
      }
    }
/*----- reset iw */
    for (j=0; j<lenl; j++) iw[iL[j]] = -1;
    for (j=0; j<lenu; j++) iw[iU[j]] = -1;
/*----- copy U part+diag and levels */
    if (ctrU+lenu+1 > U->nnz) {
      int newsize = (int)(U->nnz*EXPAND_FACT);
      realloc_csr(U, newsize);
      Realloc(lev, newsize, int)
    }
/*------ diag entry */
    (U->ja)[ctrU] = i+1;
    ctrU++;
/*-------- U part */
    for (j=0; j<lenu; j++) {
      (U->ja)[ctrU] = iU[j]+1;
      lev[ctrU] = ulev[j];
      ctrU++;
    }
    (U->ia)[i+1] = ctrU+1;
/*----- copy L part */
    if (ctrL+lenl > L->nnz) {
      int newsize = (int) (L->nnz*EXPAND_FACT);
      realloc_csr(L, newsize);
    }
    for (j=0; j<lenl; j++) {
      (L->ja)[ctrL] = iL[j]+1;
      ctrL++;
    }
    (L->ia)[i+1] = ctrL+1;
  }

  free(iw);
  free(iL);
  free(llev);
  free(lev);
  return 0;
}

int iluk(csr_t *csmat, lu_t *lu, int lfil) {
/*----------------------------------------*/
  int n = csmat->n;
  int *ja, *pA, *iw;
  int *jl, *pl, *ju, *pu;
  int k1, k2, i, j, k;
  double *ml, *mu, fact; 
  int row, ipos, col;
  double *ma, lxu; 
  csr_t *L, *U;
/*-------------------- symbolic ilu(k) */
  if (symblc(lfil, csmat, lu)) 
    return 1;
/*------------------------------------------- */
  L = lu->l; U = lu->u;
  ja = csmat->ja; ma = csmat->a; pA = csmat->ia;
  jl = L->ja; ml = L->a; pl = L->ia;
  ju = U->ja; mu = U->a; pu = U->ia;
/*---------------- set marker arrays iw to -1 */
  Malloc(iw, n, int);
  for(i=0; i<n; i++) iw[i] = -1;
/*-------------------- beginning of main loop */
  for (i=0; i<n; i++) {
    k1 = pu[i]; k2 = pu[i+1];
    for (j=k1; j<k2; j++) {
      col = ju[j-1]-1;
      mu[j-1] = 0.0;
      iw[col] = j-1;
    }
    k1 = pl[i]; k2 = pl[i+1];
    for (j=k1; j<k2; j++) {
      col = jl[j-1]-1;
      ml[j-1] = 0.0;
      iw[col] = j-1;
    }
/*--------------- unpack the ith row of A */
    k1 = pA[i]; k2 = pA[i+1];
    for (j=k1; j<k2; j++) {
      col = ja[j-1]-1;
      ipos = iw[col];
      if (col < i)
        ml[ipos] = ma[j-1];
      else
        mu[ipos] = ma[j-1];
    }
/*---------- eliminate prev rows */
    k1 = pl[i]; k2 = pl[i+1];
    for (j=k1; j<k2; j++) {
      row = jl[j-1]-1;
      fact = ml[j-1] * mu[pu[row]-1];
      ml[j-1] = fact;
      for (k=pu[row]+1; k<pu[row+1]; k++) {
        col = ju[k-1]-1;
        ipos = iw[col];
        if (ipos == -1) continue;
        lxu = -mu[k-1] * fact;
        if (col < i)
          ml[ipos] += lxu;
        else
          mu[ipos] += lxu;
      }
    }
    if (mu[iw[i]] == 0.0) {
      printf("zero diag(%d)\n", i);
      mu[iw[i]] = 1e6;
    } else {
      mu[iw[i]] = 1.0 / mu[iw[i]];
    }
/*----- reset */
    k1 = pu[i]; k2 = pu[i+1];
    for (j=k1; j<k2; j++)
      iw[ju[j-1]-1] = -1;
    k1 = pl[i]; k2 = pl[i+1];
    for (j=k1; j<k2; j++)
      iw[jl[j-1]-1] = -1;
  }
/*----------- resize L/U */
  k = L->ia[n]-1;
  realloc_csr(L, k);
  k = U->ia[n]-1;
  realloc_csr(U, k);
  free(iw);

  return 0;
}

/*------------------------------------------*/
void SetupILUK(matrix_t *mat, precon_t *prec,
options_t *opts) {
/*------------------------------------------*/
  int err, lfil,lusolgpu,n,nnzl,nnzu;
  double filfact;
  csr_t *csr;
  lu_t *h_lu;
/*-------------------------------*/
  lfil = opts->iluk_opt->lfil;
  lusolgpu = opts->lusolgpu;
/*-------------------------------*/
  Calloc(prec->ilu, 1, ilu_prec_t);
  Calloc(prec->ilu->h_lu, 1, lu_t);
/*-------------------------------*/
  csr = mat->h_csr;
  h_lu = prec->ilu->h_lu;
  printf("being ilu(%d) ...\n",lfil);
  err = iluk(csr, h_lu, lfil);
  if (err) {
    printf("iluk error:%d\n", err);
    exit(-1);
  }
/*------------------------------*/
  nnzl = h_lu->l->nnz;
  nnzu = h_lu->u->nnz;
  filfact = (nnzl+nnzu)/(double)(csr->nnz);
  printf("  ilu(%d) ends, fill-factor %f\n",\
  lfil, filfact);
  opts->result.filfact = filfact;
/*--------------------------*/
  n = csr->n;
  if (lusolgpu == 0) {
/*---------- cpu solve */
    prec->ilu->h_x = 
    (double*)cuda_malloc_host(n*sizeof(double));
    prec->ilu->h_b = 
    (double*)cuda_malloc_host(n*sizeof(double));
    return;
  }
/*---------- gpu solve */
/*---------- level scheduling */
  Calloc(prec->ilu->h_lev, 1, level_t);
  make_level(h_lu, prec->ilu->h_lev);
  opts->result.ulev = prec->ilu->h_lev->nlevU;
  opts->result.llev = prec->ilu->h_lev->nlevL;
/*-------------------------------------*/
  Calloc(prec->ilu->d_lev, 1, level_t);
  copy_level_h2d(n, prec->ilu->h_lev, 
                    prec->ilu->d_lev);
/*---------- copy LU to device */
  Calloc(prec->ilu->d_lu, 1, lu_t);
  cuda_malloc_lu(n, nnzl, nnzu, prec->ilu->d_lu);
  copy_lu_h2d(h_lu, prec->ilu->d_lu);
}

