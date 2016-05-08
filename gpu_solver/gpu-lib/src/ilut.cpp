#include "gpusollib.h"
#define EXPAND_FACT 1.5

/*--------------------------------------------*/
int qsplitC(double *a, int *ind, int n, int ncut) {
/*----------------------------------------------
| does a quick-sort split of a real array.
| on input a[0 : (n-1)] is a real array
| on output is permuted such that 
| its elements satisfy:
|
| abs(a[i]) >= abs(a[ncut-1]) for i < ncut-1 and
| abs(a[i]) <= abs(a[ncut-1]) for i > ncut-1
|
| ind[0 : (n-1)] is an integer array permuted 
| in the same way as a.
|----------------------------------------------*/
   double tmp, abskey;
   int j, itmp, first, mid, last;
   first = 0;
   last = n-1;
   if (ncut<first || ncut>=last) return 0;
/* outer loop -- while mid != ncut */
label1:
   mid = first;
   abskey = fabs(a[mid]);
  for (j=first+1; j<=last; j++) {
     if (fabs(a[j]) > abskey) {
	 tmp = a[++mid];
	 itmp = ind[mid];
	 a[mid] = a[j];
	 ind[mid] = ind[j];
	 a[j]  = tmp;
	 ind[j] = itmp;
      }
   }
/* interchange */
   tmp = a[mid];
   a[mid] = a[first];
   a[first]  = tmp;
   itmp = ind[mid];
   ind[mid] = ind[first];
   ind[first] = itmp;
/* test for while loop */
   if (mid == ncut) return 0;
   if (mid > ncut) 
      last = mid-1;
   else
      first = mid+1;
   goto label1;
}

int ilut(csr_t *csr, lu_t *lu, double tol, int p) {
/*-----------------------------------------------*/
/*   ikj version ILUT (row-by-row, up-looking)   */
/*          dual throshold (tol, lfil)           */
/*         csr matrix is 1-based index           */
/*-----------------------------------------------*/
  int n = csr->n; 
  int len, lenu, lenl;
  int *ia, *ja, *jbuf, *iw, i, j, k,i1,i2;
  int col, jpos, jrow, upos,ctrL,ctrU;
  double t, tnorm, tolnorm, fact, lxu, *a, *w;
  csr_t *L, *U;
/*-----------------------------------------------*/
  malloc_lu(n, 2*n, 2*n, lu);
  L = lu->l;
  U = lu->u;
  Malloc(iw, n, int);
  Malloc(jbuf, n, int);
  Malloc(w, n, double);
  ia = csr->ia;
  ja = csr->ja;
  a = csr->a;
  ctrL = ctrU = 0;
  (L->ia)[0] = (U->ia)[0] = 1;
/*----- set indicator array jw to -1 */
  for (i=0; i<n; i++) 
    iw[i] = -1;
/*----- beginning of main loop */
  for (i=0; i<n; i++) {
    i1 = ia[i];
    i2 = ia[i+1];
    tnorm = 0.0;
    for(j=i1; j<i2; j++ )
      tnorm += fabs(a[j-1]);
    if( tnorm == 0.0 ) {
      printf("ilut: zero row encountered.\n");
      return -2;
    }
    tnorm /= (double)(i2-i1);
    tolnorm = tol*tnorm;
/*--------- unpack L/U-part of row of A in w */
    lenu = 0;
    lenl = 0;
    jbuf[i] = i;
    w[i] = 0.0;
    iw[i] = i;
    for(j=i1; j<i2; j++) {
      col = ja[j-1]-1;
      t = a[j-1];
      if(col < i) {
        iw[col] = lenl;
        jbuf[lenl] = col;
        w[lenl] = t;
        lenl++;
      } else if( col == i ) {
        w[i] = t;
      } else {
        lenu++;
        jpos = i + lenu;
        iw[col] = jpos;
        jbuf[jpos] = col;
        w[jpos] = t;
      }
    }
    j = -1;
    len = 0;
/*--------- eliminate previous rows */
    while( ++j < lenl ) {
/*---------------------------------------
 *  in order to do the elimination in 
 *  the correct order we must select the
 *  smallest column index among 
 *  jbuf[k], k = j+1, ..., lenl
 *---------------------------------------*/
      jrow = jbuf[j];
      jpos = j;
      /* determine smallest column index */
      for( k=j+1; k<lenl; k++ ) {
        if( jbuf[k] < jrow ) {
          jrow = jbuf[k];
          jpos = k;
        }
      }
      if( jpos != j ) {
        col = jbuf[j];
        jbuf[j] = jbuf[jpos];
        jbuf[jpos] = col;
        iw[jrow] = j;
        iw[col]  = jpos;
        t = w[j];
        w[j] = w[jpos];
        w[jpos] = t;
      }
 /*--------- get the multiplier */
      fact = w[j] * (U->a)[(U->ia)[jrow]-1];
      w[j] = fact;
 /*------ zero out by resetting iw(jrow) to -1 */
      iw[jrow] = -1;
 /*-------- combine current row and row jrow */
      i1 = (U->ia)[jrow]+1;
      i2 = (U->ia)[jrow+1];
      for(k=i1; k<i2; k++) {
        col = (U->ja)[k-1]-1;
        jpos = iw[col];
        lxu = -fact * (U->a)[k-1];
/*--if fill-in element is small then disregard*/
        if(fabs(lxu) < tolnorm && jpos == -1) 
          continue;
/*-------------------- */
        if( col < i ) {
/*------ dealing with lower part */
          if( jpos == -1 ) {
/*------ this is a fill-in element */
            jbuf[lenl] = col;
            iw[col] = lenl;
            w[lenl] = lxu;
            lenl++;
          } else {
            w[jpos] += lxu;
          }
        } else {
/*----------- dealing with upper part */
          if( jpos == -1 ) {
//if( jpos == -1 && fabs(lxu) > tolnorm)
/*--------- this is a fill-in element */
            lenu++;
            upos = i + lenu;
            jbuf[upos] = col;
            iw[col] = upos;
            w[upos] = lxu;
          } else {
            w[jpos] += lxu;
          }
        }
      }
    }
/*------- restore iw */
    iw[i] = -1;
    for(j=0; j<lenu; j++) {
      iw[jbuf[i+j+1]] = -1;
    }
/*---------- case when diagonal is zero */
    if( w[i] == 0.0 ) {
      fprintf(stdout, "zero diag(%d)\n",i);
      w[i] = 1e-6;
    }
/*------ update L-matrix */
    len = lenl < p ? lenl : p;
    if (ctrL+len > L->nnz)
      realloc_csr(L, (int)(L->nnz*EXPAND_FACT));
    qsplitC(w, jbuf, lenl, len);
    for (j=0; j<len; j++) {
      (L->ja)[ctrL] = jbuf[j]+1;
      (L->a)[ctrL] = w[j];
      ctrL++;
    }
    (L->ia)[i+1] = ctrL+1;
/*----------- update U-matrix */
    len = lenu < p ? lenu : p;
    if (ctrU+len+1 > U->nnz)
      realloc_csr(U, (int)(U->nnz*EXPAND_FACT));
    qsplitC(w+i+1, jbuf+i+1, lenu, len);
/*----------- update diagonal */    
    (U->ja)[ctrU] = i+1;
    (U->a)[ctrU] = 1.0 / w[i];
    ctrU++;
/*------------------------- */
    for (j=0; j<len; j++) {
      (U->ja)[ctrU] = jbuf[i+1+j]+1;
      (U->a)[ctrU] = w[i+1+j];
      ctrU++;
    }
    (U->ia)[i+1] = ctrU+1;
  }
/*----------- resize L/U */
  k = L->ia[n]-1;
  realloc_csr(L, k);
  k = U->ia[n]-1;
  realloc_csr(U, k);
/*------- done */
  free(iw);
  free(jbuf);
  free(w);
  return 0;
}

/*------------------------------------------*/
void SetupILUt(matrix_t *mat, precon_t *prec, 
options_t *opts) {
/*------------------------------------------*/
  int err, lfil,lusolgpu,n,nnzl,nnzu;
  double filfact, tol;
  csr_t *csr;
  lu_t *h_lu;
/*-------------------------------*/
  tol  = opts->ilut_opt->tol;
  lfil = opts->ilut_opt->lfil;
  lusolgpu = opts->lusolgpu;
/*-------------------------------*/
  Calloc(prec->ilu, 1, ilu_prec_t);
  Calloc(prec->ilu->h_lu, 1, lu_t);
/*-------------------------------*/
  csr = mat->h_csr;
  h_lu = prec->ilu->h_lu;
  printf("being ilut ...\n");
  err = ilut(csr, h_lu, tol, lfil);
  if (err) {
    printf("ilut error:%d\n", err);
    exit(-1);
  }
/*------------------------------*/
  nnzl = h_lu->l->nnz;
  nnzu = h_lu->u->nnz;
  filfact = (nnzl+nnzu)/(double)(csr->nnz);
  printf("  ilut ends, fill-factor %f\n",\
  filfact);
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
/*--------------------------------------*/
  Calloc(prec->ilu->d_lev, 1, level_t);
  copy_level_h2d(n, prec->ilu->h_lev, 
                    prec->ilu->d_lev);
/*---------- copy LU to device */
  Calloc(prec->ilu->d_lu, 1, lu_t);
  cuda_malloc_lu(n, nnzl, nnzu, prec->ilu->d_lu);
  copy_lu_h2d(h_lu, prec->ilu->d_lu);
}

