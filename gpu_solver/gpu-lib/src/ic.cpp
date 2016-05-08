#include "gpusollib.h"
#define EXPAND_FACT 1.5

/*-----------------------------*/
void GetU2(csr_t *A, csr_t *U) {
  int n = A->n;
  int nnz = A->nnz;
/*-----------------------------*/
  Malloc(U->a, nnz, double);
  Malloc(U->ja, nnz, int);
  Malloc(U->ia, n+1, int);
  getu_(&n, A->a, A->ja, A->ia,
        U->a, U->ja, U->ia);
  U->n = n;
  U->nnz = U->ia[n]-1;
  Realloc(U->ja, U->nnz, int);
  Realloc(U->a,  U->nnz, double);
}

/*-----------------*/
/* column pointer  */
/*-----------------*/
typedef struct colptr_t_ {
  int size;
  int len;
  int *rowidx;
  double *value;
} colptr_t;

/*-----------------------------------------------*/
void col_insert(colptr_t *col, int row, double val) {
  if (col->len == col->size) {
    col->size = col->size*2 + 1;
    Realloc(col->rowidx, col->size, int);
    Realloc(col->value, col->size, double);
  }
  col->rowidx[col->len] = row;
  col->value[col->len] = val;
  col->len ++;
}

/*----------------------------------------*/
void col_delete(colptr_t *col) {
  if (col->rowidx != NULL) free(col->rowidx);
  if (col->value  != NULL) free(col->value);
  col->size = 0;
  col->len  = 0;
  col->rowidx = NULL;
  col->value  = NULL;
}

/*------------------------------------------*
|  (modified) Incomplete Chol preconditioner 
 *------------------------------------------*/
int ict(csr_t *csr, double tau, int p, 
int modi, csr_t *LT, double **diag) {
/*------------------------------------------*
  A = L * L^{T}
  input:
    csr: CSR Format upper triangular matrix
      p: max # of nnz in each row
    tau: drop tolerance
   modi: modified IC
  output:
      LT: CSR Format strict of L^{T}
       d: diagonal of L^{T}
 *---------------------------------------*/
  int i,j,k,lenu,*jw,*jr,precnnz;
  double tl,*d,*w,tolnorm;
  colptr_t *col;
  int n = csr->n;
  int *ib = csr->ia;
  int *jb = csr->ja;
  double *b = csr->a;
/*--------------------------------*/
/*--------- init L^{T} */
  malloc_csr(n, 2*n, LT);
/*--------- diagonal */
  Calloc(d, n, double);
/*------ work arrays */
  Malloc(jw, n, int);  
  Malloc(jr, n, int);
  Malloc(w, n, double);
/*----- column pointers */
  Calloc(col, n, colptr_t);
/*----------------------*/
  LT->ia[0] = 1;  precnnz = 0;
/*------ Init work array jr */
  for (j=0; j<n; j++) 
    jr[j] = -1;
/*------ main loop for each row */
  for (i=0; i<n; i++) {
/*------ length of row */
    lenu = 0;
/*------ copy row i to work arrays */
    for (j=ib[i]; j<ib[i+1]; j++) {
      int jcol = jb[j-1]-1;
      assert(jcol >= i);
      if (jcol == i)  // pivot
	d[i] += b[j-1];
      else {          // U part
        lenu++;
        jr[jcol]   = i+lenu;
	jw[i+lenu] = jcol;
	w[i+lenu]  = b[j-1];
      }
    }
/*--- linear combination of prev rows */
    for (j=0; j<col[i].len; j++) {
      int jrow = col[i].rowidx[j];
      assert(jrow < i);
      /* mutiplier */      
      tl = col[i].value[j] / d[jrow];
      for (k=LT->ia[jrow]; k<LT->ia[jrow+1]; k++) {
        double s = tl * LT->a[k-1];
	/* column number */
	int jcol = LT->ja[k-1]-1;
        // lower part
        if (jcol < i) continue;
	// pivot element
        if (jcol == i) {
	  d[i] -= s;
          continue;
	}
	// U part (jcol > i), position in w
        int pos = jr[jcol];
        if (pos == -1) { // a fill-in
          lenu++;
          w[i+lenu] = -s;
          jw[i+lenu] = jcol;
          jr[jcol] = i+lenu;	      
	}
	else // not a fill-in
          w[pos] -= s;
      } /* end of for (k=uptr...*/ 
    } /* end of eliminate rows */
/*--- reset jr and compute row norm ---*/
    tolnorm = fabs(d[i]);
    for (j=0; j<lenu; j++) {
      tolnorm += fabs(w[i+1+j]);
      jr[jw[i+1+j]] = -1;
    }
    //printf("row 1-norm = %f\n", tolnorm);
    tolnorm *= tau/(double)(1+lenu);
/*----- drop small entries */
    k = 0;
    for (j=0; j<lenu; j++) {
      double vv = w[i+1+j];
      int cc = jw[i+1+j];
      if (fabs(vv) > tolnorm) {
        w[i+1+k] = vv;
	jw[i+1+k] = cc;
	k++;
      }
      else if (modi) {
        d[i]  += fabs(vv);
        d[cc] += fabs(vv);
      }
    }
    lenu = k;
    // Only keep p largest, by quick splitting
    if (lenu > p) {
      qsplitC(&w[i+1],&jw[i+1],lenu,p);
      if (modi) {
        for (j=p; j<lenu; j++) {
          double vv = w[i+1+j];
          int cc = jw[i+1+j];
          d[i]  += fabs(vv);
          d[cc] += fabs(vv);
        }
      }
      lenu = p;
    }
    if (!(d[i] > 0)) {
      printf("non-positive diag d(%d) = %e\n",\
      i, d[i]);
      return(i+1);
    }
/*--------------------insufficient memory */
    if (precnnz+lenu > LT->nnz)
      realloc_csr(LT, (int)(LT->nnz*EXPAND_FACT));
/*----------- copy w to L^{T} */
    for (j=0; j<lenu; j++) {
      int cid = jw[i+1+j];
      double val = w[i+1+j];
      LT->a[precnnz+j] = val;
      LT->ja[precnnz+j] = cid+1;
      col_insert(&col[cid], i, val);
    }
    precnnz += lenu;
    LT->ia[i+1] = precnnz+1;
    col_delete(&col[i]);
  } /* end of main loop */
/*------------------------------ */
/*---- D = D^{1/2}, U = D^{-1}*U */
  for (i=0; i<n; i++) {
    d[i] = sqrt(d[i]);
    for (j=LT->ia[i]; j<LT->ia[i+1]; j++)
      LT->a[j-1] /= d[i];
  }
  realloc_csr(LT, precnnz);
  *diag = d;
/*-----------------------*/
  free(jw);
  free(jr);
  free(w);
  free(col);
  return (0);
}

/*----------------------------------------*/
void SetupIC(matrix_t *mat, precon_t *prec,
options_t *opts) {
/*------------------------------*/
  double tol;
  int n,p, modi,lusolgpu,err;
  int nnzl,nnzu;
  csr_t *B;
/*------------------------------*/
  n = mat->n;
  Calloc(prec->ic, 1, ic_prec_t);
  Calloc(prec->ic->LT, 1, csr_t);
/*------- B: upper part in CSR */
  Calloc(B, 1, csr_t);
  GetU2(mat->h_csr, B);
/*------- IC fact */
  tol = opts->ic_opt->tol;
  p = opts->ic_opt->lfil;
  modi = opts->ic_opt->modi;
  printf("being IC(%.2e,%d)  ...\n",tol,p);
/*------ chol factor saved in D+LT */
  err = ict(B, tol, p, modi, prec->ic->LT, 
  &(prec->ic->D));
  if (err) {
    printf("  I-C Error %d\n", err);
    exit(0);
  }
/*---------------------------------*/
  int precnnz = n + prec->ic->LT->nnz;
  double fillfact = (double)precnnz/B->nnz;
  printf("  I-C ends ... fill-factor %.2f\n", 
  fillfact);
  opts->result.filfact = fillfact;
  free_csr(B);
/*-----------------------*/
  lusolgpu = opts->lusolgpu;
  if (lusolgpu == 0) {
/*------------ cpu LL^{T} solve*/
    prec->ic->h_x = 
    (double*) cuda_malloc_host(n*sizeof(double));
    prec->ic->h_b =
    (double*) cuda_malloc_host(n*sizeof(double));
    return;
  }
/*----------- gpu L/U solve */
/*----------- convert LLT to L/U */
  Calloc(prec->ic->h_lu, 1, lu_t);
  DLT2LU(prec->ic->D, prec->ic->LT, prec->ic->h_lu);
/*----------- copy LU  to device */
  Calloc(prec->ic->d_lu, 1, lu_t);
  nnzl = prec->ic->h_lu->l->nnz;
  nnzu = prec->ic->h_lu->u->nnz;
  cuda_malloc_lu(n, nnzl, nnzu, prec->ic->d_lu);
  copy_lu_h2d(prec->ic->h_lu, prec->ic->d_lu);
/*----------- level scheduling */
  Calloc(prec->ic->h_lev, 1, level_t);
  make_level(prec->ic->h_lu, prec->ic->h_lev);
  opts->result.ulev = prec->ic->h_lev->nlevU;
  opts->result.llev = prec->ic->h_lev->nlevL;
/*----------- copy level to device */
  Calloc(prec->ic->d_lev, 1, level_t);
  copy_level_h2d(n, prec->ic->h_lev, 
                    prec->ic->d_lev);
}

/*------------------------------------------*/
void DLT2LU(double *d0, csr_t *R0, lu_t *lu) {
/*------------------------------------------*
   Input: 
     d0: D^{1/2}
     R0: L^{T}
   Output:
     L U
 *------------------------------------------*/
  int n,nnz,i,j,k;
/*------------------------------------------*/
  n = R0->n;
  nnz = R0->nnz;
/*---- nnz in L is 'nnz', in U is 'nnz+n' */
  malloc_lu(n, nnz, nnz+n, lu);
/*-------- L: transposition of D^{1/2}*LT */
  csrcsc(n, n, 1, 1, R0->a, R0->ja, R0->ia,
         lu->l->a, lu->l->ja, lu->l->ia);
/*-------- L: L*D^{-1/2} */
  for (i=0; i<n; i++)
    for (j=lu->l->ia[i]; j<lu->l->ia[i+1]; j++)
      lu->l->a[j-1] /= d0[lu->l->ja[j-1]-1];
/*-------- U: 1/D + D^{1/2}*LT */
  k = 0;
  lu->u->ia[0] = 1;
  for (i=0; i<n; i++) {
/*-------- diagonal */
    lu->u->a[k] = 1/(d0[i]*d0[i]);
    lu->u->ja[k] = i+1;
    k++;
/*-------- strict upper part */
    for (j=R0->ia[i]; j<R0->ia[i+1]; j++) {
      lu->u->a[k] = d0[i]*R0->a[j-1];
      lu->u->ja[k] = R0->ja[j-1];
      k++;
    }
    lu->u->ia[i+1] = k + 1;
  }
  assert(k+1 == nnz+1+n);
}

