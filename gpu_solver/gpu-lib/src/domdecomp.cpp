#include "gpusollib.h"

/*---------------------------*/
void map2perm(int n, int pnum,
int *map, int *noff, int *perm) {
/*---------------------------*/
  int i,j,p;
/*---------------------------*/
  for (i=0; i<n; i++) {
/*--- partition of this node */
    p = map[i] - 1;
/*--- first pos of this part */
    j = noff[p];
/*--- node i goto position j */
    perm[i] = j+1;
    noff[p] ++;
  }
/*---------------------------*/
  for (i=pnum-1; i>0; i--)
    noff[i] = noff[i-1];
  noff[0] = 0;
}

/*-------------------------------------*/
/*  Domain-Decomposition & reordering  */
/*-------------------------------------*/
void reorder_dd(csr_t *A, precon_t *prec,
options_t *opts, int **perm) {
/*-------------------------------------*/
  PREC_TYPE ptype;
  int i,n,*map,pnum,*nrow,*noff;
  csr_t *C;
  double t1,t2;
/*-------------------------------------*/
  *perm = NULL;
  ptype = opts->prectype;
  if (ptype != BILUK && ptype != BILUT)
    return;
  assert(opts->bilu_opt);
  if (opts->bilu_opt->dd == 0)
    return;
/*------------------------------------*/
  n = A->n;
  pnum = opts->bilu_opt->bn;
/*----------------- symmetrize A to C */
/*------ only the pattern(ia,ja) in C */
  Calloc(C, 1, csr_t);
  symmgraph(A, C);
  remove_diag(C);
/*-------- METIS parameters*/
  int wtflag  = 0;
  int numflag = 1;
  int option  = 0;
  int edgecut;
  Calloc(map, n, int);
/*------------------------ call METIS */
  t1 = wall_timer();
  METIS_PartGraphKway(&n, C->ia, C->ja, 
  NULL, NULL, &wtflag, &numflag, 
  &pnum, &option, &edgecut, map);
  t2 = wall_timer()-t1;
  //printf("METIS Time: %lf\n", t2);
  opts->result.tm_dd = t2;
  free(C->ia); free(C->ja); free(C);
/*----- num of nodes of each partition */
  Calloc(nrow, pnum, int);
  for (i=0; i<n; i++)
    nrow[map[i]-1] ++;
/*----- row offset of each partition */
  Malloc(noff, pnum, int);
  noff[0] = 0;
  for (i=1; i<pnum; i++)
    noff[i] = noff[i-1] + nrow[i-1];
/*----- permutaion array */
  Malloc(*perm, n, int);
  map2perm(n, pnum, map, noff, *perm);
/*----------------------------------*/
  Calloc(prec->bilu, 1, bilu_prec_t);
  Calloc(prec->bilu->host, 1, bilu_host_t);
  prec->bilu->host->nrow = nrow;
  prec->bilu->host->noff = noff;
/*----------------------------------*/
  free(map);
}

