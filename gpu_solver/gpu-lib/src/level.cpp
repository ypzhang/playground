#include "gpusollib.h"

/*-----------------------------------------*/
void make_level(lu_t *h_lu, level_t *h_lev) {
  int i,j,l,n,*level;
  n = h_lu->l->n;
  Malloc(level, n, int);
/*----- L */
  h_lev->nlevL = 0;
  Calloc(h_lev->jlevL, n, int);
/*----- at most n level */
  Calloc(h_lev->ilevL, n+1, int);
  h_lev->ilevL[0] = 1;

  for (i=0; i<n; i++) {
    l = 0;
    for(j=h_lu->l->ia[i]; j<h_lu->l->ia[i+1]; j++)
      l = max(l, level[h_lu->l->ja[j-1]-1]);
    level[i] = l+1;
    h_lev->ilevL[l+1] ++;
    h_lev->nlevL = max(h_lev->nlevL, l+1);
  }

  for (i=1; i<=h_lev->nlevL; i++)
    h_lev->ilevL[i] += h_lev->ilevL[i-1];

  for (i=0; i<n; i++) {
    int *k = &h_lev->ilevL[level[i]-1];
    h_lev->jlevL[(*k)-1] = i+1;
    (*k)++;
  }

  for (i=h_lev->nlevL-1; i>0; i--)
    h_lev->ilevL[i] = h_lev->ilevL[i-1];

  h_lev->ilevL[0] = 1;
/*----- U */
  h_lev->nlevU = 0;
  Malloc(h_lev->jlevU, n, int);
  Calloc(h_lev->ilevU, n+1, int);

  h_lev->ilevU[0] = 1;
  for (i=n-1; i>=0; i--) {
    l = 0;
    for (j=h_lu->u->ia[i]+1; j<h_lu->u->ia[i+1]; j++)
      l = max(l, level[h_lu->u->ja[j-1]-1]);

    level[i] = l+1;
    h_lev->ilevU[l+1] ++;
    h_lev->nlevU = max(h_lev->nlevU, l+1);
  }

  for (i=1; i<=h_lev->nlevU; i++)
    h_lev->ilevU[i] += h_lev->ilevU[i-1];

  for (i=0; i<n; i++) {
    int *k = &h_lev->ilevU[level[i]-1];
    h_lev->jlevU[(*k)-1] = i+1;
    (*k)++;
  }

  for (i=h_lev->nlevU-1; i>0; i--)
    h_lev->ilevU[i] = h_lev->ilevU[i-1];

  h_lev->ilevU[0] = 1;

  free(level);
}

/*-----------------------------------------*/
void copy_level_h2d(int n, level_t *h_lev, 
                    level_t *d_lev) {
  //printf("L nlev = %d\n", h_lev->nlevL);
  //printf("U nlev = %d\n", h_lev->nlevU);
/*--------- copy level info to device */
  d_lev->nlevL = h_lev->nlevL;
  d_lev->nlevU = h_lev->nlevU;
  d_lev->jlevL = (int*)cuda_malloc(n*sizeof(int));
  d_lev->ilevL = 
  (int*)cuda_malloc((h_lev->nlevL+1)*sizeof(int));
  d_lev->jlevU = (int*)cuda_malloc(n*sizeof(int));
  d_lev->ilevU = 
  (int*)cuda_malloc((h_lev->nlevU+1)*sizeof(int));
/*---------------------------------------------*/
  memcpyh2d(d_lev->jlevL, h_lev->jlevL, 
  n*sizeof(int));
  memcpyh2d(d_lev->ilevL, h_lev->ilevL,
  (h_lev->nlevL+1)*sizeof(int));
  memcpyh2d(d_lev->jlevU, h_lev->jlevU, 
  n*sizeof(int));
  memcpyh2d(d_lev->ilevU, h_lev->ilevU,
  (h_lev->nlevU+1)*sizeof(int));
}

