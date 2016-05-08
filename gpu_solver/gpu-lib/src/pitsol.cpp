#include "gpusollib.h"

void pitsol(matrix_t *mat, precon_t *prec,
options_t *opts, double *d_x, double *d_b) {
/*---------------------------------------*/
  double t1,t2;
/*---------------------------------------*/
  t1 = wall_timer();
  switch (opts->solver) {
    case GMRES:
      fgmres(mat, prec, opts, d_x, d_b);
    break;
    case CG:
      pcg(mat, prec, opts, d_x, d_b);
    break;
  }
  t2 = wall_timer();
  opts->result.tm_iter = t2-t1;
}

