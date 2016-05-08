int gpusol_init();
int gpusol_finalize();
double wall_timer();
void err_norm(int,double*, double*,result_t*);
void resd_norm(csr_t*,double*, double*,result_t*);
void read_input(char* fn, options_t *opts);
void output_result(matrix_t*,options_t *opts);
void COO2CSR(coo_t *coo, csr_t *csr);
void PadJAD32(jad_t *jad);
void CSR2JAD(csr_t *csr, jad_t *jad);
int CSR2DIA(csr_t *csr, dia_t *dia);

void csrcsc(int n, int n2, int job, int ipos, 
            double *a, int *ja, int *ia, 
	    double *ao, int *jao, int *iao);

void sortrow(csr_t*);

void setup_matrix(matrix_t *mat, options_t *opts);

void fgmres(matrix_t*, precon_t*, options_t*,
double*, double*);

int pcg(matrix_t*, precon_t*, options_t*, 
double *, double *);

int qsplitC(double *a, int *ind, int n, int ncut);

int ilut(csr_t *csr, lu_t *lu, double tol, int p);

int iluk(csr_t *csmat, lu_t *lu, int lfil);

void SetupILUt(matrix_t*, precon_t*, options_t*);

void SetupILUK(matrix_t*, precon_t*, options_t*);

void SetupMCSOR(matrix_t*, precon_t*, options_t*);

void SetupMCILU0(matrix_t*, precon_t*,options_t*);

void SetupBILUK(matrix_t*, precon_t*,options_t*);

void SetupBILUT(matrix_t*, precon_t*,options_t*);

void SetupIC(matrix_t*, precon_t*,options_t*);

void SetupLSPOLY(matrix_t*,precon_t*,options_t*);

void DLT2LU(double *d0, csr_t *R0, lu_t *lu);

int read_coo_MM(coo_t *coo, options_t *opts);

void make_level(lu_t *h_lu, level_t *h_lev);

void copy_level_h2d(int, level_t*, level_t*);

void *cuda_malloc(int size);

void *cuda_malloc_host(int size);

void memcpyh2d(void *dest, void* src, int size);

void memcpyd2d(void *dest, void* src, int size);

void memcpyd2h(void *dest, void *src, int size);

void cuda_memset(void *, int, int);

void malloc_csr(int n, int nnz, csr_t *csr);
void realloc_csr(csr_t *csr, int nnz);
void cuda_malloc_csr(int n, int nnz, csr_t *d_csr);
void copy_csr_h2d(csr_t *h_csr, csr_t *d_csr);
void copy_csr_h2h(csr_t *csr1, csr_t *csr2);
void cuda_malloc_jad(int n, int njad, 
int nnz, jad_t *d_jad);
void copy_jad_h2d(jad_t *h_jad, jad_t *d_jad);
void cuda_malloc_dia(int nd, int strd, 
                     dia_t *d_dia);
void copy_dia_h2d(dia_t *h_dia, dia_t *d_dia);
void malloc_lu(int n, int nnzl, int nnzu, lu_t *lu);
void cuda_malloc_lu(int n, int nnzl, int nnzu, lu_t*);
void copy_lu_h2d(lu_t *h_lu, lu_t *d_lu);
void Free(void *p);
void cuda_free(void *p);
void cuda_free_host(void *p);
void free_coo(coo_t *coo);
void free_csr(csr_t *csr);
void cuda_free_csr(csr_t *d_csr);
void free_jad(jad_t *jad);
void cuda_free_jad(jad_t *d_jad);
void free_dia(dia_t *dia);
void cuda_free_dia(dia_t *d_dia);
void free_lu(lu_t *h_lu);
void cuda_free_lu(lu_t *d_lu);
void free_lev(level_t *h_lev);
void cuda_free_lev(level_t *d_lev);
void free_matrix(matrix_t *mat);
void free_ilu(ilu_prec_t *ilu);
void free_precon(precon_t *prec);
void free_opts(options_t *opts);
void diag_scal(csr_t*, double*, 
options_t*, double**);
void scal_vec(int,double*, double *);
void pitsol(matrix_t *mat, precon_t *prec,
options_t *opts, double *d_x, double *d_b);
void setup_precon(matrix_t*, precon_t*, options_t*);
void reorder_rcm(csr_t *A, int *perm);
void reorder_nd(csr_t *A, int *perm);
void reorder_mmd(csr_t *A, int *perm);
void reorder_metis(csr_t*, int, int*);
void reorder_sym(csr_t *A, options_t *opts, int **);
void reorder_mc(csr_t*,precon_t*,options_t*,int**);
void reorder_dd(csr_t*,precon_t*,options_t*, int**);
void perm_mat_sym(csr_t *A, int *perm);
void perm_vec(int n, double *x, int *perm);
void iperm_vec(int n, double *x, int *perm);
void reorder_mc(csr_t *A, options_t *opts,
                int **perm);
void spmv_csr_vector(matrix_t *mat, double *x, 
                     double *y, int neg);
void spmv_jad(matrix_t *mat, double *x, 
              double *y, int neg);

void spmv_dia(matrix_t *mat, double *x, 
double *y, int neg);

void spmv_csr_cpu(csr_t *, double *, double*);

void spmv_ilu0_1(int, int, int*, int*, 
double*, double*);

void spmv_ilu0_2(int, int, int*, int*, 
double*, double*, double*);

void lusol(matrix_t *mat, precon_t *prec, 
options_t *opts, double *d_x, double *d_b);

void lltsol(matrix_t*, precon_t*, 
options_t*, double*, double*);

void blusol(matrix_t*, precon_t*, 
options_t*, double*, double*);

void ssor(matrix_t *mat, precon_t *prec, 
options_t *opts, double *d_y, double *d_x);

void mcilu0op(matrix_t*, precon_t*, 
options_t*,double*, double*);

void polyapprox(matrix_t*, precon_t*, 
options_t*, double *, double *);

void noprecop(matrix_t*, precon_t*,
options_t*, double*, double*);

void dump_mat_coo(csr_t *A, char*);

void check_mc(csr_t *A, int ncol, int *il);

void mulcol(csr_t *A, options_t *opts, 
precon_t *prec, int *perm);

void mulcol_sp(csr_t *A, options_t *opts,
precon_t *prec, int *perm);

void symmgraph(csr_t *A, csr_t *C);

void remove_diag(csr_t *);

void Filter(csr_t *A, csr_t *B, double drptol);

void Filter2(csr_t *A, double drptol, int ncol);

void spmv_sor(int n, int nrow, int *d_ia, int *d_ja, 
double *d_a, double *d_y, double *d_x, double *d_w);

void test_mcilu0(matrix_t *mat, precon_t *prec, 
options_t *opts, csr_t*, csr_t*,double*);

void mcprune_lu(lu_t*, mcilu0_prec_t*);

void cuda_check_err();

csr_t* mcprune_csr(csr_t*, mcilu0_prec_t*);

void blu_h2d(bilu_host_t*, bilu_dev_t*, int, int);
extern "C" void coocsr_(int *, int *, double *,
int *, int *, double *, int *, int *);

void Partition1D(int len, int pnum, int idx, 
                 int &j1, int &j2);

void DLT2LU(double*, csr_t *, lu_t *);

void lanczos(matrix_t *mat, int msteps, 
             double *maxev, double *minev);

void lspol(int deg, int nintv, double *intv,
double *alpha, double *beta, double *gamma);

extern "C" void csrjad_(int *, double *, 
int *, int *, int *, int *, double *,
int *, int *);

extern "C" void multic_(int*, int*, int*, 
int*, int*, int*,int*, int*, int*);

extern "C" void aplb_(int*, int*, int*, 
double*, int*, int*,
double*, int*, int*, double*, int*, int*,
int*, int*, int*);

extern "C" void filter_(int*, int*, double*, 
double*, int*, int*, double*, 
int*, int*, int*, int*);

extern "C" void dperm_(int*, double *, int *, 
int*, double *,int *, int *, int *, int *, int*);

extern "C" void vperm_(int*, double *, int*);

extern "C" void getu_(int*,double*,int*,
int*,double*,int*,int*);

extern "C" void submat_(int*, int*, int*, 
int*, int*, int*, double*, int*, 
int*, int*, int*, double*, int*, int*);

extern "C" void csrdia_(int*, int*, int*, 
double*, int*, int*, int*, double *, 
int*, double*, int *, int*, int*);

extern "C" void METIS_PartGraphKway
(int*, int*, int*, int*, int*, 
int*, int*, int*, int*, int*, int*);

extern "C" void METIS_NodeND
(int*, int*, int*, int*, int*, int*, int*);

extern "C" void dsteqr_
(char*, int*, double*, double*, double*, int*, 
double*, int*);

/*----------------------------------------*/

#define STEQR dsteqr_
#define GGEVX dggevx_
#define CUDOT cublasDdot
#define CUAXPY cublasDaxpy
#define CUSCAL cublasDscal
#define CUNRM2 cublasDnrm2
#define CUGEMV cublasDgemv
#define CUGEMM cublasDgemm

#define CHECKERR(ierr) assert(!(ierr))
#define Malloc(base, nmem, type) {\
  (base) = (type *)malloc((nmem)*sizeof(type)); \
  CHECKERR((base) == NULL); \
}
#define Calloc(base, nmem, type) {\
  (base) = (type *)calloc((nmem), sizeof(type)); \
  CHECKERR((base) == NULL); \
}
#define Realloc(base, nmem, type) {\
  (base) = (type *)realloc((base), (nmem)*sizeof(type)); \
  CHECKERR((base) == NULL && nmem > 0); \
}

#define CUDA_SAFE_CALL_NO_SYNC( call) {                               \
    cudaError err = call;                                             \
    if( cudaSuccess != err) {                                         \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", \
                __FILE__, __LINE__, cudaGetErrorString( err) );       \
        exit(EXIT_FAILURE);                                           \
    } }
#define CUDA_SAFE_CALL( call)    CUDA_SAFE_CALL_NO_SYNC(call);

#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))

