#include "matrix_convert.h"
#include "comprow_double.h"
#include "matrix_reader.h"

namespace gs {
  void import_to_comprow(CompRow_Mat_double &dst, const MatrixImporter &src)
  {
    new (&dst) CompRow_Mat_double(src.num_rows(), src.num_cols(), src.num_nnzs(),
				  src.get_coefs(), src.get_row_ptrs(), src.get_cols());
  }  
}
