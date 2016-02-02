#include <chrono>
#include "utility.h"

using namespace std;
using namespace std::chrono;

namespace gs {

//typedef std::chrono::high_resolution_clock jusha_perf_clock;
typedef std::chrono::system_clock gs_perf_clock;

gs_perf_clock::time_point gs_g_world_start = gs_perf_clock::now();

double get_cur_time()
{
  auto start = gs_perf_clock::now();
  std::chrono::microseconds elapse_since_world_start = duration_cast<microseconds>(start - gs_g_world_start);
  //  cout << "count  "<<elapse_since_world_start.count() << std::endl;
  return double(elapse_since_world_start.count()) / 1000000.0;
}

}
