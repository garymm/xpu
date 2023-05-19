#include "util.hpp"
using namespace std;

void print_elapsed(chrono::time_point<chrono::system_clock> *start, const char *description) {
  const auto end = chrono::system_clock::now();
  cout << description << ": ";
  const auto elapsed_mus = chrono::duration_cast<chrono::microseconds>(end - *start).count();
  if (elapsed_mus < 1000) {
    cout << elapsed_mus << "µs\n";
  } else {
    cout << elapsed_mus / 1000 << "ms\n";
  }
  *start = end;
}
