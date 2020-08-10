// code based on pytorch3d's dispatch.cuh

namespace {

template <
    template<int> class Kernel,
    int minN,
    int maxN,
    int curN,
    typename... Args>
struct DispatchKernelHelper1D {
  static void run(const int N, Args... args) {
    if (curN == N) {
      Kernel<curN>::run(args...);
    } else if (curN < N) {
      DispatchKernelHelper1D<Kernel, minN, maxN, curN+1, Args...>::run(
          N, args...);
    }
  }
};

template <
    template <int> class Kernel,
    int minN,
    int maxN,
    typename... Args>
struct DispatchKernelHelper1D<Kernel, minN, maxN, maxN, Args...> {
  static void run(int N, Args... args) {
    if (N == maxN) {
      Kernel<maxN>::run(args...);
    }
  }
};

}

template <
    template <int> class Kernel,
    int minN,
    int maxN,
    typename... Args>
void DispatchKernel1D(int N, Args... args) {
  if (minN <= N && N <= maxN) {
    DispatchKernelHelper1D<Kernel, minN, maxN, minN, Args...>::run(
        N, args...);
  }
  else {
    std::cout << "K should be in this range [" << minN << ", " << maxN << "]" << std::endl;
    throw;
  }
}