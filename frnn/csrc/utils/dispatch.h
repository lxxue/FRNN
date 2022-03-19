// code based on pytorch3d's dispatch.cuh
#pragma once
#include <iostream>

namespace {

template <template <int> class Kernel, int minN, int maxN, int curN,
          typename... Args>
struct DispatchKernelHelper1D {
  static void run(const int N, Args... args) {
    if (curN == N) {
      Kernel<curN>::run(args...);
    } else if (curN < N) {
      DispatchKernelHelper1D<Kernel, minN, maxN, curN + 1, Args...>::run(
          N, args...);
    }
  }
};

template <template <int> class Kernel, int minN, int maxN, typename... Args>
struct DispatchKernelHelper1D<Kernel, minN, maxN, maxN, Args...> {
  static void run(int N, Args... args) {
    if (N == maxN) {
      Kernel<maxN>::run(args...);
    }
  }
};

template <template <int, int> class Kernel, int minN, int maxN, int curN,
          int minM, int maxM, int curM, typename... Args>
struct DispatchKernelHelper2D {
  static void run(const int N, const int M, Args... args) {
    if (curN == N && curM == M) {
      Kernel<curN, curM>::run(args...);
    } else if (curN < N && curM < M) {
      // Increment both curN and curM. This isn't strictly necessary; we could
      // just increment one or the other at each step. But this helps to cut
      // on the number of recursive calls we make.
      DispatchKernelHelper2D<Kernel, minN, maxN, curN + 1, minM, maxM, curM + 1,
                             Args...>::run(N, M, args...);
    } else if (curN < N) {
      // Increment curN only
      DispatchKernelHelper2D<Kernel, minN, maxN, curN + 1, minM, maxM, curM,
                             Args...>::run(N, M, args...);
    } else if (curM < M) {
      // Increment curM only
      DispatchKernelHelper2D<Kernel, minN, maxN, curN, minM, maxM, curM + 1,
                             Args...>::run(N, M, args...);
    }
  }
};

// 2D dispatch, specialization for curN == maxN
template <template <int, int> class Kernel, int minN, int maxN, int minM,
          int maxM, int curM, typename... Args>
struct DispatchKernelHelper2D<Kernel, minN, maxN, maxN, minM, maxM, curM,
                              Args...> {
  static void run(const int N, const int M, Args... args) {
    if (maxN == N && curM == M) {
      Kernel<maxN, curM>::run(args...);
    } else if (curM < maxM) {
      DispatchKernelHelper2D<Kernel, minN, maxN, maxN, minM, maxM, curM + 1,
                             Args...>::run(N, M, args...);
    }
    // We should not get here -- throw an error?
  }
};

// 2D dispatch, specialization for curM == maxM
template <template <int, int> class Kernel, int minN, int maxN, int curN,
          int minM, int maxM, typename... Args>
struct DispatchKernelHelper2D<Kernel, minN, maxN, curN, minM, maxM, maxM,
                              Args...> {
  static void run(const int N, const int M, Args... args) {
    if (curN == N && maxM == M) {
      Kernel<curN, maxM>::run(args...);
    } else if (curN < maxN) {
      DispatchKernelHelper2D<Kernel, minN, maxN, curN + 1, minM, maxM, maxM,
                             Args...>::run(N, M, args...);
    }
    // We should not get here -- throw an error?
  }
};

// 2D dispatch, specialization for curN == maxN, curM == maxM
template <template <int, int> class Kernel, int minN, int maxN, int minM,
          int maxM, typename... Args>
struct DispatchKernelHelper2D<Kernel, minN, maxN, maxN, minM, maxM, maxM,
                              Args...> {
  static void run(const int N, const int M, Args... args) {
    if (maxN == N && maxM == M) {
      Kernel<maxN, maxM>::run(args...);
    }
    // We should not get here -- throw an error?
  }
};
}  // namespace

template <template <int> class Kernel, int minN, int maxN, typename... Args>
void DispatchKernel1D(int N, Args... args) {
  if (minN <= N && N <= maxN) {
    DispatchKernelHelper1D<Kernel, minN, maxN, minN, Args...>::run(N, args...);
  } else {
    std::cout << "K should be in this range [" << minN << ", " << maxN << "]"
              << std::endl;
    throw;
  }
}

template <template <int, int> class Kernel, int minN, int maxN, int minM,
          int maxM, typename... Args>
void DispatchKernel2D(const int N, const int M, Args... args) {
  if (minN <= N && N <= maxN && minM <= M && M <= maxM) {
    // Kick off the template recursion by calling the Helper with curN = minN
    // and curM = minM
    DispatchKernelHelper2D<Kernel, minN, maxN, minN, minM, maxM, minM,
                           Args...>::run(N, M, args...);
  }
  // Maybe throw an error if we tried to dispatch outside the specified range?
}
