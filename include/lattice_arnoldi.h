#ifndef _LATTICE_ARNOLDI_H
#define _LATTICE_ARNOLDI_H
#include "./arnoldi.h"
#include "./lattice_bistabcg.h"
namespace qcu
{
  template <typename T>
  struct LatticeWilsonMultiplier
  {
    int size, if_input;
    Eigen::Matrix<std::complex<T>, Eigen::Dynamic, Eigen::Dynamic> _matrix;
    LatticeBistabCg<T> *bistacg_ptr;
    void give(int _size)
    {
      auto start = std::chrono::high_resolution_clock::now();
      if_input = 0;
      size = _size;
      {
        {
          _matrix = Eigen::Matrix<std::complex<T>, Eigen::Dynamic, Eigen::Dynamic>::Random(size, size);
        }
        // {
        //     Eigen::Matrix<std::complex<T>, Eigen::Dynamic, 1> eigenvalues = Eigen::Matrix<std::complex<T>, Eigen::Dynamic, 1>::Random(size);
        //     Eigen::Matrix<std::complex<T>, Eigen::Dynamic, Eigen::Dynamic> V = Eigen::Matrix<std::complex<T>, Eigen::Dynamic, Eigen::Dynamic>::Random(size, size);
        //     // {
        //     //     Eigen::Matrix<std::complex<T>, Eigen::Dynamic, 1> eigenvalues(size);
        //     //     for (int i = 0; i < size; i++)
        //     //     {
        //     //         double real = cos(2.0 * M_PI * i / size);
        //     //         double imag = sin(2.0 * M_PI * i / size);
        //     //         eigenvalues(i) = complex<double>(real, imag);
        //     //     }
        //     // }
        //     Eigen::HouseholderQR<Eigen::Matrix<std::complex<T>, Eigen::Dynamic, Eigen::Dynamic>> qr(V);
        //     Eigen::Matrix<std::complex<T>, Eigen::Dynamic, Eigen::Dynamic> Q = qr.householderQ();
        //     _matrix = Q * eigenvalues.asDiagonal() * Q.adjoint();
        // }
      }
      auto end = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
      std::cout << "Give _matrix time: " << duration.count() << " milliseconds" << std::endl;
    }
    void give(LatticeBistabCg<T> *_bistacg_ptr)
    {
      auto start = std::chrono::high_resolution_clock::now();
      if_input = 1;
      size = _bistacg_ptr->set_ptr->lat_4dim_SC;
      bistacg_ptr = _bistacg_ptr;
      auto end = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
      std::cout << "Give lattice multiplier time: " << duration.count() << " milliseconds" << std::endl;
    }
    Eigen::Matrix<std::complex<T>, Eigen::Dynamic, 1> operator*(const Eigen::Matrix<std::complex<T>, Eigen::Dynamic, 1> &src) const
    {
      if (if_input == 0)
      {
        return _matrix * src;
      }
      else
      {
        Eigen::Matrix<std::complex<T>, Eigen::Dynamic, 1> dst = Eigen::Matrix<std::complex<T>, Eigen::Dynamic, 1>::Zero(size);
        checkCudaErrors(cudaStreamSynchronize(bistacg_ptr->set_ptr->stream));
        checkCudaErrors(cudaMemcpyAsync(bistacg_ptr->x_o, src.data(), size * sizeof(T) * _REAL_IMAG_, cudaMemcpyHostToDevice, bistacg_ptr->set_ptr->stream));
        checkCudaErrors(cudaStreamSynchronize(bistacg_ptr->set_ptr->stream));
        bistacg_ptr->_wilson_dslash(bistacg_ptr->r, bistacg_ptr->x_o, bistacg_ptr->gauge); // Dslash
        checkCudaErrors(cudaStreamSynchronize(bistacg_ptr->set_ptr->stream));
        checkCudaErrors(cudaMemcpyAsync(dst.data(), bistacg_ptr->r, size * sizeof(T) * _REAL_IMAG_, cudaMemcpyDeviceToHost, bistacg_ptr->set_ptr->stream));
        checkCudaErrors(cudaStreamSynchronize(bistacg_ptr->set_ptr->stream));
        return dst;
      }
    }
  };
  template <typename T>
  struct LatticeArnoldi
  {
    void *Q, *H, *gauge;
    LatticeBistabCg<T> *bistacg_ptr;
    LatticeWilsonMultiplier<T> lattice_multiplier;
    int m, n, max_rest, if_input;
    T tol;
    void give(LatticeBistabCg<T> *_bistacg_ptr)
    {
      if_input = 1;
      bistacg_ptr = _bistacg_ptr;
      lattice_multiplier.give(bistacg_ptr);
      m = bistacg_ptr->set_ptr->krylov_size();
      n = bistacg_ptr->set_ptr->lat_4dim_SC;
      max_rest = bistacg_ptr->set_ptr->max_rest();
      tol = bistacg_ptr->set_ptr->tol();
    }
    void give()
    {
      if_input = 0;
      m = 48;
      n = 200;
      lattice_multiplier.give(n);
      max_rest = 30;
      tol = 1e-9;
    }
    std::complex<T> computeRayleighQuotient(const Eigen::Matrix<std::complex<T>, Eigen::Dynamic, 1> &v)
    {
      Eigen::Matrix<std::complex<T>, Eigen::Dynamic, 1> Av = lattice_multiplier * v;
      return (v.adjoint() * Av)(0, 0) / (v.adjoint() * v)(0, 0);
    }
    void arnoldiIteration(Eigen::Matrix<std::complex<T>, Eigen::Dynamic, Eigen::Dynamic> &V,
                          Eigen::Matrix<std::complex<T>, Eigen::Dynamic, Eigen::Dynamic> &H, int k_start, int k_end)
    {
      auto start = std::chrono::high_resolution_clock::now();
      for (int j = k_start; j < k_end; j++)
      {
        Eigen::Matrix<std::complex<T>, Eigen::Dynamic, 1> w = lattice_multiplier * V.col(j);
        for (int i = 0; i <= j; i++)
        {
          std::complex<T> h = std::complex<T>(V.col(i).adjoint() * w);
          H(i, j) = h;
          w -= h * V.col(i);
          std::complex<T> h2 = std::complex<T>(V.col(i).adjoint() * w);
          H(i, j) += h2;
          w -= h2 * V.col(i);
        }
        T norm_w = w.norm();
        if (norm_w < tol || j == k_end - 1)
        {
          H.col(j).setZero();
          V.col(j + 1).setZero();
          std::cout << "Arnoldi iteration converged at iteration " << j << std::endl;
          std::cout << "Arnoldi iteration norm_w: " << norm_w << std::endl;
          break;
        }
        H(j + 1, j) = norm_w;
        V.col(j + 1) = w / norm_w;
      }
      auto end = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
      std::cout << "Arnoldi iteration time: " << duration.count() << " milliseconds" << std::endl;
    }
    Eigen::Matrix<std::complex<T>, Eigen::Dynamic, 1> refineSingleVector(std::complex<T> &eigenvalue,
                                                                         const Eigen::Matrix<std::complex<T>, Eigen::Dynamic, 1> &initial_vector,
                                                                         int max_iter = 15)
    {
      auto start = std::chrono::high_resolution_clock::now();
      Eigen::Matrix<std::complex<T>, Eigen::Dynamic, 1> current_vector = initial_vector;
      std::complex<T> prev_eigenvalue = eigenvalue;
      for (int iter = 0; iter < max_iter; ++iter)
      {
        Eigen::Matrix<std::complex<T>, Eigen::Dynamic, 1> new_vector = lattice_multiplier * current_vector;
        eigenvalue = computeRayleighQuotient(current_vector);
        new_vector.normalize();
        T diff = (new_vector - current_vector).norm();
        current_vector = new_vector;
        if (diff < tol && std::abs(eigenvalue - prev_eigenvalue) < tol * 1e-6)
        {
          break;
        }
        prev_eigenvalue = eigenvalue;
      }
      auto end = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
      std::cout << "Refining vector time: " << duration.count() << " milliseconds" << std::endl;
      return current_vector;
    }
    void _run(std::vector<std::complex<T>> &eigenvalues, Eigen::Matrix<std::complex<T>, Eigen::Dynamic, Eigen::Dynamic> &eigenvectors)
    {
      auto start = std::chrono::high_resolution_clock::now();
      Eigen::Matrix<std::complex<T>, Eigen::Dynamic, Eigen::Dynamic> V = Eigen::Matrix<std::complex<T>, Eigen::Dynamic, Eigen::Dynamic>::Zero(n, m + 1);
      Eigen::Matrix<std::complex<T>, Eigen::Dynamic, Eigen::Dynamic> H = Eigen::Matrix<std::complex<T>, Eigen::Dynamic, Eigen::Dynamic>::Zero(m + 1, m);
      Eigen::Matrix<std::complex<T>, Eigen::Dynamic, 1> initial = Eigen::Matrix<std::complex<T>, Eigen::Dynamic, 1>::Random(n);
      initial.normalize();
      V.col(0) = initial;
      for (int restart = 0; restart < max_rest; restart++)
      {
        arnoldiIteration(V, H, 0, m);
        Eigen::Matrix<std::complex<T>, Eigen::Dynamic, Eigen::Dynamic> H_m = H.topLeftCorner(m, m);
        Eigen::ComplexEigenSolver<Eigen::Matrix<std::complex<T>, Eigen::Dynamic, Eigen::Dynamic>> ces(H_m);
        std::vector<std::pair<T, int>> eig_order(m);
        for (int i = 0; i < m; i++)
        {
          eig_order[i] = std::make_pair(std::abs(ces.eigenvalues()(i)), i);
        }
        std::sort(eig_order.begin(), eig_order.end(), std::greater<std::pair<T, int>>());
        bool converged = true;
        for (int i = 0; i < m / 2; i++)
        {
          Eigen::Matrix<std::complex<T>, Eigen::Dynamic, 1> ritz_vector = V.leftCols(m) * ces.eigenvectors().col(eig_order[i].second);
          Eigen::Matrix<std::complex<T>, Eigen::Dynamic, 1> residual = lattice_multiplier * ritz_vector -
                                                                       ces.eigenvalues()(eig_order[i].second) * ritz_vector;
          if (residual.norm() > tol)
          {
            converged = false;
            break;
          }
        }
        if (converged)
          break;
        Eigen::Matrix<std::complex<T>, Eigen::Dynamic, Eigen::Dynamic> Q = ces.eigenvectors();
        V.leftCols(m) = V.leftCols(m) * Q;
        H.setZero();
      }
      Eigen::Matrix<std::complex<T>, Eigen::Dynamic, Eigen::Dynamic> H_m = H.topLeftCorner(m, m);
      Eigen::ComplexEigenSolver<Eigen::Matrix<std::complex<T>, Eigen::Dynamic, Eigen::Dynamic>> ces(H_m);
      eigenvalues.resize(m / 2);
      eigenvectors = Eigen::Matrix<std::complex<T>, Eigen::Dynamic, Eigen::Dynamic>::Zero(n, m / 2);
      std::vector<std::pair<T, int>> eig_order(m);
      for (int i = 0; i < m; i++)
      {
        eig_order[i] = std::make_pair(std::abs(ces.eigenvalues()(i)), i);
      }
      std::sort(eig_order.begin(), eig_order.end(), std::greater<std::pair<T, int>>());
      for (int i = 0; i < m / 2; i++)
      {
        int idx = eig_order[i].second;
        std::complex<T> eigenvalue = ces.eigenvalues()(idx);
        Eigen::Matrix<std::complex<T>, Eigen::Dynamic, 1> eigenvector = V.leftCols(m) * ces.eigenvectors().col(idx);
        eigenvector.normalize();
        eigenvectors.col(i) = refineSingleVector(eigenvalue, eigenvector);
        eigenvalues[i] = eigenvalue;
      }
      auto end = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
      std::cout << "Eigenvalue computation time: " << duration.count() << " milliseconds" << std::endl;
    }
    void verify_results(const std::vector<std::complex<T>> &computed_eigenvalues,
                        const Eigen::Matrix<std::complex<T>, Eigen::Dynamic, Eigen::Dynamic> &computed_eigenvectors)
    {
      auto start = std::chrono::high_resolution_clock::now();
      std::cout << "Verification Results:" << std::endl;
      std::vector<std::pair<T, int>> errors(computed_eigenvalues.size());
      for (size_t i = 0; i < computed_eigenvalues.size(); i++)
      {
        Eigen::Matrix<std::complex<T>, Eigen::Dynamic, 1> eigenvector = computed_eigenvectors.col(i);
        Eigen::Matrix<std::complex<T>, Eigen::Dynamic, 1> Av = lattice_multiplier * eigenvector;
        Eigen::Matrix<std::complex<T>, Eigen::Dynamic, 1> lambda_v = computed_eigenvalues[i] * eigenvector;
        T error = (Av - lambda_v).norm() / eigenvector.norm();
        errors[i] = std::make_pair(error, i);
        std::cout << "Eigenvalue " << i + 1 << ": " << computed_eigenvalues[i] << std::endl;
        std::cout << "Relative Error: " << error << std::endl;
        std::cout << "-------------------" << std::endl;
      }
      std::sort(errors.begin(), errors.end());
      std::cout << "\nBest 24 Results:" << std::endl;
      for (int i = 0; i < std::min(24, (int)errors.size()); ++i)
      {
        std::cout << "Eigenvalue " << errors[i].second + 1
                  << " Relative Error: " << errors[i].first << std::endl;
      }
      auto end = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
      std::cout << "Verification time: " << duration.count() << " milliseconds" << std::endl;
    }
    void run()
    {
      std::vector<std::complex<T>> eigenvalues;
      Eigen::Matrix<std::complex<T>, Eigen::Dynamic, Eigen::Dynamic> eigenvectors;
      _run(eigenvalues, eigenvectors);
    }
    void run_test()
    {
      auto start = std::chrono::high_resolution_clock::now();
      std::vector<std::complex<T>> eigenvalues;
      Eigen::Matrix<std::complex<T>, Eigen::Dynamic, Eigen::Dynamic> eigenvectors;
      _run(eigenvalues, eigenvectors);
      verify_results(eigenvalues, eigenvectors);
      auto end = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
      std::cout << eigenvalues.size() << std::endl;
      std::cout << eigenvectors.size() << std::endl;
      std::cout << "Total execution time: " << duration.count() << " milliseconds" << std::endl;
    }
  };
}
#endif
