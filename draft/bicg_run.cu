void run(void *gauge) {
    for (int loop = 0; loop < _MAX_ITER_; loop++) {
      dot(r_tilde, r, _rho_, _a_);
      checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_b_]));
      {
        // beta = (rho / rho_prev) * (alpha / omega);
        bistabcg_give_1beta<<<1, 1, 0, set_ptr->streams[_a_]>>>(device_vals);
      }
      {
        // rho_prev = rho;
        bistabcg_give_1rho_prev<<<1, 1, 0, set_ptr->streams[_b_]>>>(
            device_vals);
      }
      {
        // p[i] = r[i] + (p[i] - v[i] * omega) * beta;
        bistabcg_give_p<<<set_ptr->gridDim, set_ptr->blockDim, 0,
                          set_ptr->streams[_a_]>>>(p, r, v, device_vals);
      }
      checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));
      dot(r, r, _norm2_tmp_, _c_);
      {
        // v = A * p;
        _dslash(v, p, gauge);
      }
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
      dot(r_tilde, v, _tmp0_, _d_);
      {
        // alpha = rho / tmp0;
        bistabcg_give_1alpha<<<1, 1, 0, set_ptr->streams[_d_]>>>(device_vals);
      }
      checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_d_]));
      {
        // s[i] = r[i] - v[i] * alpha;
        bistabcg_give_s<<<set_ptr->gridDim, set_ptr->blockDim, 0,
                          set_ptr->streams[_a_]>>>(s, r, v, device_vals);
      }
      checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));
      {
        // t = A * s;
        _dslash(t, s, gauge);
      }
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
      dot(t, s, _tmp0_, _c_);
      dot(t, t, _tmp1_, _d_);
      checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_c_]));
      {
        // break;
        checkCudaErrors(cudaMemcpyAsync(
            ((static_cast<LatticeComplex *>(host_vals)) + _norm2_tmp_),
            ((static_cast<LatticeComplex *>(device_vals)) + _norm2_tmp_),
            sizeof(LatticeComplex), cudaMemcpyDeviceToHost,
            set_ptr->streams[_d_]));
      }
      {
        // omega = tmp0 / tmp1;
        bistabcg_give_1omega<<<1, 1, 0, set_ptr->streams[_d_]>>>(device_vals);
      }
      checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_d_]));
      {
        // r[i] = s[i] - t[i] * omega;
        bistabcg_give_r<<<set_ptr->gridDim, set_ptr->blockDim, 0,
                          set_ptr->streams[_a_]>>>(r, s, t, device_vals);
      }
      {
        // x_o[i] = x_o[i] + p[i] * alpha + s[i] * omega;
        bistabcg_give_x_o<<<set_ptr->gridDim, set_ptr->blockDim, 0,
                            set_ptr->streams[_b_]>>>(x_o, p, s, device_vals);
      }
      {
        std::cout << "##RANK:" << set_ptr->node_rank << "##LOOP:" << loop
                  << "##Residual:" << host_vals[_norm2_tmp_].real << std::endl;
        if ((host_vals[_norm2_tmp_].real < _TOL_ || loop == _MAX_ITER_ - 1)) {
          break;
        }
      }
    }
  }