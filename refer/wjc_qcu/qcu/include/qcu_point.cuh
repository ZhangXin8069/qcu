#pragma once
#include "qcu_macro.cuh"
#include "qcu_complex.cuh"
#include <assert.h>
class Point {
private:
  int x_;
  int y_;
  int z_;
  int t_;
  int parity_;
public:
  Point() = default;
  __device__ Point(const Point& rhs) : x_(rhs.x_), y_(rhs.y_), z_(rhs.z_), t_(rhs.t_), parity_(rhs.parity_) {}
  __device__ Point(int x, int y, int z, int t, int parity) : x_(x), y_(y), z_(z), t_(t), parity_(parity) {}
  // __device__ void outputInfo() {
  //   printf("Point: (x,y,z,t)=(%d, %d, %d, %d), parity = %d\n", x_, y_, z_, t_, parity_);
  // }
  __device__ int getParity() const { return parity_;}
  __device__ Point move(int front_back, int direction, int Lx, int Ly, int Lz, int Lt) const{ // direction +-1234
    // 1-front 0-back
    assert(abs(direction) >= 0 && abs(direction) < 4);
    assert(front_back == BACK || front_back == FRONT);

    int new_pos;
    int eo = (y_ + z_ + t_) & 0x01;    // (y+z+t)%2

    if (direction == 0) {
      if (!front_back) {
        new_pos = x_ + (eo == parity_) * (-1 + (x_ == 0) * Lx);
        return Point(new_pos, y_, z_, t_, 1-parity_);
      } else {
        new_pos = x_ + (eo != parity_) * (1 + (x_ == Lx-1) * (-Lx));
        return Point(new_pos, y_, z_, t_, 1-parity_);
      }
    } else if (direction == 1) {  // y 前进
      if (!front_back) {
        new_pos = y_ - 1 + (y_ == 0) * Ly;
        return Point(x_, new_pos, z_, t_, 1-parity_);
      } else {
        new_pos = y_ + 1 + (y_ == Ly-1) * (-Ly);
        return Point(x_, new_pos, z_, t_, 1-parity_);
      }
    } else if (direction == 2) {
      if (!front_back) {
        new_pos = z_ - 1 + (z_ == 0) * Lz;
        return Point(x_, y_, new_pos, t_, 1-parity_);
      } else {
        new_pos = z_ + 1 + (z_ == Lz-1) * (-Lz);
        return Point(x_, y_, new_pos, t_, 1-parity_);
      }
    } else if (direction == 3) {
      if (!front_back) {
        new_pos = t_ - 1 + (t_ == 0) * Lt;
        return Point(x_, y_, z_, new_pos, 1-parity_);
      } else {
        new_pos = t_ + 1 + (t_ == Lt-1) * (-Lt);
        return Point(x_, y_, z_, new_pos, 1-parity_);
      }
    } else {
      return *this;
    }
  }

  __device__ Complex* getPointGauge(Complex* origin, int direction, int Lx, int Ly, int Lz, int Lt) const{
    return origin + (((((((direction << 1) + parity_) * Lt + t_) * Lz + z_) * Ly + y_) * Lx) + x_) * Nc * Nc;
  }


  __device__ Complex* getPointVector(Complex* origin, int Lx, int Ly, int Lz, int Lt) const{
    return origin + (((t_ * Lz + z_) * Ly + y_) * Lx + x_) * Ns * Nc;
  }

  __device__ double* getCoalescedVectorAddr (void* origin, int Lx, int Ly, int Lz, int Lt) const{
    return static_cast<double*>(origin) + (((t_ * Lz + z_) * Ly + y_) * Lx + x_);
  }
  __device__ double* getCoalescedGaugeAddr (void* origin, int direction, int sub_Lx, int Ly, int Lz, int Lt) const{
    return static_cast<double*>(origin) + (direction * 2 + parity_)* sub_Lx * Ly * Lz * Lt * Nc * (Nc - 1) * 2 + (((t_ * Lz + z_) * Ly + y_) * sub_Lx + x_);
    // direction <<1 thread_id;
    // return origin + (((((((direction << 1) + parity_) * Lt + t_) * Lz + z_) * Ly + y_) * Lx) + x_) * Nc * Nc;
  }

  __device__ Point& operator= (const Point& rhs) {
    x_ = rhs.x_;
    y_ = rhs.y_;
    z_ = rhs.z_;
    t_ = rhs.t_;
    parity_ = rhs.parity_;
    return *this;
  }
  __device__ Complex* getPointClover(Complex* origin, int Lx, int Ly, int Lz, int Lt) const{
    return origin + (((((parity_ * Lt + t_) * Lz + z_) * Ly + y_) * Lx) + x_) * (Nc * Ns * Nc * Ns / 2);
  }

};