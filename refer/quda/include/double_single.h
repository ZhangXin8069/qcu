#pragma once

__host__ __device__ inline void dsadd(float2 &c, const float2 &a, const float2 &b) {
  float t1 = a.x + b.x;
  float e = t1 - a.x;
  float t2 = ((b.x - e) + (a.x - (t1 - e))) + a.y + b.y;
  // The result is t1 + t2, after normalization.
  c.x = e = t1 + t2;
  c.y = t2 - (e - t1);
}

struct doublesingle {
  float2 a;
  __host__ __device__ inline doublesingle() : a(make_float2(0.0f,0.0f)) { ; }
  __host__ __device__ inline doublesingle(const doublesingle &b) : a(make_float2(b.a.x, b.a.y)) { ; }
  __host__ __device__ inline doublesingle(const float a) : a(make_float2(a, 0.0)) { ; }

  __host__ __device__ inline void operator+=(const doublesingle &b) { dsadd(this->a, this->a, b.a); }
  __host__ __device__ inline void operator+=(const float &b) { 
    float2 b2 = make_float2(b, 0.0); 
    dsadd(this->a, this->a, b2); }
  
  __host__ __device__ inline doublesingle& operator=(const doublesingle &b)
    { a.x = b.a.x; a.y = b.a.y; return *this; }
    
  __host__ __device__ inline doublesingle& operator=(const float &b)
  { a.x = b; a.y = 0.0f; return *this; }
};

__host__ __device__ inline doublesingle operator+=(doublesingle &a, const doublesingle &b)
{ dsadd(a.a, a.a, b.a); return a;}

__host__ __device__ double operator+=(double& a, doublesingle &b) { a += b.a.x; a += b.a.y; return a; }

struct doublesingle2 { 
  doublesingle x; 
  doublesingle y; 
  __host__ __device__ inline doublesingle2& operator=(const double &a) 
  { x = a; y = a; return *this; } 
  __host__ __device__ inline void operator+=(const double2 &b) {x += b.x; y += b.y;}
  __host__ __device__ inline void operator+=(const doublesingle2 &b) {x += b.x; y += b.y;}
};

__host__ __device__ double2 operator+=(double2& a, doublesingle2 &b) 
{ a.x += b.x.a.x; a.x += b.x.a.y; a.y += b.y.a.x; a.y += b.y.a.y; return a; }

struct doublesingle3 { 
  doublesingle x; 
  doublesingle y; 
  doublesingle z; 
  __host__ __device__ inline void operator+=(const double3 &b) {x += b.x; y += b.y; z += b.z;}
  __host__ __device__ inline void operator+=(const doublesingle3 &b) {x += b.x; y += b.y; z+= b.z;}
};

__host__ __device__ double3 operator+=(double3& a, doublesingle3 &b)
{ a.x += b.x.a.x; a.x += b.x.a.y; a.y += b.y.a.x; a.y += b.y.a.y; a.z += b.z.a.x; a.z += b.z.a.y; return a; }
