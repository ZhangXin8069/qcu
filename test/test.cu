#include <iostream>
#include <cuda_fp16.h> // 包含半精度相关函数

// 通用精度转换模板函数
template <typename T1, typename T2>
__device__ T2 convert(T1 a);

// 专门化 float 和 double 之间的转换
template <>
__device__ double convert<float, double>(float a)
{
    return static_cast<double>(a);
}

template <>
__device__ float convert<double, float>(double a)
{
    return static_cast<float>(a);
}

// 专门化 float 和 half 之间的转换
template <>
__device__ __half convert<float, __half>(float a)
{
    return __float2half(a);
}

template <>
__device__ float convert<__half, float>(__half a)
{
    return __half2float(a);
}

// 专门化 half 和 double 之间的转换
template <>
__device__ double convert<__half, double>(__half a)
{
    float temp = __half2float(a);
    return static_cast<double>(temp);
}

template <>
__device__ __half convert<double, __half>(double a)
{
    float temp = static_cast<float>(a);
    return __float2half(temp);
}

__global__ void precisionConversionKernel()
{
    // 测试不同精度之间的转换

    // 测试float和double之间的转换
    float a = 3.14159f;
    double b = convert<float, double>(a);
    float c = convert<double, float>(b);

    // 测试float和half之间的转换
    __half d = convert<float, __half>(a);
    float e = convert<__half, float>(d);

    // 测试half和double之间的转换
    double f = convert<__half, double>(d);
    __half g = convert<double, __half>(f);

    // 打印结果
    printf("a (float): %f\n", a);
    printf("b (double from float): %f\n", b);
    printf("c (float from double): %f\n", c);
    printf("d (half from float): %f\n", __half2float(d));
    printf("e (float from half): %f\n", e);
    printf("f (double from half): %f\n", f);
    printf("g (half from double): %f\n", __half2float(g));
}

int main()
{
    // 启动内核函数
    precisionConversionKernel<<<1, 1>>>();
    cudaDeviceSynchronize();

    return 0;
}
