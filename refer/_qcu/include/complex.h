#ifndef _COMPLEX_MY_H
#define _COMPLEX_MY_H

#include "stdio.h"



#define checkCudaErrors(err)                                                   \
{                                                                            \
if (err != cudaSuccess) {                                                  \
    fprintf(stderr,                                                          \
            "checkCudaErrors() API error = %04d \"%s\" from file <%s>, "     \
            "line %i.\n",                                                    \
            err, cudaGetErrorString(err), __FILE__, __LINE__);               \
    exit(-1);                                                                \
}                                                                          \
}



class Complex {
public:
    double real;
    double imag;
    __device__ void print(void) {
        printf("%e + %e j\n", this->real, this->imag);
    }
    __device__ Complex(double real = 0.0, double imag = 0.0) {
        this->real = real;
        this->imag = imag;
    }
    __device__ Complex& operator=(const Complex& other) {

        this->real = other.real;
        this->imag = other.imag;
    
        return *this;
    }
    __device__ Complex operator=(const double& other) {
        this->real = other;
        this->imag = 0;
        return *this;
    }
    __device__ Complex operator+(const Complex& other) const {
        return Complex(this->real + other.real, this->imag + other.imag);
    }

    __device__ Complex operator-(const Complex& other) const {
        return Complex(this->real - other.real, this->imag - other.imag);
    }
    __device__ Complex operator*(const Complex& other) const {
        return Complex(this->real * other.real - this->imag * other.imag,
            this->real * other.imag + this->imag * other.real);
    }
    __device__ Complex operator*(const double& other) const {
        return Complex(this->real * other, this->imag * other);
    }

    __device__ Complex operator/(const Complex& other) const {
        double denom = other.real * other.real + other.imag * other.imag;
        return Complex((this->real * other.real + this->imag * other.imag) / denom,
            (this->imag * other.real - this->real * other.imag) / denom);
    }
    __device__ Complex operator/(const double& other) const {
        return Complex(this->real / other, this->imag / other);
    }
    __device__ Complex operator-() const { return Complex(-this->real, -this->imag); }
    
    __device__ Complex& operator+=(const Complex& other) {
        this->real += other.real;
        this->imag += other.imag;
        return *this;
    }
    __device__ Complex& operator-=(const Complex& other) {
        this->real -= other.real;
        this->imag -= other.imag;
        return *this;
    }
    __device__ Complex& operator*=(const Complex& other) {
        this->real = this->real * other.real - this->imag * other.imag;
        this->imag = this->real * other.imag + this->imag * other.real;
        return *this;
    }
    __device__ Complex& operator*=(const double& scalar) {
        this->real *= scalar;
        this->imag *= scalar;
        return *this;
    }
    __device__ Complex& operator/=(const Complex& other) {
        double denom = other.real * other.real + other.imag * other.imag;
        this->real = (real * other.real + imag * other.imag) / denom;
        this->imag = (imag * other.real - real * other.imag) / denom;
        return *this;
    }
    __device__ Complex& operator/=(const double& other) {
        this->real /= other;
        this->imag /= other;
        return *this;
    }
    __device__ bool operator==(const Complex& other) const {
        return (this->real == other.real && this->imag == other.imag);
    }
    __device__ bool operator!=(const Complex& other) const { return !(*this == other); }
    __device__ friend std::ostream& operator<<(std::ostream& os, const Complex& c) {
        if (c.imag >= 0.0) {
            os << c.real << " + " << c.imag << "i";
        }
        else {
            os << c.real << " - " << std::abs(c.imag) << "i";
        }
        return os;
    }
    __device__ Complex conj() { return Complex(this->real, -this->imag); }
    
};


class Complex_2 {
public:
    double real_1;
    double imag_1;
    double real_2;
    double imag_2;


    __device__ Complex_2& operator=(const Complex_2& other) {
        this->real_1 = other.real_1;
        this->imag_1 = other.imag_1;
        this->real_2 = other.real_2;
        this->imag_2 = other.imag_2;
        return *this;
    }

};



#endif