#pragma optimize(5)
#include "../../include/qcu_cuda.h"

#ifdef JOD316
void cg(LatticeGauge &U,const LatticeFermi &b, LatticeFermi &x, const int &MAX_ITER, const double &TOL, const double &test)
{
    Complex rho_prev(1.0, 0.0), rho(0.0, 0.0), alpha(1.0, 0.0), omega(1.0, 0.0), beta(0.0, 0.0);
    double r_norm2 = 0;
    LatticeFermi
        r(b.lat_x, b.lat_y, b.lat_z, b.lat_t, b.lat_s, b.lat_c),
        r_tilde(b.lat_x, b.lat_y, b.lat_z, b.lat_t, b.lat_s, b.lat_c),
        p(b.lat_x, b.lat_y, b.lat_z, b.lat_t, b.lat_s, b.lat_c),
        v(b.lat_x, b.lat_y, b.lat_z, b.lat_t, b.lat_s, b.lat_c),
        s(b.lat_x, b.lat_y, b.lat_z, b.lat_t, b.lat_s, b.lat_c),
        t(b.lat_x, b.lat_y, b.lat_z, b.lat_t, b.lat_s, b.lat_c);
    // x.rand(); // initial guess
    // // ComplexVector r = b - A * x;
    x.assign_random(666);
    dslash(U, x, r, test);
    r = b - r;
    r_tilde = r;
    // r.print();
    // if x=0;r_tilde = r0 = b;
    // x.assign_zero();
    // r = b;
    // r_tilde = r;
    for (int i = 0; i < MAX_ITER; i++)
    {
        rho = r_tilde.dot(r);
        std::cout << "######rho:" << rho << " ######" << std::endl;
        beta = (rho / rho_prev) * (alpha / omega);
        std::cout << "######beta:" << beta << " ######" << std::endl;
        p = r + (p - v * omega) * beta;
        std::cout << "######p.norm_2():" << p.norm_2() << std::endl;
        // v = A * p;
        dslash(U, p, v, test);
        std::cout << "######v.norm_2():" << v.norm_2() << std::endl;
        alpha = rho / r_tilde.dot(v);
        std::cout << "######alpha:" << alpha << " ######" << std::endl;
        s = r - v * alpha;
        std::cout << "######s.norm_2():" << s.norm_2() << std::endl;
        // t = A * s;
        dslash(U, s, t, test);
        std::cout << "######t.norm_2():" << t.norm_2() << std::endl;
        omega = t.dot(s) / t.dot(t);
        std::cout << "######omega:" << omega << " ######" << std::endl;
        x = x + p * alpha + s * omega;
        std::cout << "######x.norm_2():" << x.norm_2() << std::endl;
        r = s - t * omega;
        r_norm2 = r.norm_2();
        std::cout << "######r.norm_2():" << r_norm2 << std::endl;
        std::cout << "##loop "
                  << i
                  << "##Residual:"
                  << r_norm2
                  << std::endl;
        // break;
        if (r_norm2 < TOL || i == MAX_ITER - 1)
        {
            x.print();
            break;
        }
        rho_prev = rho;
    }
}

#endif