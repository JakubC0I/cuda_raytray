#ifndef COLOR_H
#define COLOR_H

#include "interval.cuh"
#include "vec3.cuh"

#include <cmath>
#include <iostream>

using color = vec3;

__host__ __device__ inline double linear_to_gamma(double linear_compoment) {
    if (linear_compoment > 0) {
        return sqrt(linear_compoment);
    }
    return 0;
}

void write_color(std::ostream& out, const color& pixel_color) {
    auto r = pixel_color.x();
    auto g = pixel_color.y();
    auto b = pixel_color.z();

    r = linear_to_gamma(r);
    g = linear_to_gamma(g);
    b = linear_to_gamma(b);


    //  translaate the [0,1]component value  to the byte range [0,255]
    static const interval intensity(0.000, 0.999);
    int rbyte = int(255.999 * intensity.clamp(r));
    int gbyte = int(255.999 * intensity.clamp(g));
    int bbyte = int(255.999 * intensity.clamp(b));

    out << rbyte << ' ' << gbyte << ' ' << bbyte << '\n';
}
// make it dynamic
__device__ void cuda_save_color(const color& pixel_color, interval* intensity) {
    auto r = pixel_color.x();
    auto g = pixel_color.y();
    auto b = pixel_color.z();

    r = linear_to_gamma(r);
    g = linear_to_gamma(g);
    b = linear_to_gamma(b);

    interval& d_intensity = *intensity;


    //  translaate the [0,1]component value  to the byte range [0,255]
    int rbyte = int(255.999 * d_intensity.clamp(r));
    int gbyte = int(255.999 * d_intensity.clamp(g));
    int bbyte = int(255.999 * d_intensity.clamp(b));
}


#endif
