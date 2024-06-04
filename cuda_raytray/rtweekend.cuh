#ifndef RTWEEKEND_H
#define RTWEEKEND_H
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include "device_launch_parameters.h"
#include "curand.h"
#include "curand_kernel.h"



// C++ Std Usings
using std::fabs;
using std::make_shared;
using std::shared_ptr;
using std::sqrt;

// Constants

const double infinity = std::numeric_limits<double>::infinity();
const double pi = 3.1415926535897932385;

// Utility Functions

inline double degrees_to_radians(double degrees) {
    return degrees * pi / 180.0;
}

__device__ inline double cuda_random_double() {
    curandStateMRG32k3a_t state;
    return curand_uniform_double(&state);
}
__device__ inline double  cuda_random_double(double min, double max) {
    return min + (max - min) * cuda_random_double();
}
inline double random_double() {
    return rand() / (RAND_MAX + 1.0);
}

inline double random_double(double min, double max) {
    return min + (max - min) * random_double();
}
// Common Headers
#include "color.cuh"
#include "ray.cuh"
#include "vec3.cuh"
#include "interval.cuh"
#endif
