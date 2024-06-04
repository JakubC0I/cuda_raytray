#include <memory>

#include "camera.cuh"
#include "color.cuh"
#include "hittable_list.cuh"
#include "material.cuh"
#include "rtweekend.cuh"
#include "sphere.cuh"
#include "vec3.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#include <stdio.h>


__global__ void r(hittable* world, camera& cam, color** d_pixels, interval* intensity) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    color pixel_color = d_pixels[i][j];
    hittable& d_world = *world;
    for (int sample = 0; sample < cam.samples_per_pixel; sample++) {
        ray r = cam.cuda_get_ray(i, j);
        pixel_color += cam.cuda_ray_color(r, cam.max_depth, d_world);
    }
    cuda_save_color(cam.pixel_samples_scale * pixel_color, intensity);
}


void populate(color** array, int rows, int cols)
{
    int i, j;
    color pixel_color(0, 0, 0);

    for (i = 0; i < rows; i++)
    {
        for (j = 0; j < cols; j++)
        {
            array[i][j] = pixel_color;
        }
    }
}

color** twoDArray(int rows, int cols) {
    int i;
    color** x;

    /* obtain values for rows & cols */
    /* allocate the array */
    x = (color**)malloc(rows * sizeof * x);
    for (i = 0; i < rows; i++)
    {
        x[i] = (color*)malloc(cols * sizeof * x[i]);
    }

    populate(x, rows, cols);
    return x;
}

void initialize(int* arr, int size)
{
    for (int i = 0; i < size; i++) {
        arr[i] = i + 1;
    }
}

int main() {
    // World
    hittable_list h_world;
    const int img_w = 400;

    auto material_ground = make_shared<lambertian>(color(0.8, 0.8, 0.0));
    auto material_center = make_shared<lambertian>(color(0.1, 0.2, 0.5));
    auto material_left = make_shared<dielectric>(1.50);
    auto material_bubble = make_shared<dielectric>(1.0 / 1.50);
    auto material_right = make_shared<metal>(color(0.8, 0.6, 0.2), 1.0);

    h_world.add(make_shared<sphere>(point3(0.0, -100.5, -1.0), 100.0, material_ground));
    h_world.add(make_shared<sphere>(point3(0.0, 0.0, -1.2), 0.5, material_center));
    h_world.add(make_shared<sphere>(point3(-1.0, 0.0, -1.0), 0.5, material_left));
    h_world.add(make_shared<sphere>(point3(-1.0, 0.0, -1.0), 0.4, material_bubble));
    h_world.add(make_shared<sphere>(point3(1.0, 0.0, -1.0), 0.5, material_right));

    camera cam;
    cam.aspect_ratio = 16.0 / 9.0;
    cam.image_width = img_w;
    cam.samples_per_pixel = 100;

    cam.lookfrom = point3(-2, 2, 1);
    cam.lookat = point3(0, 0, -1);
    cam.vup = vec3(0, 1, 0);

    cam.defocus_angle = 10.0;
    cam.focus_dist = 3.4;
    int matrix = cam.image_height * img_w;
    const int height = cam.image_height;
    //VLA - variable length array
    double blocks = std::ceil(matrix / 1024);

    hittable *d_world;
    int size = 400 * 200 * sizeof(int);
    interval *d_intensity;
    interval intensity(0.000, 0.999);

    cudaMalloc(&d_world, size);
    cudaMemcpy(&d_world, &h_world, size, cudaMemcpyHostToDevice);
    cudaMalloc(&d_intensity, sizeof(interval));
    cudaMemcpy(&d_intensity, &intensity, sizeof(interval), cudaMemcpyHostToDevice);

    r<<<blocks, 1024>>>(d_world, cam, twoDArray(400, 200), d_intensity);
    return 0;
}



