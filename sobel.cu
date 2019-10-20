/*
                ***** sobel.cpp *****

Description:
Author:       John M. Weiss, Ph.D.
Modified By:  Jeremy Goens
Class:        CSC 461 Programming Languages
Date:         Fall 2017
Compilation   g++ -std=c++11 -g *.cpp
*/

#include <stdio.h>
#include <cstdlib>
#include <string.h>
#include <math.h>
#include <chrono>
#include <ctime>
#include "lodepng.h"
#include <thread>
#include <iostream>
using namespace std;

typedef unsigned char byte;

unsigned greyScale(char* str, byte*& image, byte*& pixels, unsigned int& width, unsigned int& height);
unsigned writeImage(byte*& image, unsigned int& width, unsigned int& height, char* str);
void sobel(byte*& image, byte*& edged, unsigned int& width, unsigned int& height);
void sobelOpenMP(byte*& image, byte*& edged, unsigned int& width, unsigned int& height);
__global__ void sobelCuda(byte* image, byte* edged, int width,int height);

// main function
int main( int argc, char** argv ){

    //Initialize variables
    char* str = argv[1];
    unsigned int width = 0;
    unsigned int height = 0;
    byte* image;
    byte* edged;
    byte* openMP;
    byte* cudaImg;
    byte* pixels;


    // CUDA device properties
    cudaDeviceProp devProp;
    cudaGetDeviceProperties( &devProp, 0 );
    int cores = devProp.multiProcessorCount;
    switch ( devProp.major )
    {
        case 2: // Fermi
            if ( devProp.minor == 1 ) cores *= 48;
            else cores *= 32; break;
        case 3: // Kepler
            cores *= 192; break;
        case 5: // Maxwell
            cores *= 128; break;
        case 6: // Pascal
            if ( devProp.minor == 1 ) cores *= 128;
            else if ( devProp.minor == 0 ) cores *= 64;
            break;
    }

    // print header
    time_t currtime = time( 0 );
    printf( "edge map benchmarks (%s)", ctime( &currtime ) );
    printf( "CPU: %d hardware threads\n", thread::hardware_concurrency() );
    printf( "GPGPU: %s, CUDA %d.%d, %d Mbytes global memory, %d CUDA cores\n",
            devProp.name, devProp.major, devProp.minor, devProp.totalGlobalMem / 1048576, cores );

    if ( argc < 2 ){
        printf( "Usage: %s infile.png\n", argv[0] );
        return -1;
    }

    unsigned error = greyScale(str, image, pixels, width, height);
    if( error ){
        printf( "Error in greyScale");
    }


    int npixels = width * height;
    edged = new byte [ npixels ];
    openMP = new byte [ npixels ];
    cudaImg = new byte [ npixels ];

    //Timing using sequential cycles
    auto a = chrono::system_clock::now();
    sobel(image, edged, width, height);
    chrono::duration<double> cpuTime = chrono::system_clock::now() - a;

    error = writeImage(edged, width, height, "image_cpu.png");
    delete [] edged;

    //Timing using OpenMP
    a = chrono::system_clock::now();
    sobelOpenMP(image, openMP, width, height);
    chrono::duration<double> ompTime = chrono::system_clock::now() - a;

    error = writeImage(openMP, width, height, "image_omp.png");
    delete [] openMP;

    int size = width * height * sizeof( int );

    // alloc device  memory
    byte* imagePass, *cudaImgPass;
    cudaMalloc( ( void ** )&imagePass, size );
    cudaMalloc( ( void ** )&cudaImgPass, size);

    // copy image to cuda memory and clean up image
    cudaMemcpy( imagePass, image, size, cudaMemcpyHostToDevice );
    delete [] image;

    // launch add() kernel on GPU with M threads per block, (N+M-1)/M blocks
    int nThreads = 512;                              // should be multiple of 32 (up to 1024)
    int nBlocks = ( (width * height) + nThreads - 1 ) / nThreads;

    printf("Made it before Chrono\n");
    a = chrono::system_clock::now();
    sobelCuda<<< nBlocks, nThreads >>>(imagePass, cudaImgPass, width, height);
    cudaError_t cudaerror = cudaDeviceSynchronize();            // waits for completion, returns error code
    if ( cudaerror != cudaSuccess ) fprintf( stderr, "Cuda failed to synchronize: %s\n", cudaGetErrorName( cudaerror ) );
    chrono::duration<double> gpuTime = chrono::system_clock::now() - a;


    cudaMemcpy( cudaImg, cudaImgPass, size, cudaMemcpyDeviceToHost );
    error = writeImage(cudaImg, width, height, "image_gpu.png");

    printf("CPU Execution time: %f msec\n", 1000 * cpuTime.count());
    printf("OpenMP Execution Time: %f msc\n", 1000 * ompTime.count());
    printf("CUDA Execution Time: %f msc\n\n", 1000 * gpuTime.count());

    printf("CPU->OMP speedup:     %f X\n",(cpuTime.count()/ompTime.count()));
    printf("OMP->GPU speedup:     %f X\n",(ompTime.count()/gpuTime.count()));
    printf("CPU->GPU speedup:     %f X\n",(cpuTime.count()/gpuTime.count()));

    //Clear out Cuda
    cudaFree( imagePass);
    cudaFree( cudaImgPass);
    return -1;
}


/*
Description:
Author:       John M. Weiss, Ph.D.
Modified By:  Jeremy Goens
Class:        CSC 461 Programming Languages
Date:         Fall 2017
Compilation   g++ *.cpp
*/
unsigned greyScale(char* str, byte*& image, byte*& pixels, unsigned int& width, unsigned int& height){
    // read input PNG file
    unsigned error = lodepng_decode_file( &pixels, &width, &height, str, LCT_RGBA, 8 );
    if ( error ){
        printf( "decoder error while reading file %s\n", str );
        printf( "error code %u: %s\n", error, lodepng_error_text( error ) );
        return -2;
    }

    // copy 24-bit RGB data into 8-bit grayscale intensity array
    int npixels = width * height;
    image = new byte [ npixels ];
    byte* img = pixels;
    for ( int i = 0; i < npixels; ++i ){
        int r = *img++;
        int g = *img++;
        int b = *img++;
        int a = *img++;     // alpha channel is not used
        image[i] = 0.3 * r + 0.6 * g + 0.1 * b + 0.5;
    }

    free ( pixels );
    return 0;

}

/*
Description:
Author:       John M. Weiss, Ph.D.
Modified By:  Jeremy Goens
Class:        CSC 461 Programming Languages
Date:         Fall 2017
Compilation   g++ *.cpp
*/
unsigned writeImage(byte*& image, unsigned int& width, unsigned int& height, char* str){

    // write grayscale PNG file
    unsigned error =  lodepng_encode_file( str, image, width, height, LCT_GREY, 8 );
    if ( error ){
        printf( "encoder error while writing file %s\n", "gray.png" );
        printf( "error code %u: %s\n", error, lodepng_error_text( error ) );
        return -3;
    }

    return 0;
}

/*
Description:  Sequentially uses the sobel formula to find the edges or an image.
Author:       John M. Weiss, Ph.D.
Modified By:  Jeremy Goens
Class:        CSC 461 Programming Languages
Date:         Fall 2017
Compilation   g++ *.cpp
*/
void sobel(byte*& image, byte*& edged, unsigned int& width, unsigned int& height){
    int gX = 0;
    int gY = 0;
    //Sobel Edge Array Calculations
    for( int i = 1; i < width-1; i++ ){
        for( int j = 1; j < height-1; j++ ) {
            gX += -1*image[(i-1)+((j-1)*width)];
            gX += -2*image[(i)+((j-1)*width)];
            gX += -1*image[(i+1)+((j-1)*width)];
            gX += 1*image[(i-1)+((j+1)*width)];
            gX += 2*image[(i)+((j+1)*width)];
            gX += 1*image[(i+1)+((j+1)*width)];

            gY += -1*image[(i-1)+((j-1)*width)];
            gY += 1*image[(i+1)+((j-1)*width)];
            gY += -2*image[(i-1)+((j)*width)];
            gY += 2*image[(i+1)+((j)*width)];
            gY += -1*image[(i-1)+((j+1)*width)];
            gY += 1*image[(i+1)+((j+1)*width)];

            edged[i+(j*width)] = sqrt((gX*gX)+(gY*gY));
            gX=gY=0;
        }
    }

    //Black Edges all around
    for(int i = 0; i < width; i ++)
      edged[i] = 255;
}

/*
Description:  Uses OpenMP to process the image concurrently.
Author:       John M. Weiss, Ph.D.
Modified By:  Jeremy Goens
Class:        CSC 461 Programming Languages
Date:         Fall 2017
Compilation   g++ *.cpp
*/
void sobelOpenMP(byte*& image, byte*& edged, unsigned int& width, unsigned int& height){
    //Sobel Edge Array Calculations
    #pragma omp parallel for
    for( int i = 1; i < width-1; i++ ){
        for( int j = 1; j < height-1; j++ ) {
            int gX = (-1)*image[(i-1)+((j-1)*width)];
            gX += (-2)*image[(i)+((j-1)*width)];
            gX += (-1)*image[(i+1)+((j-1)*width)];
            gX += 1*image[(i-1)+((j+1)*width)];
            gX += 2*image[(i)+((j+1)*width)];
            gX += 1*image[(i+1)+((j+1)*width)];

            int gY = (-1)*image[(i-1)+((j-1)*width)];
            gY += 1*image[(i+1)+((j-1)*width)];
            gY += (-2)*image[(i-1)+((j)*width)];
            gY += 2*image[(i+1)+((j)*width)];
            gY += (-1)*image[(i-1)+((j+1)*width)];
            gY += 1*image[(i+1)+((j+1)*width)];

            edged[i+(j*width)] = sqrt((gX*gX)+(gY*gY));
        }
    }

    //Black Edges all around
//    for(int i=0; i < width; i ++)
//    	edged[i*width] = 255
}

/*
Description:
Author:       John M. Weiss, Ph.D.
Modified By:  Jeremy Goens
Class:        CSC 461 Programming Languages
Date:         Fall 2017
Compilation   g++ *.cpp
*/
__global__ void sobelCuda(byte* image, byte* edged, int width, int height){
    int x = threadIdx.x + blockIdx.x * blockDim.x;

    int j = x/width;
    int i = x%width;

    if( i < 1 || i >= (width-1) || j < 1 || j >= (height-1) )
    	return;

    int gX = (-1)*image[(i-1)+((j-1)*width)];
    gX += (-2)*image[(i)+((j-1)*width)];
    gX += (-1)*image[(i+1)+((j-1)*width)];
    gX += 1*image[(i-1)+((j+1)*width)];
    gX += 2*image[(i)+((j+1)*width)];
    gX += 1*image[(i+1)+((j+1)*width)];

    int gY = (-1)*image[(i-1)+((j-1)*width)];
    gY += 1*image[(i+1)+((j-1)*width)];
    gY += (-2)*image[(i-1)+((j)*width)];
    gY += 2*image[(i+1)+((j)*width)];
    gY += (-1)*image[(i-1)+((j+1)*width)];
    gY += 1*image[(i+1)+((j+1)*width)];

    edged[i+(j*width)] = ( byte )min( sqrt( (float) (gX*gX)+(gY*gY)), 255.0);

    //Black Edges all around
}
