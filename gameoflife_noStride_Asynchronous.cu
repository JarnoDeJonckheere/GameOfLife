#include <cstdio>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <unistd.h>
#include <fstream>


//#define WIDTH 50
//#define HEIGHT 20

__global__ void gameOfLifeKernel(int* d_current, int* d_next, int width, int height, int generation){
    int x = blockIdx.x * blockDim.x + threadIdx.x; 
    int y = blockIdx.y * blockDim.y + threadIdx.y; 

    if (x >= width || y >= height){
        return;
    } 

    int index = y * width + x; //Cell selection 
    int LiveNeighbors = 0;  

    for (int dy = -1; dy <= 1; dy++){ //counts live neighbors
        for (int dx = -1; dx <=1; dx++){
            if (dx == 0 && dy == 0 ) continue;
            int nx = x + dx; 
            int ny = y + dy;

            if(nx >= 0 && nx <width && ny >= 0 && ny < height){
                LiveNeighbors += d_current[ny * width + nx];
            }
        }
    }
    
    if (d_current[index] == 1){ // implements game-rules
        d_next[index] = (LiveNeighbors == 2 || LiveNeighbors == 3) ? 1 : 0;
    } else {
        d_next[index] = (LiveNeighbors==3) ? 1 : 0; 
    }
    /*
    if (x == 0 && y == 0){ //prints output to console
        printf("Generation %d:\n", generation);
        for (int row = 0; row < height; row++){
            for (int col = 0; col < width; col++){
                int value = d_current[row * width + col];
                printf("%c", value ? '0' : '.');
            }
            printf("\n");
        }
        printf("\n");
    }

    */
}

__global__ void checkAllDeadKernel(int* grid, int width, int height, int* d_allDead){
    int x = blockIdx.x * blockDim.x + threadIdx.x; 
    int y = blockIdx.y * blockDim.y + threadIdx.y; 

    if (x >= width || y >= height) return;

    int index = y * width + x;

    if(grid[index] == 1){
        *d_allDead = 0;
    }   
}

__global__ void checkNoChangeKernel(int* d_current, int* d_next, int width, int height, int* d_noChange){
    int x = blockIdx.x * blockDim.x + threadIdx.x; 
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y>= height) return; 

    int index = y * width + x; 

    if(d_current[index] != d_next[index]){
        *d_noChange = 0;
    } 
}



int main() {
    
    cudaStream_t stream1; // for asynchronous file-write
    cudaStreamCreate(&stream1);


    // opening the outputfile to write to
    std::ofstream outPutFile;
    outPutFile.open("outPut.txt");


    //check if the file is open
    if(!outPutFile){
        std::cerr<<"error: could not open the file! (outPutFile.txt)"<<std::endl;
        return 1;
    }

    std::ofstream timingFile;
    timingFile.open("timingFile.txt");

     //check if the file is open
    if(!timingFile){
        std::cerr<<"error: could not open the file! (timingFile.txt)"<<std::endl;
        return 1;
    }

    //file header
    timingFile<<"Timing:"<<std::endl;

    //setting up random start-configuration
    int WIDTH, HEIGHT;
    printf("Enter Grid width: ");
    scanf("%d", &WIDTH);
    printf("Enter Grid Height: ");
    scanf("%d", &HEIGHT);

    clock_t beginTotal = clock();

    //printf("Starting Game of Life! \n");
    size_t size = WIDTH * HEIGHT * sizeof(int);
    int* h_current = (int*)malloc(size);
   

    //change seed to change start-configuration
    //same seed will result in same sequence
    //seed can be any int-value
    srand(42);
    for (int i = 0; i < WIDTH * HEIGHT; i++){
        h_current[i] = rand() % 2;
    }
    //write grid-parameters and first generation to file
    //write to the file
    outPutFile<<"Output:"<<std::endl; //file header
    outPutFile<<"grid_width;"<<WIDTH<<"grid_height;"<<HEIGHT<<std::endl; //grid parameters
    for(int i=0; i<WIDTH*HEIGHT;i++){
        outPutFile<<h_current[i]<<";";
    }
    outPutFile<<std::endl;

    //initialising device memory
    int *d_current, *d_next, *d_allDead, *d_noChange;
    cudaMalloc(&d_current, size);
    cudaMalloc(&d_next, size);
    cudaMalloc(&d_allDead, sizeof(int));
    cudaMalloc(&d_noChange, sizeof(int));

    // copy to device memory
    cudaMemcpyAsync(d_current, h_current, size, cudaMemcpyHostToDevice,stream1);

    // initialise cuda kernel launch-dimensions
    // pick a blockdim devisable by 32 for full warps
    dim3 blockDim(16,16);
    dim3 gridDim((WIDTH+blockDim.x-1)/blockDim.x,(HEIGHT+blockDim.y-1)/blockDim.y);

    int max_generations = 500;
    int generation= 0;
    
    while(generation<max_generations){
        int allDead = 1, noChange = 1;
        //for timing purposes
        clock_t begin = clock();
        cudaMemcpyAsync(d_allDead, &allDead, sizeof(int), cudaMemcpyHostToDevice,stream1);
        cudaMemcpyAsync(d_noChange, &noChange, sizeof(int), cudaMemcpyHostToDevice,stream1);

        gameOfLifeKernel<<<gridDim, blockDim,0,stream1>>>(d_current, d_next, WIDTH, HEIGHT, generation); //updates to the next generation
        //cudaDeviceSynchronize(); not nesecarry because cudamemcpy synchronize

        //copy this generations output to CPU
        cudaMemcpyAsync(h_current, d_next,size,cudaMemcpyDeviceToHost,stream1);
        

        checkAllDeadKernel<<<gridDim, blockDim,0,stream1>>>(d_next, WIDTH, HEIGHT, d_allDead);
        checkNoChangeKernel<<<gridDim, blockDim,0,stream1>>>(d_current, d_next, WIDTH, HEIGHT, d_noChange);

        cudaMemcpyAsync(&allDead, d_allDead, sizeof(int), cudaMemcpyDeviceToHost,stream1);
        cudaMemcpyAsync(&noChange, d_noChange, sizeof(int), cudaMemcpyDeviceToHost,stream1);

        //write to the file
        cudaStreamSynchronize(stream1);
        for(int i =0;i<WIDTH*HEIGHT;i++){
            outPutFile<<h_current[i]<<";";
        }
        outPutFile<<std::endl;

        if(allDead){
            //printf("Simulation stopped: All cells are dead at generation %d.\n", generation);
            break;
        }

        if(noChange){
            //printf("Simulation stopped: No changes detected at generation %d.\n", generation);
            break;
        }

        int* temp = d_current;
        d_current = d_next;
        d_next = temp;

        generation++;
        clock_t end = clock();
        double time = (double) (end-begin) / CLOCKS_PER_SEC; //generation time
        timingFile<<"generation_time;"<<time<<std::endl;

    }
    if(generation = max_generations){
        //printf("simulation ended after %d generations \n",generation);
    }

    
    free(h_current);
    cudaFree(d_current);
    cudaFree(d_next); 
    cudaFree(d_allDead);
    cudaFree(d_noChange);

    clock_t endTotal = clock();
    double timeTotal = (double) (endTotal-beginTotal) / CLOCKS_PER_SEC; //generation time
    outPutFile<<"totalTime;"<<timeTotal<<std::endl;

    //close the files
    outPutFile.close();
    timingFile.close();

    return 0;
}
