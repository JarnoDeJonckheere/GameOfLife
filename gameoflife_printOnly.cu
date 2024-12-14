#include <cstdio>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <unistd.h>

//#define WIDTH 50
//#define HEIGHT 20

__global__ void gameOfLifeKernel(int* d_current, int* d_next, int width, int height, int generation){
    int x = blockIdx.x * blockDim.x + threadIdx.x; 
    int y = blockIdx.y * blockDim.y + threadIdx.y; 

    if (x >= width || y >= height){
        return;
    } 

    int index = y * width + x; //De cell die we de buren van gaan bekijken. De matrix is eigenlijk een 1D array. 
    int LiveNeighbors = 0; //aantal levende buren. 

    for (int dy = -1; dy <= 1; dy++){ //functie voor het bekijken van de buren. 
        for (int dx = -1; dx <=1; dx++){
            if (dx == 0 && dy == 0 ) continue;
            int nx = x + dx; 
            int ny = y + dy;

            if(nx >= 0 && nx <width && ny >= 0 && ny < height){
                LiveNeighbors += d_current[ny * width + nx];
            }
        }
    }
    
    if (d_current[index] == 1){
        d_next[index] = (LiveNeighbors == 2 || LiveNeighbors == 3) ? 1 : 0;
    } else {
        d_next[index] = (LiveNeighbors==3) ? 1 : 0; 
    }

    if (x == 0 && y == 0){
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
    int WIDTH, HEIGHT;
    printf("Enter Grid width: ");
    scanf("%d", &WIDTH);
    printf("Enter Grid Height: ");
    scanf("%d", &HEIGHT);

    printf("Starting Game of Life! \n ");
    size_t size = WIDTH * HEIGHT * sizeof(int);
    int* h_current = (int*)malloc(size);
    int* h_next = (int*)malloc(size);

    srand(42);
    for (int i = 0; i < WIDTH * HEIGHT; i++){
        h_current[i] = rand() % 2;
    }

    int *d_current, *d_next, *d_allDead, *d_noChange;
    cudaMalloc(&d_current, size);
    cudaMalloc(&d_next, size);
    cudaMalloc(&d_allDead, sizeof(int));
    cudaMalloc(&d_noChange, sizeof(int));

    cudaMemcpy(d_current, h_current, size, cudaMemcpyHostToDevice);

    dim3 blockDim(16,16);
    dim3 gridDim((WIDTH+blockDim.x-1)/blockDim.x,(HEIGHT+blockDim.y-1)/blockDim.y);

    int generation= 0;
    while(true){
        int allDead = 1, noChange = 1;
        cudaMemcpy(d_allDead, &allDead, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_noChange, &noChange, sizeof(int), cudaMemcpyHostToDevice);

        gameOfLifeKernel<<<gridDim, blockDim>>>(d_current, d_next, WIDTH, HEIGHT, generation);
        cudaDeviceSynchronize();
        
        checkAllDeadKernel<<<gridDim, blockDim>>>(d_next, WIDTH, HEIGHT, d_allDead);
        cudaDeviceSynchronize();
        cudaMemcpy(&allDead, d_allDead, sizeof(int), cudaMemcpyDeviceToHost);

        checkNoChangeKernel<<<gridDim, blockDim>>>(d_current, d_next, WIDTH, HEIGHT, d_noChange);
        cudaDeviceSynchronize();
        cudaMemcpy(&noChange, d_noChange, sizeof(int), cudaMemcpyDeviceToHost);

        if(allDead){
            printf("Simulation stopped: All cells are dead at generation %d.\n", generation);
            break;
        }

        if(noChange){
            printf("Simulation stopped: No changes detected at generation %d.\n", generation);
            break;
        }

        int* temp = d_current;
        d_current = d_next;
        d_next = temp;

        generation++;
        usleep(200000);
    }

    cudaDeviceSynchronize();

    int* temp = d_current;
    d_current = d_next;
    d_next = temp;
    usleep(200000);

    free(h_current);
    free(h_next);
    free(d_current);
    free(d_next); 
    free(d_allDead);
    free(d_noChange);
    return 0;
}
