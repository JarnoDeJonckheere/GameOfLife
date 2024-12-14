# Game of Life
A multithreaded implementation of John Conway's Game of Life using NVIDIA's CUDA.

## Implementations

### gameoflife_noStride_Synchronous
- This implementation uses synchronous memory copy operations and such.
- The kernels in this version do not implement striding.

### gameoflife_noStride_Asynchronous
- This implementation uses asynchronous memory copy operations and such.
- Similar to the synchronous version, the kernels in this implementation do not implement striding.

### gameoflife_Stride_Asynchronous
- This implementation uses asynchronous memory copy operations and such.
- The kernels in this version implement striding, optimizing performance for larger grids and improving memory access patterns.