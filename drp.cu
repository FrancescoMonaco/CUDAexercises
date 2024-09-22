#include <math.h>
#include <iostream>
#include <cuda.h>

class DiscretizedRandomProjections{
  public:
    DiscretizedRandomProjections(int dim, int r, int K, int L):
    dim(dim), r(r), K(K), L(L){
      sqrtL = int(sqrt(L));
      K_half = int(K/2);

      // Initialize a_l and a_r as 3d cuda int arrays of size sqrtL, K_half, dim
      cudaMalloc3DArray(&a_l, make_cudaExtent(sqrtL, K_half, dim));
      cudaMalloc3DArray(&a_r, make_cudaExtent(sqrtL, K_half, dim));

      // Initialize b_l and b_r as 2d cuda int arrays of size sqrtL, K_half
      cudaMalloc3DArray(&b_l, make_cudaExtent(sqrtL, K_half, 0));
      cudaMalloc3DArray(&b_r, make_cudaExtent(sqrtL, K_half, 0));

      // Fill with the random values using the kernel function
      int blockSize = 256;
      int numBlocks = (dim + blockSize - 1) / blockSize;
      init_random<<<numBlocks, blockSize>>>();   
      
      // Wait for GPU to finish before accessing on host
      cudaDeviceSynchronize();

    }

    ~DiscretizedRandomProjections(){
      cudaFreeArray(a_l);
      cudaFreeArray(a_r);
      cudaFreeArray(b_l);
      cudaFreeArray(b_r);
    }

    __global__
    void init_random(DiscretizedRandomProjections rp);

  __global__
   auto compute_hash(const int[] sub, const DiscretizedRandomProjections rp);

  private:
    int dim;
    int r;
    int K;
    int L;
    int sqrtL;
    int K_half;
    float* a_l, b_l, a_r, b_r;




}



void main(int argc, char** argv){

  return;
}

DiscretizedRandomProjections::init_random(DiscretizedRandomProjections rp){
  // Find the indices for all the arrays
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  int stride = blockDim.x * gridDim.x;

  // Fill the arrays with random values
  for(int i = x; i < rp.sqrtL; i += stride){
    for(int j = y; j < rp.K_half; j += stride){
      for(int k = z; k < rp.dim; k += stride){
        rp.a_l[i][j][k] = rand();
        rp.a_r[i][j][k] = rand();
      }

      rp.b_l[i][j] = rand();
      rp.b_r[i][j] = rand();
    }
  }

  return;
}