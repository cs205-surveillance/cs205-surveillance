// 3x3 median filter
__global__ void minimum_3x3(float *in_values,
                            float *out_values,
                            int w, int h,
                            int buf_w, int buf_h,
                            const int halo) {
  __shared__ float buffer[25];

  // Global position of output pixel
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  // Local position relative to (0, 0) in workgroup
  const int lx = threadIdx.x;
  const int ly = threadIdx.y;

  // coordinates of the upper left corner of the buffer in image
  // space, including halo
  const int buf_corner_x = x - lx - halo;
  const int buf_corner_y = y - ly - halo;

  // coordinates of our pixel in the local buffer
  const int buf_x = lx + halo;
  const int buf_y = ly + halo;

  // Local index within work-group
  const int localIndex = ly * blockDim.x + lx;


  if ((y < h) && (x < w)) { 
    if (localIndex < buf_w)
      for (int row = 0; row < buf_h; row++) {
        
        // Calculate x and y for buffer
        int yIndex = buf_corner_y + row;
        int xIndex = buf_corner_x + localIndex;

        // Check for bounds
        if (xIndex < 0) xIndex = 0;
        if (yIndex < 0) yIndex = 0;
        if (xIndex >= w) xIndex = w - 1;
        if (yIndex >= h) yIndex = h - 1;
        
        // Store in buffer with corrected index
        buffer[row * buf_w + localIndex] = in_values[yIndex * w + xIndex];
      }
  }

  __syncthreads();

  if ((y < h) && (x < w)) {

    float s0 = buffer[(buf_y - 1) * buf_w + (buf_x - 1)];
    float s1 = buffer[(buf_y - 1) * buf_w + buf_x];
    float s2 = buffer[(buf_y - 1) * buf_w + (buf_x + 1)];
    float s3 = buffer[buf_y * buf_w + (buf_x - 1)];
    float s4 = buffer[buf_y * buf_w + buf_x];
    float s5 = buffer[buf_y * buf_w + (buf_x + 1)];
    float s6 = buffer[(buf_y + 1) * buf_w + (buf_x - 1)];
    float s7 = buffer[(buf_y + 1) * buf_w + buf_x];
    float s8 = buffer[(buf_y + 1) * buf_w + (buf_x + 1)]);

    out_values[y * w + x] = min(s0, min(s1, min(s2, min(s3, min(s4, min(s5, min(s6, min(s7, s8))))))));

  }
}

