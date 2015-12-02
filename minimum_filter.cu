#include "minimum9.h"

// 3x3 median filter
__global__ void minimum_3x3(float *in_values,
                            float *out_values,
                            __local float *buffer,
                            int w, int h,
                            int buf_w, int buf_h,
                            const int halo) {

  // Global position of output pixel
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  // Local position relative to (0, 0) in workgroup
  const int lx = get_local_id(0);
  const int ly = get_local_id(1);

  // coordinates of the upper left corner of the buffer in image
  // space, including halo
  const int buf_corner_x = x - lx - halo;
  const int buf_corner_y = y - ly - halo;

  // coordinates of our pixel in the local buffer
  const int buf_x = lx + halo;
  const int buf_y = ly + halo;

  // Local index within work-group
  const int localIndex = ly * get_local_size(0) + lx;


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

  __syncthreads()

  if ((y < h) && (x < w)) {
    out_values[y * w + x] = minimum9(buffer[(buf_y - 1) * buf_w + (buf_x - 1)], buffer[(buf_y - 1) * buf_w + buf_x], buffer[(buf_y - 1) * buf_w + (buf_x + 1)],
                                     buffer[buf_y * buf_w       + (buf_x - 1)], buffer[buf_y * buf_w       + buf_x], buffer[buf_y * buf_w       + (buf_x + 1)],
                                     buffer[(buf_y + 1) * buf_w + (buf_x - 1)], buffer[(buf_y + 1) * buf_w + buf_x], buffer[(buf_y + 1) * buf_w + (buf_x + 1)]);
  }
}

