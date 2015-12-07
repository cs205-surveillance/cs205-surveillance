import numpy as np


def coordinates(inputs):
    # Reshape input array to 2D
    inputs = inputs.reshape(1080 / 30, 1920 / 32)

    # Determine corresponding top-left corner pixel location for superpixels with 1's
    output = []
    for i in range(1080 / 30):
        for j in range(1920 / 32):
            if inputs[i, j] == 1:
                point = [i * 30, j * 32]
                output.append(point)

    return output








