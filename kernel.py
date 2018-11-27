# The accerate kernel of N-body computation

# Input: Position [N * 3]
# Output: Force [N * N]

import math
import numpy as np
from numba import cuda, f8

# Parameter
Smothen_length = 0.1
Boxsize = 10
G = 1
N = 128
p = 8

ThreadPerBlock = p
BlockPerGrid = N // p

# Pre-compute
Smoth_2 = Smothen_length * Smothen_length

# Kernal
@cuda.jit(device=True)
def closestDistence(p1x, p2x, dx):
    for i in range(len(p1x)):
        dx[i] = p1x[i] - p2x[i]
        if dx[i] < - Boxsize / 2:
            dx[i] += Boxsize
        elif dx[i] > Boxsize / 2:
            dx[i] -= Boxsize

@cuda.jit(device=True)
def bodybody(pi, pj, acc):
    dr = cuda.local.array(3, dtype=f8)
    closestDistence(pi, pj, dr)
    dr2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2]
    
    distSqr  = dr2 + Smoth_2
    insDis = 1.0/math.sqrt(distSqr * distSqr * distSqr)
    s = G * insDis
    
    acc[0] = dr[0] * s
    acc[1] = dr[1] * s
    acc[2] = dr[2] * s
    
 
@cuda.jit(device=True)
def tile_calculate(myPosition, shPosition, accel, start_i):
    for i in range(cuda.blockDim.x):
        _acc = cuda.local.array(shape=3, dtype=f8)
        bodybody(myPosition, shPosition[i], _acc)
        for k in range(3):
            accel[start_i + i][k] = _acc[k]


@cuda.jit
def force_kernel(plist, accel):
    shPosition = cuda.shared.array(shape=(p, 3), dtype=f8)
    gtid = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
    tile = 0
    myPosition = plist[gtid]

    acc = cuda.local.array(shape=(N, 3), dtype=f8)
    for i in range(0, N, p):
        idx = tile * cuda.blockDim.x + cuda.threadIdx.x
        for k in range(3):
            shPosition[cuda.threadIdx.x][k] = plist[idx][k]
        cuda.syncthreads()
        tile_calculate(myPosition, shPosition, acc, i)
        cuda.syncthreads()
        tile += 1
    # save result
    for k in range(N):
        for kk in range(3):
            accel[gtid][k][kk] = acc[k][kk]



def GForce(PosMatrix):
    # input  PosMatrix: N * 3
    # Output ForceMatrix: N * N * 3
    Accel = np.zeros((N, N, 3))
    force_kernel[BlockPerGrid,ThreadPerBlock](PosMatrix, Accel)
    return Accel

#Speed Test Here.
PosMatrix = np.random.rand(N, 3)

import timeit
start = timeit.default_timer()
FM = Kernel(PosMatrix)
elpse = timeit.default_timer() - start
print(elpse)