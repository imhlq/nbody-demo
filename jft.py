# Test
import numpy as np
from Ebox2 import Particle

particle_num = 2048
box_size = (10, 10, 10)
soften_length = 0.1

def calcRowForce(i):
    row_data = np.zeros((particle_num, 3))
    pi = particles[i]
    for j in range(i+1, particle_num):
        row_data[j] = Gravity_Force(pi, particles[j]) # get N*dim array
    return row_data

def calcForces():
    ForceMatrix = list(map(calcRowForce, range(particle_num)))

    ForceMatrix = np.array(ForceMatrix)
    ForceMatrix += - np.transpose(ForceMatrix, axes=(1, 0, 2))    # assume diag always zero
    # copy up-right to down-left
    # calc force by sum column


def Gravity_Force(pi, pj):
    # force i to j
    if pi == pj:
        return np.zeros(3)
    dr = closestDistence(pi, pj) # with cloest distance relation
    dr2 = np.sum(np.power(dr, 2))
    # Plummer core
    sl2 = soften_length * soften_length
    return np.power(dr2 + sl2, -3/2) * (dr)

def closestDistence(pi, pj):
    # the closest image distence of particle i and j
    # move j to closest position
    dr_0 = pi.r - pj.r
    for i in range(len(dr_0)):
        if dr_0[i] < - box_size[i] / 2:
            dr_0[i] += box_size[i]
        elif dr_0[i] > box_size[i] / 2:
            dr_0[i] -= box_size[i]
    # return [dx,dy,...]
    return dr_0

particles = []
for _ in range(particle_num):
    particles.append(Particle(np.random.rand(3)))

import timeit
start = timeit.default_timer()
FM = calcForces()
elpse = timeit.default_timer() - start
print(elpse)