# Re-coding the Ebox module
# Version 2

import numpy as np
from scipy.stats import truncnorm

from multiprocessing import Pool

# Class N-dim Particle
class Particle:
    m = 1   # mass
    def __init__(self, r):
        # deposit static particle by position r
        # r can be any dimension
        self.r = r
        self.v = np.zeros(len(r))
        self.a = np.zeros(len(r))
        # previous pos
        self.pr = None

    def giveAccForce(self, f):
        # force array
        self.a = f / self.m

    def updatePos(self, t, bounds):
        # energy conservation verlet
        if self.pr is None:
            # First Time
            self.pr = self.r - self.v * t
        
        tmp_r = 2*self.r - self.pr + self.a * t * t
        
        # update speed
        self.v = (tmp_r - self.pr) / (2*t) # x(n+1) - x(n-1)

        # save pervious position
        self.pr = self.r.copy()
        self.r = tmp_r.copy()  # x(n+1)
        
        # periodic bound condition
        self.pr -= self.r - self.r % bounds
        # previous point should shift with new point to get correct velocity
        self.r = self.r % bounds

class Experiment2DBox:
    # --- Parameter ---
    epsion = 1
    sigma = 1
    k_b = 1
    G = 1   # Gravitation Constant
    # -----------------
    # --- something ---
    particles = []
    
    # -----------------
    def __init__(self, box_size, potential='Lennar-Jones'):
        # Initize Box
        ''' 
        Provide:
          1. box_shape
          2. potential 
        '''
        # ################################ #
        # Define Size of box
        self.box_size = box_size
        # Dimension of Box
        self.Dim = len(self.box_size)   

        # Define potential(force) of box
        if potential == 'Lennar-Jones':
            self.force = self.Lennar_Force
        elif potential == 'Gravity':
            self.force = self.Gravity_Force
        


    # ==================================================================
    def initParticles(self, num, vi_max, init_position='Random', moment_conserved=True):
        # Define particles in system
        self.particle_num = num
        row_num = int(self.particle_num / self.Dim)
        pos_i = np.ones(self.Dim)       # eg [1,1,1] for lattice
        for _ in range(self.particle_num):
            # ==================================
            # inital positions of particles
            if init_position == 'Random':
                # Random distribution
                newParticle = Particle(np.multiply(np.random.rand(self.Dim), self.box_size))
                
            elif init_position == 'Lattice':
                # Fixed lattice
                if (self.box_size == self.box_size[0]).all() and self.particle_num % self.Dim == 0:
                    newParticle = Particle((pos_i - 0.5) * self.box_size / row_num)
                    # get next pos
                    for i in range(len(pos_i)):
                        if pos_i[i] < row_num:
                            pos_i[i] += 1
                            break
                        else:
                            pos_i[i] = 1
                    # --- over ---
                else:
                    print('Cubic shape needed or not adaptive particle num')
                    exit()
            # ==================================
            # initial velocity (follow uniform distr.)
            anti = None
            if not moment_conserved:
                v = np.random.uniform(-vi_max, vi_max, self.Dim)
                newParticle.v = v
            else:
                if anti is None:
                    v = np.random.uniform(-vi_max, vi_max, self.Dim)
                    newParticle.v = v
                    anti = -v
                else:
                    newParticle.v = anti
                    anti = None
            # =====================================
            # initial mass(follow poisson mass function)
            newParticle.m = np.random.poisson(lam=1)

            # Append to System
            self.particles.append(newParticle)
        print('Particles initization finished.')


    def calcForces(self, Parallal=True):
        # Multi-core calculation
        pass


    # ============================================
    # ============ Type of Force =================
    def Lennar_Force(self, pi, pj):
        # Analytical devirivative F = -D U(r)
        r = np.sqrt(np.sum(np.power(pi.r - pj.r, 2)))
        if r < 3 * self.sigma:
            return 24 * self.epsion / self.sigma * (2 * (self.sigma/r)**13 - (self.sigma/r)**7)
        return 0

    def Gravity_Force(self, pi, pj):
        # force i to j
        soften_length = 1/20 # force softening length
        r2 = np.sum(np.power(pi.r - pj.r, 2))
        # Plummer core
        sl2 = soften_length * soften_length
        if r2 < sl2:
            return self.G * np.power(r2 + sl2, -3/2) * (pi.r - pj.r)
        else:
            return self.G * np.power(r2, -3/2) * (pi.r - pj.r)
    # =============================================
    # =============================================