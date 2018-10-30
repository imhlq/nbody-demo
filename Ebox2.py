# Re-coding the Ebox module
# Version 2

import numpy as np
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
        self.v = (tmp_r - self.pr) / (2*t) # x(n+1) - x(n-1)

        # save pervious position
        self.pr = self.r.copy()
        self.r = tmp_r.copy()  # x(n+1)
        
        # periodic bound condition
        self.pr -= self.r - self.r % bounds
        # previous point should shift with new point to get correct velocity
        self.r = self.r % bounds

class ExperimentBox:
    # --- Parameter ---
    epsion = 1
    sigma = 1
    k_b = 1
    G = 1   # Gravitation Constant
    # -----------------
    # --- something ---
    particles = []
    # -----------------
    def __init__(self, box_size, potential='Gravity'):
        # Initize Box
        ''' 
        Provide:
          1. box_shape
          2. potential 
        '''
        # ################################ #
        # Define Size of box
        self.box_size = box_size
        self.Dim = len(self.box_size)   

        # Define potential(force) of box
        if potential == 'Lennar-Jones':
            self.force = self.Lennar_Force
        elif potential == 'Gravity':
            self.force = self.Gravity_Force
        else:
            print('Please give right potential!')
            exit()

        # Running Parameter
        self.time = 0   # actual time
        self.time_n = 0 # times of update
        


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
            # initial mass(follow uniform mass function)
            newParticle.m = np.random.uniform(1, 10)

            # Append to System
            self.particles.append(newParticle)
        print('Particles initization finished.')

    # Idea From Nvdia Gem3
    def calcRowForce(self, i):
        row_data = np.zeros((self.particle_num, self.Dim))
        pi = self.particles[i]
        for j in range(i+1, self.particle_num):
            row_data[j] = self.force(pi, self.particles[j]) # get N*dim array
        return row_data

    def calcForces(self, Parallal=False):
        if Parallal:
            # Multi-core Parallal calculation! Too slow??!! Why!
            pool = Pool()
            ForceMatrix = pool.map(self.calcRowForce, range(self.particle_num))
            pool.close()
            pool.join() # sync (wait terminate)
            # Calculate over
        else:
            ForceMatrix = list(map(self.calcRowForce, range(self.particle_num)))

        ForceMatrix = np.array(ForceMatrix)
        ForceMatrix += - np.transpose(ForceMatrix, axes=(1, 0, 2))    # assume diag always zero
        # copy up-right to down-left
        # calc force by sum column
        sumMatrix = ForceMatrix.sum(axis=0)
        
        for j in range(self.particle_num):
            self.particles[j].giveAccForce(sumMatrix[j])
    
    def update(self, t):
        # update one frame by time interval t
        self.calcForces()
        for p in self.particles:
            p.updatePos(t, self.box_size)

        # Measurement code here
        # ...
        
        # count
        self.time += t
        self.time_n += 1


    # ===========================================================================
    # ============ Type of Force ================================================
    def Lennar_Force(self, pi, pj):
        # Analytical devirivative F = -D U(r)
        r = np.sqrt(np.sum(np.power(pi.r - pj.r, 2)))
        if r < 3 * self.sigma:
            return 24 * self.epsion / self.sigma * (2 * (self.sigma/r)**13 - (self.sigma/r)**7)
        return 0

    def Gravity_Force(self, pi, pj):
        # force i to j
        if pi == pj:
            return np.zeros(self.Dim)
        soften_length = 1/20 # force softening length
        r2 = np.sum(np.power(pi.r - pj.r, 2))
        # Plummer core
        sl2 = soften_length * soften_length
        if r2 < sl2:
            return self.G * np.power(r2 + sl2, -3/2) * (pi.r - pj.r)
        else:
            return self.G * np.power(r2, -3/2) * (pi.r - pj.r)
    # ========================================================================
    # ========================================================================

    # below functions are not used yet
    def closestDistence(self, pi, pj):
        # the closest image distence of particle i and j
        # normal distence
        dr_0 = pi.r - pj.r
        # !!
        dr = min(dr_0, dr_0 + self.box_size[0], dr_0 - self.box_size[0], key=abs)
        # return r,dx,dy
        return np.sqrt(np.power(dr,2))

    def doMerge(self):
        isRemoved = False
        for pi in self.particles:
            for pj in self.particles:
                if pi == pj: continue
                rij, _, _ = self.closestDistence(pi, pj)
                if rij < 0.1 * self.sigma * np.power(pi.m + pj.m, 1/3):  # merge distance
                    r_center = (pi.r * pi.m).sum() / (pi.m + pj.m)
                    newP = Particle(r_center)
                    newP.m = pi.m + pj.m
                    newP.v = (pi.v * pi.m).sum() / (pi.m + pj.m)
                    self.particles.append(newP)
                    self.particles.remove(pi)
                    self.particles.remove(pj)
                    isRemoved = True
                    break
            if isRemoved: break
        if isRemoved: self.doMerge()