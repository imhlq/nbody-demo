# Experiment Box Class
import numpy as np
from scipy.stats import truncnorm
from multiprocessing import Pool

class Particle:
    m = 1   # mass
    def __init__(self, x, y):
        # deposit static particle
        self.x = x
        self.y = y
        self.vx = 0
        self.vy = 0
        self.ax = 0
        self.ay = 0
        # previous pos
        self.px = None
        self.py = None

    def giveV(self, vx, vy):
        # only used for start
        self.vx = vx
        self.vy = vy

    def giveAccForce(self, fx, fy):
        self.ax = fx / self.m
        self.ay = fy / self.m

    def updatePos(self, t, bounds):
        # energy conservation verlet
        if self.px is None or self.py is None:
            # First Time
            self.px = self.x - self.vx * t
            self.py = self.y - self.vy * t
        
        tmp_x = 2*self.x - self.px + self.ax * t * t
        tmp_y = 2*self.y - self.py + self.ay * t * t
        
        # update speed
        self.vx = (tmp_x - self.px) / (2*t) # x(n+1) - x(n-1)
        self.vy = (tmp_y - self.py) / (2*t)

        # save pervious position
        self.px = self.x  
        self.py = self.y
        self.x = tmp_x  # x(n+1)
        self.y = tmp_y
        
        # periodic bound condition
        self.px -= self.x - self.x % bounds[0]  
        self.py -= self.y - self.y % bounds[1]
        # previous point should shift with new point to get correct velocity
        self.x = self.x % bounds[0]
        self.y = self.y % bounds[1]


    def heatup(self, bounds, ratio_dT):
        # get little longer transition
        if ratio_dT is None: return None

        # In certain time, velocity change is the distance change
        self.x = self.px + (self.x - self.px) * np.sqrt(1 + ratio_dT)
        self.y = self.py + (self.y - self.py) * np.sqrt(1 + ratio_dT)
        
        # periodic bound condition
        self.px -= self.x - self.x % bounds[0]  
        self.py -= self.y - self.y % bounds[1]
        # previous point should shift with new point to get correct velocity
        self.x = self.x % bounds[0]
        self.y = self.y % bounds[1]

class Experiment2DBox:
    # --- Parameter ---
    epsion = 1
    sigma = 1
    kb = 1
    G = 0.5
    heatup = None
    # --- ---
    def __init__(self, box_shape, particle_num, max_vi, potential='Lennar-Jones', merge=False):
        # ================================================================================ # 
        # input:shape of box, num of particles, max of initial velocity, type of potential #
        # ================================================================================ # 
        # box_shape: type:{normal, lattice}, box_size:{}/lattice_size:{}, lattice_length:{}
        # particle_num: (1, inf)
        # max_vi: vi = (- max_vi, + max_vi)
        # initilizing
        isRandomMass = True

        self.type = box_shape['type']
        if box_shape['type'] == 'normal':
            self.box_size = box_shape['box_size']
        elif box_shape['type'] == 'lattice':
            self.box_size = np.multiply(box_shape['lattice_size'], box_shape['lattice_length'])
        # Potential:
        if potential == 'Lennar-Jones':
            self.force = self.Lennar_Force
        elif potential == 'Gravity':
            self.force = self.Gravity_Force
        self.particle_num = particle_num
        self.potential = potential
        self.merge = merge
        self.time = 0   # real time
        self.nstep = 0  # done step
        self.Measure = {'r2': [], 'v2': [], 'der': []}    # quantity to measure
        # Generate particles
        self.particles = []
        min_distance = 0.7 * np.sqrt(np.prod(self.box_size) / particle_num)

        anti = None # initial condition: sum of v is 0
        for _ in range(particle_num):
            # initial position
            if box_shape['type'] == 'normal':
                while True:
                    isClose = 0
                    # random x, y value in boxsize
                    newParticle = Particle(*np.multiply(np.random.rand(2), self.box_size))
                    # if close to any other particle, again
                    for p in self.particles:
                        if self.closestDistence(newParticle, p)[0] < min_distance:
                            isClose = 1
                            break
                    if isClose == 0:
                        # if not
                        break
            elif box_shape['type'] == 'lattice':
                n_x = 0.5 * box_shape['lattice_length'] + (_ % box_shape['lattice_size'][0]) * box_shape['lattice_length']
                n_y = 0.5 * box_shape['lattice_length'] + (_ // box_shape['lattice_size'][0]) * box_shape['lattice_length']
                newParticle = Particle(n_x, n_y)

            # initial velocity (follow uniform distr.)
            if anti is not None:
                # only work in problem 1
                newParticle.giveV(*anti)
                anti = None
            else:
                # generate from normal(boltzmann) distribution
                v = np.random.uniform(-max_vi, max_vi, 2)
                newParticle.giveV(*v)
                anti = - v

            # initial mass (gravity)
            if isRandomMass:
                newParticle.m = truncnorm.rvs(.2, 10)
            # done, add
            self.particles.append(newParticle)
        print('Box initization finished.')
    

            

    def calcAllForces(self):
        for pi in self.particles:
            # make 2 imaging particles to calculate forces (deviative)
            # force = d potential / d (x,y)
            fx = 0
            fy = 0
            for pj in self.particles:
                if pi == pj:
                    continue
                # calculate potential from closet image
                rij, dx, dy = self.closestDistence(pi, pj)  # pi - pj
                fij = self.force(rij, pi.m, pj.m)
                fx += fij * dx / rij
                fy += fij * dy / rij
            pi.giveAccForce(fx, fy)  # set all forces to particle acceration
        
        # print('Calculate all forces finished')
    
    def update(self, t):
        # update once by time interval t
        for p in self.particles:
            # Follow Energy conserved
            p.updatePos(t, self.box_size)  # use the update function of particle
            # heatup
            if self.nstep > 601 and self.nstep % 300 == 0:
                p.heatup(self.box_size, self.heatup)
        '''
        elif self.type == 'lattice':
            for p in self.particles:
                # Follow velocity verlet
                p.updatePosVerlet(t, self.box_size)
                self.calcAllForces()
                p.updateV(t)
        ''' 
        # print('Update finished')

    def ev_measure(self):
        # what do you want to measure with step?
        # init measure
        measure = {}
    
    def doMerge(self):
        isRemoved = False
        for pi in self.particles:
            for pj in self.particles:
                if pi == pj: continue
                rij, _, _ = self.closestDistence(pi, pj)
                if rij < 0.1 * self.sigma * np.power(pi.m + pj.m, 1/3):  # merge distance
                    x_center = (pi.x * pi.m + pj.x * pj.m) / (pi.m + pj.m)
                    y_center = (pi.y * pi.m + pj.y * pj.m) / (pi.m + pj.m)
                    newP = Particle(x_center, y_center)
                    newP.m = pi.m + pj.m
                    newP.vx = (pi.vx * pi.m + pj.vx * pj.m) / (pi.m + pj.m)
                    newP.vy = (pi.vy * pi.m + pj.vy * pj.m) / (pi.m + pj.m)
                    self.particles.append(newP)
                    self.particles.remove(pi)
                    self.particles.remove(pj)
                    isRemoved = True
                    break
            if isRemoved: break
        if isRemoved: self.doMerge()

    def goRun(self, t, n_loop=1):
        for _ in range(n_loop):
            # calculate force
            self.calcAllForces()
            # update
            self.update(t)
            # merge
            if self.merge == True:
                self.doMerge()
                continue
            # measure
            self.ev_measure()
            # - NEXT -
            self.time += t
            self.nstep += 1
            #print('Update 1 loop')


    def Lennar_Potential(self, r):
        # The interactive potential
            # Lennar-Jones Potential
        if r < 3 * self.sigma:
            LJ = lambda r: 4 * self.epsion * ((self.sigma/r)**12 - (self.sigma/r)**6)
            return LJ(r) - LJ(3)
        else:
            return 0
    
    def Lennar_Force(self, r, *extra):
        # Analytical devirivative F = -D U(r)
        if r < 3 * self.sigma:
            return 24 * self.epsion / self.sigma * (2 * (self.sigma/r)**13 - (self.sigma/r)**7)
        return 0

    def Gravity_Force(self, r, mi, mj):
        soften_length = 0.05 # force softening length
        # softening kernel
        def S(r):
            # Plummer core
            return np.power(r*r + soften_length*soften_length, -3/2) * r
        if r < soften_length:
            return - self.G * mi * mj * S(r)
        if r < 6 * self.sigma:
            return - self.G * mi * mj / (r * r)
        return 0

    
    def closestDistence(self, pi, pj):
        # the closest image distence of particle i and j
        # normal distence
        dx_0 = pi.x - pj.x
        dy_0 = pi.y - pj.y
        # !!
        dx = min(dx_0, dx_0 + self.box_size[0], dx_0 - self.box_size[0], key=abs)
        dy = min(dy_0, dy_0 + self.box_size[1], dy_0 - self.box_size[1], key=abs)
        # return r,dx,dy
        return np.sqrt(np.power(dx, 2) + np.power(dy, 2)), dx, dy
  