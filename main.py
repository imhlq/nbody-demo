from Ebox2 import ExperimentBox
import numpy as np
import pandas as pd
# import matplotlib.animation as animation
# import matplotlib.pyplot as plt

#### Change Parameter Here ####
box_size = (10, 10, 10)    # shape of Box (W * H)
particle_num = 30  # Initial Total Number of particle
max_vi = 0.3   # Init velocity
delta_t = 0.001 # Time interval
tstep = 100    # How many Frame
n_time = 1      # how many update to get one frame
# ---
Soften_length = 0.1
Hubble_friction_constant = 0.5
GravityConstant = 1
Particle_mass = 1

#### Dont change below if you don't know ## ## 

mybox = ExperimentBox(box_size, potential='Gravity')
mybox.initParticles(particle_num, max_vi)
mybox.setParameter(G=GravityConstant, m=Particle_mass, soften_length=Soften_length, h_f=Hubble_friction_constant)

final = []
for i in range(tstep):
    pos = []
    t = mybox.time
    for p in mybox.particles:
        pos.append(p.r)
    final.append({'t':t, 'pos':pos})
    mybox.updateN(delta_t, n_time)
    print(i)
final = pd.DataFrame(final)
final.set_index('t', inplace=True)
print(final)
with open('Data.pkl', 'wb') as fp:
    final.to_pickle(fp)

'''
## Animation Create 2D
fig = plt.figure()
ax = plt.axes(xlim=(0, mybox.box_size[0]), ylim=(0, mybox.box_size[1]))
x, y, c = np.random.random((3, mybox.particle_num))
sca = ax.scatter(x, y, s=5, animated=True)
text = ax.text(0.80, 1.05, '', transform=ax.transAxes)


def updateFrame(frame, box, delta_t):
    xdata = []
    ydata = []
    sdata = []
    # get all position
    for p in box.particles:
        xdata.append(p.r[0])
        ydata.append(p.r[1])
        # size
        s = 15 * p.m
        sdata.append(s)
    data = np.c_[xdata, ydata]
    sca.set_offsets(data)
    sca._sizes = sdata
    text.set_text('%.2f' % mybox.time)
    for _ in range(5):
        box.update(delta_t) # update
    print(frame)
    return sca,


ani = animation.FuncAnimation(fig, updateFrame, frames=np.arange(tstep), interval=20, fargs=(mybox, delta_t), blit=True)
#plt.show()
ani.save('demo.mp4')
'''