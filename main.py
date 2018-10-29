from EBox import Experiment2DBox
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt

box_shape = {'type': 'normal', 'box_size': (20, 20)}
#box_shape = {'type': 'lattice', 'lattice_size': (4, 4), 'lattice_length': 2.5}
particle_num = 100
max_vi = 1.5    # init velocity
delta_t = 0.005
mybox = Experiment2DBox(box_shape, particle_num, max_vi, potential='Gravity')
tstep = 2000

## Animation Create
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
        xdata.append(p.x)
        ydata.append(p.y)
        # size
        s = 10 * p.m
        sdata.append(s)
    data = np.c_[xdata, ydata]
    sca.set_offsets(data)
    sca._sizes = sdata
    box.goRun(delta_t)
    print(frame)
    return sca,



ani = animation.FuncAnimation(fig, updateFrame, frames=np.arange(1, tstep), interval=10, fargs=(mybox, delta_t), blit=True)
ani.save('demo.mp4')
#plt.show()

