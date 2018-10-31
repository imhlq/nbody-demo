import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation
import numpy as np

df = pd.read_pickle('Data.pkl')


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
title = ax.set_title('3D N-body')

t0 = df.index[0]
data=df['pos'][t0]
graph = ax.scatter(data[0], data[1], data[2])

def update_graph(num):
    t = df.index[num]
    data = df['pos'][t]
    graph._offsets3d = (data[0], data[1], data[2])
    title.set_text('3D N-body, time={}'.format(num))


ani = matplotlib.animation.FuncAnimation(fig, update_graph, len(df.index), 
                               interval=40, blit=False)

plt.show()
ani.save('demo3D.mp4')