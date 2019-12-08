import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

k = [2, 5,7,9,11,15,49]
freq = 0.6 * 10
freq=75
# k = [2, 3, 4, 5, 7, 15, 49]
radii = np.linspace(0.01, 1, 100)
iterations = 400
movement = 'Rotate'

path1 = 'logs/' + movement +  '_k' + str(k[0]) + "Freq"+str(freq)+ '/correlations'
path2 = 'logs/' + movement +  '_k' + str(k[1]) + "Freq"+str(freq)+ '/correlations'
path3 = 'logs/' + movement +  '_k' + str(k[2]) + "Freq"+str(freq)+ '/correlations'
path4 = 'logs/' + movement +  '_k' + str(k[3]) + "Freq"+str(freq)+ '/correlations'
path5 = 'logs/' + movement +  '_k' + str(k[4]) + "Freq"+str(freq)+ '/correlations'
path6 = 'logs/' + movement + '_k' + str(k[5]) + "Freq"+str(freq)+ '/correlations'
path7 = 'logs/' + movement + '_k' + str(k[6]) + "Freq"+str(freq)+ '/correlations'

correlation1 = pd.read_csv(path1 + '/' + movement + '_averaged_correlation.csv')
correlation2 = pd.read_csv(path2 + '/' + movement + '_averaged_correlation.csv')
correlation3 = pd.read_csv(path3 + '/' + movement + '_averaged_correlation.csv')
correlation4 = pd.read_csv(path4 + '/' + movement + '_averaged_correlation.csv')
correlation5 = pd.read_csv(path5 + '/' + movement + '_averaged_correlation.csv')
correlation6 = pd.read_csv(path6 + '/' + movement + '_averaged_correlation.csv')
correlation7 = pd.read_csv(path7 + '/' + movement + '_averaged_correlation.csv')

fig = plt.figure()
ax = plt.axes(xlim=(0, 1), ylim=(-1, 1), title='Swarm Velocity Fluctuation Correlation \n Oscillatory Motion '
                                               '\n Frequency %02f' % (freq / 10))
line1, = ax.plot([], [], label='k = 2')
line2, = ax.plot([], [], label='k = 5')
line3, = ax.plot([], [], label='k = 7')
line4, = ax.plot([], [], label='k = 9')
line5, = ax.plot([], [], label='k = 11')
line6, = ax.plot([], [], label='k = 15')
line7, = ax.plot([], [], label='k = 49')
counter = ax.text(0.8, 0.035, '', transform=plt.gcf().transFigure)
ax.legend(loc='lower left')


def init():
    line1.set_data([], [])
    line2.set_data([], [])
    line3.set_data([], [])
    line4.set_data([], [])
    line5.set_data([], [])
    line6.set_data([], [])
    line7.set_data([], [])
    # return line3, line5, line6, line7
    return line1, line2, line3, line4, line5,line6,line7


def animate(i):
    x = radii
    y1 = correlation1.iloc[i]
    line1.set_data(x, y1)
    y2 = correlation2.iloc[i]
    line2.set_data(x, y2)
    y3 = correlation3.iloc[i]
    line3.set_data(x, y3)
    y4 = correlation4.iloc[i]
    line4.set_data(x, y4)
    y5 = correlation5.iloc[i]
    line5.set_data(x, y5)
    y6 = correlation6.iloc[i]
    line6.set_data(x, y6)
    y7 = correlation7.iloc[i]
    line7.set_data(x, y7)

    counter.set_text('Iteration %01d' % i)
    return line1, line2, line3, line4, line5, line6, line7, counter
    #return line1, line2, line3, line4, line5, counter


anim = FuncAnimation(fig, animate, init_func=init, frames=iterations)
# plt.show()
anim.save('Frequency'+str(freq)+"comparison"+'.mp4', writer='ffmpeg', fps=30)
