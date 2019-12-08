import pandas as pd
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

runs = np.linspace(1, 100, 100)
#runs=[1]
movement = 'Rotate'
k = 49
freq = 75


class Particle:

    def __init__(self, name, position, dphi):
        self.name = name
        self.position = position
        self.dphi = dphi
        self.neighbour_distances = []
        self.correlation = []
        self.n_dphis = []


nag = 50
radii = np.linspace(0.01, 1, 100)
iterations = 400

path = 'logs/' + movement + '_k' + str(k) +"Freq"+str(freq)+ '/correlations'

'''
Section creates correlation files for all runs at all iterations. Correlation graphs at each iteration for each run can
be made from created files
'''

for run in runs:
    path1 = 'logs/' + movement + '_k' + str(k)+"Freq"+str(freq) + '/trial%02d' % run
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path, 0o777)
        print('Directory Created')

    for it in range(iterations):
        pList = []
        df = pd.read_csv(path1 + '/' + movement + '%01d' % it + '.csv')

        for i in range(nag):
            particle = Particle('Particle%02d' % i, [df['x'][i], df['y'][i]], [df['dpx'][i], df['dpy'][i]])
            pList.append(particle)

        for particle in pList:
            for neighbour in pList:
                if particle.name == neighbour.name:
                    continue
                particle.neighbour_distances.append(distance.euclidean(particle.position, neighbour.position))
                particle.n_dphis.append(neighbour.dphi)
            for radius in radii:
                counter = 0
                local_c = 0
                for dist in particle.neighbour_distances:
                    if radius <= dist <= (radius + 0.01):
                        index = particle.neighbour_distances.index(dist)
                        counter += 1
                        local_c += np.dot(particle.dphi, particle.n_dphis[index])
                if counter > 0:
                    particle.correlation.append(local_c / counter)
                else:
                    particle.correlation.append(np.nan)

        global_correlation = []

        for i in range(len(radii)):
            radius_correlation = []
            for j in range(nag):
                radius_correlation.append(pList[j].correlation[i])
            global_correlation.append(np.nanmean(radius_correlation))

        file_name = os.path.join(path, movement + '%02d.csv' % run)
        with open(file_name, 'a') as fd:
            fd.write('\n' + ','.join(str(d) for d in global_correlation))
        print('Run: %01d' % run + ' Iteration: %01d' % it)

'''
Section creates averaged correlation and a standard deviation data file at each iteration.
Fluctuation graph can be generated at each iteration from created file. Standard deviations and averaged correlation
data stored separately
'''
all_correlations = pd.DataFrame()
iter_correlation = pd.DataFrame()
std_dev_all = pd.DataFrame()

for i in range(iterations):
    iter_correlation = iter_correlation.iloc[0:0]
    for run in runs:
        extract = path + '/' + movement + '%02d.csv' % run
        df = pd.read_csv(extract, header=None)
        iter_correlation = iter_correlation.append(df.iloc[i])
    all_correlations = iter_correlation.mean(axis=0)
    std_dev_all = iter_correlation.std(axis=0)

    file = os.path.join(path, movement + '_averaged_correlation.csv')
    with open(file, 'a') as fd:
        fd.write('\n' + ','.join(str(d) for d in all_correlations))
    file = os.path.join(path, movement + '_stdevs.csv')
    with open(file, 'a') as fd:
        fd.write('\n' + ','.join(str(d) for d in std_dev_all))
    print('Iteration %01d' % i)

# '''
# This section makes a video of the velocity fluctuation correlations over all iterations
# '''
correlations = pd.read_csv(path + '/Rotate_averaged_correlation.csv')

fig = plt.figure()
ax = plt.axes(xlim=(0, 1), ylim=(-1, 1), title='Swarm Velocity Fluctuation Correlation \n Rectangle Motion')
line, = ax.plot([], [])
counter = ax.text(0.8, 0.035, '', transform=plt.gcf().transFigure)


def init():
    line.set_data([], [])
    return line


def animate(i):
    print(i)
    x = radii
    y = correlations.iloc[i]
    line.set_data(x, y)
    counter.set_text('Iteration %01d' % i)
    return line, counter


anim = FuncAnimation(fig, animate, init_func=init, frames=(iterations-1))
#plt.show()
anim.save('testcorrelation'+str(k)+str(freq)+'.mp4', writer='ffmpeg', fps=30)

'''
This section generates a velocity fluctuation correlation graph for one iteration with error bars
'''

# correlations = pd.read_csv(path + '/' + movement + '_averaged_correlation.csv')
# std_devs = pd.read_csv(path + '/' + movement + '_stdevs.csv')
#
# iteration = 173
# fig = plt.figure()
# ax = plt.axes(xlim=(0, 1), ylim=(-1, 1), title='Swarm Velocity Fluctuation Correlation \n Rectangle Motion')
# ax.errorbar(radii, correlations.iloc[iteration], yerr=std_devs.iloc[iteration], ecolor='k', elinewidth=0.2)
# ax.text(0.8, 0.035, 'Iteration: %01d' % iteration, transform=plt.gcf().transFigure)
# plt.show()
