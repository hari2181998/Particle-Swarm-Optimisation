import random
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib as mpl
mpl.rc('figure', max_open_warning = 0)
import sys

'''
PSO Algorithm 
Particle queries its fitness value from the search space based on its position and the number of elapsed iterations
Search space is dynamic and based on the following optimisation functions:
Easom Function (easom) search space from x1: [-10, 10], x2: [-10, 10] with global minimum at [pi, pi]
Rule List:
1. PSO
Standard PSO rules. Move particle towards global and personal best.
2. nBest Rumour
A particle's neighbourhood is updated at every iteration to take into account its k-nearest neighbours
Particle will get and broadcast its neighbourhood's best value and position
If particle sees its neighbours broadcasting a better value, particle will update its broadcast
Particle will be attracted to a sum of the vector to its personal best and the neighbourhood's best location

User Parameters
:param w:               Inertial weight
:param c1:              Cognitive weight
:param c2:              Social weight
:param chi:             Restriction factor
:param n_neighbours:    Number of nearest neighbours
:param max100:          Maximum units travelled in 100 iterations (Speed limit)
:param iter_max:        Maximum number of iterations
:param n_particles:     Number of particles used
:param search_space:    Search space
'''

# PSO Settings
w = 0.5
c1 = 0.
c2 = 0.5
chi = 1
n_neighbours = 2
# n_neighbours = int(sys.argv[2])
max100 = 10  # Number of "units" a particle can move in 100 iterations
iter_max = 10
n_particles = 30
# n_particles = int(sys.argv[2])

assert n_neighbours <= (n_particles - 1), 'n_neighbours higher than expected'

# Particle Memory Settings
gbest_val = float('inf')
gbest_mem_count = 0
# gbest_mem = float('inf')
gbest_mem = 0
memory = 0
# memory = float('inf')
# gbest_mem = int(sys.argv[2])
# memory = int(sys.argv[2])

# Repulsion Settings
repulsion = True
d = 6
rep_radius = 3
S = np.pi * rep_radius ** 2
alpha_r = 0.38 * np.sqrt(S / n_particles)
weight = 1000.
# weight = float(sys.argv[2])

# Search Space Settings
rule = 'nBest Rumour'
rule_list = ['PSO', 'nBest Rumour']
assert rule in rule_list, 'Invalid rule'

search_space = '2min'
search_spaces = ['easom','2min']
assert search_space in search_spaces, 'Invalid Search Space'

random.seed(25)

if search_space == 'easom':
    gbest_pos = np.array([random.uniform(-10, 10), random.uniform(-10, 10)])
if search_space == '2min':
    gbest_pos = np.array([random.uniform(-10, 10), random.uniform(-10, 10)])

movement = 'oscillate'
movements = ['line', 'rectangle', 'oscillate']
assert movement in movements, 'Invalid movement'

search_space_speed = 6.5
move_per_it = search_space_speed / 100
# For oscillate
target_amp = 5
# target_freq = float(sys.argv[1])
target_freq = 1.0  # Number of complete cycles per 100 iterations
# Run settings
trans = 100

# Save
show_ani = True
save = True
show_fig = True
save_fig = False


class SearchSpace:
    def __init__(self, move_per_it, movement, amp=13., freq=2.,search_space="easom"):
        self.move_per_it = move_per_it
        self.movement = movement
        self.amp = amp
        self.freq = freq
        self.search_space=search_space

    def fitness(self, iteration, position):

        X = position[0]
        Y = position[1]
        if self.search_space=="easom":
            if self.movement == 'line':
                fit = -np.cos(X + self.move_per_it * iteration - 3) * np.cos(Y) * \
                      np.exp(-(((X + self.move_per_it * iteration - 3) - np.pi) ** 2 + (Y - np.pi) ** 2))
            elif self.movement == 'rectangle':
                if iteration < 200:
                    fit = -np.cos(X + self.move_per_it * iteration - 3) * np.cos(Y) * \
                          np.exp(-(((X + self.move_per_it * iteration - 3) - np.pi) ** 2 + (Y - np.pi) ** 2))
                elif 200 <= iteration < 400:
                    fit = -np.cos(X + self.move_per_it * 200 - 3) * np.cos(Y + (self.move_per_it * (iteration - 200))) * \
                          np.exp(-(((X + self.move_per_it * 200 - 3) - np.pi) ** 2
                                   + (Y - np.pi + (self.move_per_it * (iteration - 200))) ** 2))
                elif iteration >= 400:
                    fit = -np.cos(X + self.move_per_it * (600 - iteration) - 3) * np.cos(Y + (self.move_per_it * 200)) * \
                          np.exp(-(((X + self.move_per_it * (600 - iteration) - 3) - np.pi) ** 2 +
                                   (Y - np.pi + (self.move_per_it * 200)) ** 2))
            elif self.movement == 'oscillate':
                mover = self.amp * np.sin(iteration / 100 * self.freq * 2 * np.pi)
                fit = -np.cos(X+3+mover+ np.pi) * np.cos(Y + np.pi) * np.exp(-((X+3+mover) ** 2 + Y ** 2))+2*(-np.cos(X-3-mover+ np.pi) * np.cos(Y + np.pi) * np.exp(-((X-3-mover) ** 2 + Y ** 2)))
        
        elif self.search_space=="2min":
            if self.movement =='oscillate':
                mover = self.amp * np.sin(iteration / 100 * self.freq * 2 * np.pi)
                fit = -200*np.exp(-0.2*np.sqrt((X-5+(mover))**2+(Y-7)**2))+-400*np.exp(-0.2*np.sqrt((X+5-(mover))**2+(Y+5)**2))
                #fit = -400*np.exp(-0.2*np.sqrt((X+5-(mover))**2+(Y+5)**2))

        return fit


class Particle:

    def __init__(self, name, k=7, max_speed=5, memory=float('inf'), d=6, alpha_r=2, repulsion=True, rep_weight=1.):
        self.name = name
        self.nearest_neighbours = k
        self.max_speed = max_speed/100
        self.it_no = 0
        self.position = np.array([random.uniform(-10, 10), random.uniform(-10, 10)])
        self.fit = float('inf')
        self.velocity = np.array([0, 0])

        self.pbest_position = self.position
        self.pbest_value = float('inf')
        self.pbest_it = 0

        self.nbest_pos = []
        self.nbest_value = float('inf')
        self.nbest_it = 0

        self.mem = memory
        self.mem_counter = 0

        self.d = d
        self.alpha_r = alpha_r
        self.repulsion = repulsion
        self.rep_weight = rep_weight

    def set_pbest(self):
        # If particle finds a better location than in history, store value and position
        if self.pbest_value > self.fit:
            self.pbest_value = self.fit
            self.pbest_position = self.position
            self.pbest_it = self.it_no
        # If pbest is better than nbest, set nbest as pbest
        if self.pbest_value > self.nbest_value:
            self.nbest_value = self.pbest_value
            self.nbest_pos = self.pbest_position

    ####################################################################################################################
    # Get neighbour methods
    ####################################################################################################################
    # Returns neighbours current positions and values. For use with normal PSO and local rule
    def get_neighbours(self, PList):
        neighbour_number = 0
        neighbour_pos = []
        neighbour_vals = []
        neighbour_dist = []
        for p in PList:
            if self.name == p.name: # To not include itself
                continue
            if neighbour_number < self.nearest_neighbours:
                neighbour_number += 1
                neighbour_pos.append(p.position)
                neighbour_vals.append(p.fit)
                neighbour_dist.append(distance.euclidean(self.position, p.position))
            elif neighbour_number == self.nearest_neighbours:
                dist = distance.euclidean(self.position, p.position)
                # If shorter distance, remove furthest neighbour (pop) and replace
                if dist < max(neighbour_dist):
                    index = neighbour_dist.index(max(neighbour_dist))
                    neighbour_dist.pop(index)
                    neighbour_vals.pop(index)
                    neighbour_pos.pop(index)
                    neighbour_pos.append(p.position)
                    neighbour_vals.append(p.fit)
                    neighbour_dist.append(distance.euclidean(self.position, p.position))
        return neighbour_pos, neighbour_vals

    # Returns rumoured best values within the neighbourhood. 3rd hand information
    # Gets values and positions from neighbours' nbest. For use with gbest rumour.
    def get_neighbours_rumour(self, PList):
        neighbour_number = 0
        r_neighbour_best_pos = []
        r_neighbour_best_vals = []
        neighbour_current_pos = []
        neighbour_dist = []
        neighbour_time = []
        for p in PList:
            if self.name == p.name:
                continue  # To not include itself
            if neighbour_number < self.nearest_neighbours:
                neighbour_number += 1
                # If neighbour's nbest value is better than its pbest value, save that value and position
                if p.nbest_value < p.pbest_value:
                    r_neighbour_best_pos.append(p.nbest_pos)
                    r_neighbour_best_vals.append(p.nbest_value)
                    neighbour_time.append(p.nbest_it)
                # If neighbour's nbest value is worse than its pbest value, save pbest value and position
                elif p.nbest_value >= p.pbest_value:
                    r_neighbour_best_pos.append(p.position)
                    r_neighbour_best_vals.append(p.pbest_value)
                    neighbour_time.append(p.pbest_it)
                neighbour_dist.append(distance.euclidean(self.position, p.position))
                neighbour_current_pos.append(p.position)
            elif neighbour_number == self.nearest_neighbours:
                dist = distance.euclidean(self.position, p.position)
                # If shorter distance, remove furthest neighbour and replace
                if dist < max(neighbour_dist):
                    index = neighbour_dist.index(max(neighbour_dist))
                    neighbour_dist.pop(index)
                    r_neighbour_best_vals.pop(index)
                    r_neighbour_best_pos.pop(index)
                    neighbour_time.pop(index)
                    neighbour_current_pos.pop(index)
                    # If neighbour's nbest value is better than its pbest value, save that value and position
                    if p.nbest_value < p.pbest_value:
                        r_neighbour_best_pos.append(p.nbest_pos)
                        r_neighbour_best_vals.append(p.nbest_value)
                        neighbour_time.append(p.nbest_it)
                    # If neighbour's nbest value is worse than its pbest value, save pbest value and position
                    elif p.nbest_value >= p.pbest_value:
                        r_neighbour_best_pos.append(p.position)
                        r_neighbour_best_vals.append(p.pbest_value)
                        neighbour_time.append(p.pbest_it)
                    neighbour_dist.append(distance.euclidean(self.position, p.position))
                    neighbour_current_pos.append(p.position)
        return r_neighbour_best_pos, r_neighbour_best_vals, neighbour_time, neighbour_current_pos

    ####################################################################################################################
    # nBest update methods
    ####################################################################################################################

    # Sets and saves nbest as the best rumoured position from neighbours. 3rd hand information.
    # For use with gbest rumour.
    def set_nbest_rumour_save(self, r_neighbour_best_pos, r_neighbour_best_vals, neighbour_best_time, memory):
        if self.nbest_value > min(r_neighbour_best_vals):
            self.nbest_value = min(r_neighbour_best_vals)
            index = r_neighbour_best_vals.index(self.nbest_value)
            # If neighbour's information was recorded before memory limit, delete.
            # Keep deleting until information is found that was recorded after memory limit
            while neighbour_best_time[index] < (self.it_no - memory):
                r_neighbour_best_pos.pop(index)
                r_neighbour_best_vals.pop(index)
                neighbour_best_time.pop(index)
                self.nbest_value = min(neighbour_best_vals)
                index = neighbour_best_vals.index(self.nbest_value)
            self.nbest_pos = r_neighbour_best_pos[index]
            self.nbest_it = neighbour_best_time[index]
        # Force change
        elif self.nbest_it < (self.it_no - memory):
            self.nbest_value = min(neighbour_best_vals)
            index = neighbour_best_vals.index(self.nbest_value)
            while neighbour_best_time[index] < (self.it_no - memory):
                r_neighbour_best_pos.pop(index)
                r_neighbour_best_vals.pop(index)
                neighbour_best_time.pop(index)
                # If all data was recorded before memory limit, set nbest pos and value as current pos and value
                if len(neighbour_best_vals) == 0:
                    self.nbest_value = self.fit
                    self.nbest_pos = self.position
                    self.nbest_it = self.it_no
                    break
                else:
                    self.nbest_value = min(neighbour_best_vals)
                    index = neighbour_best_vals.index(self.nbest_value)
                    self.nbest_pos = neighbour_best_pos[index]
                    self.nbest_it = neighbour_best_time[index]
        if self.nbest_value > self.pbest_value:
            self.nbest_value = self.pbest_value
            self.nbest_pos = self.pbest_position
            self.nbest_it = self.pbest_it

    ####################################################################################################################
    # Velocity update methods
    ####################################################################################################################

    # Particle updates velocity in accordance to standard PSO rules. Needs gbest_position as an input
    def update_pso_velocity(self, gbest_position):
        self.velocity = chi*((w * self.velocity) + (c1 * random.random() * (self.pbest_position - self.position)) +
                             (c2 * random.random() * (gbest_position - self.position)))

    # Particle updates velocity in accordance to modified PSO rules. Takes in nbest position (particle property)
    # instead of gbest
    def update_velocity_local(self):
        self.velocity = chi*((w * self.velocity) + (c1 * random.random() * (self.pbest_position - self.position)) +
                             (c2 * random.random() * (self.nbest_pos - self.position)))

    # Particle sums a vector pushing it away from its neighbours. Takes in the positions of its current neighbours
    def update_repulsion_velocity(self, neighbour_pos):
        for particle in neighbour_pos:
            vector = self.position - particle
            dist = distance.euclidean(self.position, particle)
            if dist<0.1:
                print(dist)
                print("sssssss")
            rep_vel = ((self.alpha_r / dist) ** d * (vector / dist)) * self.rep_weight
            self.velocity += rep_vel

    ####################################################################################################################
    # Move
    ####################################################################################################################
    # Particle takes velocity vector and adds to its own position
    def move(self):
        if np.linalg.norm(self.velocity) > self.max_speed:
            self.velocity = self.max_speed / np.linalg.norm(self.velocity) * self.velocity

        self.position = self.position + self.velocity


while target_freq<1.6:
    n_neighbours_list=[2,3,4,5,6,7,10,15,20,29]
    for i in n_neighbours_list:
        n_neighbours=i
        pos_current = []
        pos_history = []

        # Initialise particle objects for each particle
        particle_list = []
        for i in range(n_particles):
            particle = Particle('Particle%02d' % i, k=n_neighbours, max_speed=max100, memory=memory, d=d, alpha_r=alpha_r,
                                repulsion=repulsion, rep_weight=weight)
            particle_list.append(particle)

        # Initialise search space
        searchspace = SearchSpace(move_per_it, movement, target_amp, target_freq,search_space)

        iteration = 0

        if rule == 'PSO':
            while iteration <= iter_max:
                pos_current = []
                # Update pbest values and positions
                for particle in particle_list:
                    particle.fit = searchspace.fitness(iteration, particle.position)
                    if particle.mem_counter > particle.mem:
                        particle.pbest_value = float('inf')
                        particle.mem_counter = 0
                    particle.set_pbest()
                    # If pbest value < gbest value, update gbest value and position
                    if gbest_mem_count <= gbest_mem:
                        if particle.pbest_value < gbest_val:
                            gbest_val = particle.pbest_value
                            gbest_pos = particle.pbest_position

                    else:
                        gbest_val = particle.fit
                        gbest_pos = particle.position
                        gbest_mem_count = 0
                        # print('-------------------------------------------------')

                    pos_current.append(particle.position)

                pos_history.append(pos_current)
                # print(gbest_pos)

                if iteration == iter_max:
                    # print('Solution: ', gbest_pos)
                    # print('gbest: ', gbest_val)
                    # print('Iterations:', iteration)
                    break

                for particle in particle_list:
                    # Calculating velocity. First calculate PSO velocity, then calculate repulsion velocity
                    particle.update_pso_velocity(gbest_pos)
                    if particle.repulsion is True:
                        neighbour_pos, neighbour_vals = particle.get_neighbours(particle_list)
                        particle.update_repulsion_velocity(neighbour_pos)

                    # Move
                    particle.move()
                    particle.mem_counter += 1

                iteration += 1
                # print('Iteration: ', iteration)
                gbest_mem_count += 1

        elif rule == 'nBest Rumour':
            # Gbest set as best rumoured value and position in the neighbourhood. No confirmation required.
            while iteration <= iter_max:
                pos_current = []
                for particle in particle_list:
                    particle.fit = searchspace.fitness(iteration, particle.position)
                    if particle.mem_counter > particle.mem:
                        particle.pbest_value = float('inf')
                        particle.mem_counter = 0
                    particle.set_pbest()

                # Report to me. Does not affect particle movement
                    if particle.fit < gbest_val:
                        gbest_val = particle.fit
                        gbest_pos = particle.position
                        gbest_it = iteration
                        # print('gbest update')
                        # print('value: ', gbest_val)
                        # print('position: ', gbest_pos)
                        # print('Iteration: ', iteration)
                        # print('-------------------------------------------------')
                    pos_current.append(particle.position)
                pos_history.append(pos_current)

                for particle in particle_list:
                    neighbour_best_pos, neighbour_best_vals, neighbour_best_time, neighbour_pos =  \
                        particle.get_neighbours_rumour(particle_list)
                    particle.set_nbest_rumour_save(neighbour_best_pos, neighbour_best_vals, neighbour_best_time, memory)
                    # Calculate PSO velocity then calculate repulsion velocity
                    particle.update_velocity_local()
                    if particle.repulsion is True:
                        particle.update_repulsion_velocity(neighbour_pos)

                    particle.move()

                    particle.it_no += 1
                    particle.mem_counter += 1

                if iteration == iter_max:
                    # print()
                    # print('Solution: ', gbest_pos)
                    # print('gbest: ', gbest_val)
                    # print('Iterations: ', iteration)
                    # print('Solution found at: ', gbest_it)
                    break

                iteration += 1
                print('-----------------------------------------------------------')
                print('Iteration: ', iteration)

        Z = []
        min_loc = []    # Actual position of global minimum (for analysis)

        x = np.linspace(-15, 15, 200)
        y = np.linspace(-15, 15, 200)
        X, Y = np.meshgrid(x, y)

        plot_title = 'K = ' + str(n_neighbours) + ' Repulsion Weight = ' + str(weight) + \
                     '\n' + str(target_freq) + ' Cycles per 1000 timesteps'

        if show_fig or show_ani:
            fig = plt.figure()
            if search_space == 'easom':
                ax = fig.add_subplot(1, 1, 1, aspect='equal')
                ax = plt.axes(xlim=(-10, 10), ylim=(-10, 10), title=plot_title)
            if search_space == '2min':
                ax = fig.add_subplot(1, 1, 1, aspect='equal')
                ax = plt.axes(xlim=(-15, 15), ylim=(-15, 15), title=plot_title)


            xdata, ydata = [], []
            points, = ax.plot([], [], 'kx')

        for ts in range(iter_max):
            if search_space=="easom":
                if movement == 'line':
                    Z.append(-np.cos(X + ts * move_per_it - 3) * np.cos(Y) * np.exp(-(((X + ts * move_per_it - 3) - np.pi) **
                                                                                  2 + (Y - np.pi) ** 2)))
                    min_loc.append([np.pi - move_per_it * ts + 3, np.pi])
                elif movement == 'rectangle':
                    if ts < 200:
                        Z.append(-np.cos(X + move_per_it * ts - 3) * np.cos(Y) * \
                              np.exp(-(((X + move_per_it * ts - 3) - np.pi) ** 2 + (Y - np.pi) ** 2)))
                        min_loc.append([np.pi - move_per_it * ts + 3, np.pi])
                    elif 200 <= ts < 400:
                        Z.append(-np.cos(X + move_per_it * 200 - 3) * np.cos(Y + (move_per_it * (ts - 200))) *
                              np.exp(-(((X + move_per_it * 200 - 3) - np.pi) ** 2 +
                                       (Y - np.pi + (move_per_it * (ts - 200))) ** 2)))
                        min_loc.append([np.pi - move_per_it * 200 + 3, np.pi - move_per_it * (ts - 200)])
                    elif ts >= 200:
                        Z.append(-np.cos(X + move_per_it * (600 - ts) - 3) * np.cos(Y + (move_per_it * 200)) *
                              np.exp(-(((X + move_per_it * (600 - ts) - 3) - np.pi) ** 2 + (Y - np.pi + (move_per_it * 200)) ** 2)))
                        min_loc.append([np.pi - move_per_it * (600 - ts) + 3, np.pi - move_per_it * 200])
                elif movement == 'oscillate':
                    mover = target_amp * np.sin(ts / 100 * target_freq * 2 * np.pi)
                    Z.append(-np.cos(X+3+mover+ np.pi) * np.cos(Y + np.pi) * np.exp(-((X+3+mover) ** 2 + Y ** 2))+2*(-np.cos(X-3-mover+ np.pi) * np.cos(Y + np.pi) * np.exp(-((X-3-mover) ** 2 + Y ** 2))))
                    min_loc.append([target_amp * np.sin(ts / 100 * target_freq * 2 * np.pi), 0])
            elif search_space=="2min":
                if movement=='oscillate':
                    mover = target_amp * np.sin(ts / 100 * target_freq * 2 * np.pi)
                    Z.append(-200*np.exp(-0.2*np.sqrt((X-5+(mover))**2+(Y-7)**2))+-400*np.exp(-0.2*np.sqrt((X+5-(mover))**2+(Y+5)**2)))
                    #Z.append(-400*np.exp(-0.2*np.sqrt((X+5-(mover))**2+(Y+5)**2)))
                    min_loc.append([5-(mover),-5])


        ave_dist_history = []
        centroid_dist_history = []
        p_to_centroid_history = []
        min_dist_history = []
        max_dist_history = []
        total_dist_history = []
        rms_history = []
        track_time = 0
        tracking_distance = 0.5
        rms_current1=0
        time_avg_dist=0

        for i in range(iter_max):
            X_pos = []
            Y_pos = []
            dist_current = []
            dist_to_centroid_current = []
            rms_current = 0
            total_dist = 0
            for position in pos_history[i]:
                X_pos.append(position[0])
                Y_pos.append(position[1])
                dist_current.append(abs(distance.euclidean(position, min_loc[i])))
                total_dist += abs(distance.euclidean(position, min_loc[i]))
                rms_current += abs(distance.euclidean(position, min_loc[i])) ** 2
            rms_current = np.sqrt(rms_current/n_particles)
            rms_current1 += rms_current
            time_avg_dist+=np.average(dist_current)

            if show_fig:
                xdata.append(X_pos)
                ydata.append(Y_pos)

            ave_dist_history.append(np.average(dist_current))
            min_dist_history.append(min(dist_current))
            max_dist_history.append(max(dist_current))
            total_dist_history.append(total_dist)
            rms_history.append(rms_current)
            if min(dist_current) < tracking_distance and i > trans:
                track_time += 1

            centroid = [sum(X_pos) / n_particles, sum(Y_pos) / n_particles]
            for position in pos_history[i]:
                dist_to_centroid_current.append(abs(distance.euclidean(position, centroid)))
            centroid_dist_history.append(distance.euclidean(centroid, min_loc[i]))
            p_to_centroid_history.append(np.average(dist_to_centroid_current))

        print(track_time)


        def init():
            points.set_data([], [])
            return points,


        def animate(i):
            ax.contourf(X, Y, Z[i])
            x = xdata[i]
            y = ydata[i]
            points.set_data(x, y)
            return points,


        if show_fig or show_ani:
            anim = animation.FuncAnimation(fig, animate, init_func=init, frames=iter_max)

        if show_ani:
            plt.show()

        if save:
            anim.save("2min"+"k="+str(n_neighbours)+"f="+str(target_freq)+".mp4", writer='ffmpeg')

        if show_fig or save_fig:
            plt.clf()
            plt.plot(range(iter_max), ave_dist_history, label='Average Particle Distance to Optimum')
            # plt.plot(range(iter_max), min_dist_history, label='Smallest Particle Distance to Optimum')
            # plt.plot(range(iter_max), centroid_dist_history, label='Distance of Centroid to Optimum')
            # plt.plot(range(iter_max), p_to_centroid_history, label='Average Particle Distance to Centroid')
            plt.plot(range(iter_max), max_dist_history, label='Maximum Particle Distance to Optimum')
            # plt.plot(range(iter_max), total_dist_history, label='Sum of All Distances to Optimum')
            plt.plot(range(iter_max), rms_history, label='RMS Distance')
            plt.xlabel('Iteration')
            plt.ylabel('Distance')
            plt.title(plot_title)
            # plt.ylim([0, 9])
            plt.legend()
            fig1 = plt.gcf()

        #if show_fig:
        #    plt.show()

        if save_fig:
            fig1.savefig(str(n_neighbours)+"sim.png")
        #n_neighbours+=1
    target_freq+=0.1
