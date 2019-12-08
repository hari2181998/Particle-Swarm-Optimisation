"""
PSO static case simulation with kinematic constraints for all the particles and initial positions of the 10 bots used in the
real experiments.Stabylinski Tank function extended to -10 to 10 search space to fit more than 10 particles.
"""
import random
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib as mpl
from math import pi
from math import sin,cos,atan2
mpl.rc('figure', max_open_warning = 0)

'''
PSO Algorithm 

Calculations of particle velocity and position done in each iteration of particle class

Choose between the following optimisation functions as the search space:
Rastrigin Function (rastrigin) search space from x1: [-5.12, 5.12], x2: [-5.12, 5.12] with global minimum at [0, 0]
Styblinski-Tang Function (st) search space from x1: [-5, 5], x2: [-5, 5] with global minimum at [-2.9035, -2.9035]
Easom Function (easom) search space from x1: [-10, 10], x2: [-10, 10] with global minimum at [pi, pi]

Rule List:
1. PSO
Standard PSO rules. Move particle towards global and personal best.

2. Local
A particle's neighbourhood is updated at every iteration to take into account its k-nearest neighbours
Particle will be attracted to a sum of the vector to its personal best and the neighbourhood's current best location

3. Gbest Rumour
A particle's neighbourhood is updated at every iteration to take into account its k-nearest neighbours
Particle will get and broadcast its neighbourhood's best value and position
If particle sees its neighbours broadcasting a better value, particle will update its broadcast
Particle will be attracted to a sum of the vector to its personal best and the neighbourhood's best location
'Elim said Aaron said (someone else said etc.) the BKT is better in KL, I'm going to KL'

4. Gbest Conf
A particle's neighbourhood is updated at every iteration to take into account its k-nearest neighbours
Particle will be attracted to a sum of the vector to its personal best and the neighbourhood's best location
'Aaron said BKT is better in KL. I'm going to KL'

User Parameters
:param w:               Inertial weight
:param c1:              Cognitive weight
:param c2:              Social weight
:param chi:             Restriction factor
:param n_neighbours:    Number of nearest neighbours
:param max100:          Maximum units travelled in 100 iterations (Speed limit)
:param iter_max:        Maximum number of iterations
:param n_particles:     Number of particles used
'''
avg_distancearr=[]
min_distarr=[]
max_distarr=[]
avg_distancereparr=[]
min_distreparr=[]
max_distreparr=[]
n_neighbours=50
weight=1  #repulsion weight

w = 0.5   
c1 = 0.5
c2 = 0.5
chi = 1
#n_neighbours =9
max100 = 10  # Number of "units" a particle can move in 100 iterations



rule = 'local'  #K nearest neighbour implementation
rule_list = ['pso', 'local', 'gbest rumour', 'gbest conf']
assert rule in rule_list, 'Invalid rule'

search_space = 'st'   #stabylinski tank function
search_spaces = ['rastrigin', 'st', 'easom']

show_ani = True
save = True

iter_max = 700
n_particles = 100

repulsion = True
d = 1
rep_radius = 3
S = np.pi * rep_radius ** 2
alpha_r = 0.01
#weight = 500.

gbest_val = 90
gbest_prev = 190

if search_space == 'rastrigin':
    gbest_pos = np.array([random.uniform(-5.12, 5.12), random.uniform(-5.12, 5.12)])
elif search_space == 'st':
    gbest_pos = np.array([random.uniform(-11, 11), random.uniform(-12.5, 12.5)])
elif search_space == 'easom':
    gbest_pos = np.array([random.uniform(-10, 10), random.uniform(-10, 10)])

pos_current = []
pos_history = []


class Particle:
    def __init__(self, name,search_func,k=7, max_speed=5, memory=float('inf'), d=6, alpha_r=0.25, repulsion=True, rep_weight=1.):
        self.name = name
        self.nearest_neighbours = k
        self.search_space = search_func
        #self.position = position
        #print (self.position,"self.position")
        self.heading = 90
        self.d = d
        self.alpha_r = alpha_r
        self.repulsion = repulsion
        self.rep_weight = rep_weight

        if self.search_space == 'rastrigin':
            self.position = np.array([random.uniform(-5.12, 5.12), random.uniform(-5.12, 5.12)])
            self.fit = 20 + (self.position[0] ** 2 - 10 * np.cos(2*np.pi*self.position[0])) + \
                       (self.position[1] ** 2 - 10 * np.cos(2*np.pi*self.position[1]))
        elif self.search_space == 'st':
            self.position = np.array([random.uniform(0, 5), random.uniform(-5, 5)])
            self.fit = 0.5 * ((self.position[0] ** 4 - 16 * self.position[0] ** 2 + 5 * self.position[0]) +
                              (self.position[1] ** 4 - 16 * self.position[1] ** 2 + 5 * self.position[1]))
        elif self.search_space == 'easom':
            #self.position = np.array([random.uniform(-10, 10), random.uniform(-10, 10)])
            self.fit = -np.cos(self.position[0]) * np.cos(self.position[1]) * np.exp(-((self.position[0] - np.pi) ** 2 +
                                                                                       (self.position[1] - np.pi) ** 2))+2*(-np.cos(self.position[0]+5) * np.cos(self.position[1]) * np.exp(-((self.position[0]+5 - np.pi) ** 2 +
                                                                                       (self.position[1] - np.pi) ** 2)))

        self.velocity = np.array([0., 0.])
        self.max_speed = max_speed/100
        self.pbest_position = self.position
        self.pbest_value = float('inf')
        self.nbest_pos = self.position
        self.nbest_value = float('inf')

    def fitness(self):
        x1 = self.position[0]
        x2 = self.position[1]
        x1=x1*(10.0/16)   # To extend the field from -5 to 5 to -10 to 10
        x2=x2*(10.0/25)
        if self.search_space == 'rastrigin':
            f = 20 + (x1 ** 2 - 10 * np.cos(2*np.pi*x1)) + (x2 ** 2 - 10 * np.cos(2*np.pi*x2))
        elif self.search_space == 'st':
            f = 0.5 * ((x1 ** 4 - 16 * x1 ** 2 + 5 * x1) + (x2 ** 4 - 16 * x2 ** 2 + 5 * x2))
        elif self.search_space == 'easom':
            f = -np.cos(x1) * np.cos(x2) * np.exp(-((x1 - np.pi) ** 2 + (x2 - np.pi) ** 2))+2*(-np.cos(x1+5) * np.cos(x2) * np.exp(-((x1+5 - np.pi) ** 2 + (x2 - np.pi) ** 2)))
        self.fit = f

    def set_pbest(self):
        if self.pbest_value > self.fit:
            self.pbest_value = self.fit
            self.pbest_position = self.position

    def move_pso(self, gbest_position):
        self.velocity = chi*((w * self.velocity) + (c1 * random.random() * (self.pbest_position - self.position)) +
                             (c2 * random.random() * (gbest_position - self.position)))

        if np.linalg.norm(self.velocity) > self.max_speed:
            self.velocity = self.max_speed / np.linalg.norm(self.velocity) * self.velocity

        self.position = self.position + self.velocity
        

    # Returns neighbours current positions and values
    def get_neighbours(self, PList,iteration):
        neighbour_number = 0
        neighbour_pos = []
        neighbour_vals = []
        neighbour_dist = []
        for p in PList:
            if self.name == p.name:
                #print("same name",iteration)
                continue
            if neighbour_number < self.nearest_neighbours:
                #print(self.name,p.name)
                neighbour_number += 1
                neighbour_pos.append(p.position)
                neighbour_vals.append(p.fit)
                #if self.name == "Particle00":
                 #   print(self.position,p.position)
                neighbour_dist.append(distance.euclidean(self.position, p.position))
            elif neighbour_number == self.nearest_neighbours:
                dist = distance.euclidean(self.position, p.position)
                # If shorter distance, remove furthest neighbour and replace
                if dist < max(neighbour_dist):
                    index = neighbour_dist.index(max(neighbour_dist))
                    neighbour_dist.pop(index)
                    neighbour_vals.pop(index)
                    neighbour_pos.pop(index)
                    neighbour_pos.append(p.position)
                    neighbour_vals.append(p.fit)
                    neighbour_dist.append(distance.euclidean(self.position, p.position))
            #print(neighbour_pos,self.name)
        return neighbour_pos, neighbour_vals

    # Returns neighbours best positions and values from their experience
    def get_neighbours_mem(self, PList):
        neighbour_number = 0
        neighbour_best_pos = []
        neighbour_best_vals = []
        neighbour_dist = []
        for p in PList:
            if self.name == p.name:
                continue
            if neighbour_number < self.nearest_neighbours:
                neighbour_number += 1
                neighbour_best_pos.append(p.pbest_position)
                neighbour_best_vals.append(p.pbest_value)
                neighbour_dist.append(distance.euclidean(self.position, p.position))
            elif neighbour_number == self.nearest_neighbours:
                dist = distance.euclidean(self.position, p.position)
                # If shorter distance, remove furthest neighbour and replace
                if dist < max(neighbour_dist):
                    index = neighbour_dist.index(max(neighbour_dist))
                    neighbour_dist.pop(index)
                    neighbour_best_vals.pop(index)
                    neighbour_best_pos.pop(index)
                    neighbour_best_pos.append(p.pbest_position)
                    neighbour_best_vals.append(p.pbest_value)
                    neighbour_dist.append(distance.euclidean(self.position, p.position))
        return neighbour_best_pos, neighbour_best_vals

    # Returns rumoured best values within the neighbourhood
    def get_neighbours_rumour(self, PList):
        neighbour_number = 0
        r_neighbour_best_pos = []
        r_neighbour_best_vals = []
        neighbour_dist = []
        for p in PList:
            if self.name == p.name:
                continue
            if neighbour_number < self.nearest_neighbours:
                neighbour_number += 1
                if p.nbest_value < p.pbest_value:
                    r_neighbour_best_pos.append(p.nbest_pos)
                    r_neighbour_best_vals.append(p.nbest_value)
                elif p.nbest_value >= p.pbest_value:
                    r_neighbour_best_pos.append(p.position)
                    r_neighbour_best_vals.append(p.pbest_value)
                #if p.name == "Particle01":
                    #print(self.position,p.position)
                neighbour_dist.append(distance.euclidean(self.position, p.position))
            elif neighbour_number == self.nearest_neighbours:
                dist = distance.euclidean(self.position, p.position)
                # If shorter distance, remove furthest neighbour and replace
                if dist < max(neighbour_dist):
                    index = neighbour_dist.index(max(neighbour_dist))
                    neighbour_dist.pop(index)
                    r_neighbour_best_vals.pop(index)
                    r_neighbour_best_pos.pop(index)
                    if p.nbest_value < p.pbest_value:
                        r_neighbour_best_pos.append(p.nbest_pos)
                        r_neighbour_best_vals.append(p.nbest_value)
                    elif p.nbest_value >= p.pbest_value:
                        r_neighbour_best_pos.append(p.position)
                        r_neighbour_best_vals.append(p.pbest_value)
                    neighbour_dist.append(distance.euclidean(self.position, p.position))
        return r_neighbour_best_pos, r_neighbour_best_vals

    # Sets nbest as the best current neighbouring position
    def set_nbest(self, neighbour_pos, neighbour_vals):
        self.nbest_value = min(neighbour_vals)
        index = neighbour_vals.index(self.nbest_value)
        self.nbest_pos = neighbour_pos[index]

    # Sets and saves nbest as the best position from neighbours history. Updates only when a better value is found
    def set_nbest_save(self, neighbour_best_pos, neighbour_best_vals):
        if self.nbest_value is None or self.nbest_value > min(neighbour_best_vals):
            self.nbest_value = min(neighbour_best_vals)
            index = neighbour_best_vals.index(self.nbest_value)
            self.nbest_pos = neighbour_best_pos[index]

    # Sets and saves nbest as the best rumoured position from neighbours. Updates only when a better value is found
    def set_nbest_rumour_save(self, r_neighbour_best_pos, r_neighbour_best_vals):
        if self.nbest_value is None or self.nbest_value > min(r_neighbour_best_vals):
            self.nbest_value = min(r_neighbour_best_vals)
            index = r_neighbour_best_vals.index(self.nbest_value)
            self.nbest_pos = r_neighbour_best_pos[index]

    def update_repulsion_velocity(self, neighbour_pos):
        for particle in neighbour_pos:
            vector = self.position - particle
            dist = distance.euclidean(self.position, particle)
            rep_vel = ((self.alpha_r / dist) ** d * (vector / dist)) * self.rep_weight
            self.velocity += rep_vel

    def update_pso_velocity(self):
        self.velocity = chi*((w * self.velocity) + (c1 * random.random() * (self.pbest_position - self.position)) +
                             (c2 * random.random() * (self.nbest_pos - self.position)))
    def update_repulsion_boundary(self):

        """
        To add repulsion from the boundary lines 
        1.x=-10
        2.x=10
        3.y=-10
        4.y=10
        """
        plist = [np.array([11.0,self.position[1]]),np.array([-11.0,self.position[1]]),np.array([self.position[0],-12.5]),np.array([self.position[0],12.5])]
        for particle in plist:
            vector = self.position - particle
            dist = distance.euclidean(self.position, particle)
            rep_vel = ((0.1/ dist) ** d * (vector / dist)) * 1
            self.velocity += rep_vel

        

    def get_waypoint(self):
        if np.linalg.norm(self.velocity) > self.max_speed:
            self.velocity = self.max_speed / np.linalg.norm(self.velocity) * self.velocity
        waypoint = self.position + self.velocity
        return waypoint

        #print(self.name,self.position,self.velocity)

    def position_update(self,waypoint):
        """Inverse kinematics model for a diff drive robot.
           R = instantaneous radius of curvature
           xicc,yicc = centre of curve
           w= angular velocity in radians 

        """
        #trial=[]
        #trial1=[]
        prev_error=0.
        sum_error=0.

        """
        To replicate the rate of lower level controller being higher than the higher level controller in real bots, a for loop 
        is run for 1000 times.
        """
        for i in range(0,1000):
            linearspeed = 2.5
            kp=1.5
            kd=2.0
            ki=0.001
            deltat=0.0001
            """
            plist = [np.array([3.5,self.position[1]]),np.array([-3.5,self.position[1]]),np.array([self.position[0],-4.5]),np.array([self.position[0],4.5])]
            for particle in plist:
                vector = self.position - particle
                dist = distance.euclidean(self.position, particle)
                if dist>1:
                    rep_vel=0
                else:
                    rep_vel = ((0.3 / dist) ** d * (vector / dist)) * 1
                waypoint += rep_vel"""
                
            
            """if 3.5-self.position[0]<1.5:
                waypoint+=np.array([-0.5,0.])*1/((3.5-self.position[0])**2)
            if self.position[0]+3.5<1.5:
                waypoint+=np.array([0.5,0.])*1/((self.position[0]+3.5)**2)
            if 4.5-self.position[1]<1.5:
                waypoint+=np.array([0.,-0.5])*1/((4.5-self.position[1])**2)
            if self.position[1]+4.5<1.5:
                waypoint+=np.array([0.,0.5])*1/((self.position[1]+4.5)**2)
            """
            angle_error = self.heading - 180/pi*atan2(waypoint[1]-self.position[1],waypoint[0]-self.position[0])
            #print(angle_error,"initial_angle_error")
            
            if angle_error<=180:
                l_vel = linearspeed+(kp*angle_error)+kd*(angle_error-prev_error)
                r_vel = linearspeed-(kp*angle_error)
            elif angle_error>180:
                angle_error= 360-angle_error
                l_vel = linearspeed-(kp*angle_error)
                r_vel = linearspeed+(kp*angle_error)
            #print(angle_error)
            if abs(l_vel-r_vel)>0.1:
                R= 2.5*(l_vel+r_vel)/-(l_vel-r_vel)
                w= -(l_vel-r_vel)/5
                xicc = self.position[0]-R*sin(self.heading)
                yicc = self.position[1]+R*cos(self.heading)
                self.position[0]=(self.position[0]-xicc)*cos(w*deltat)-(self.position[1]-yicc)*sin(w*deltat)+xicc
                self.position[1]=(self.position[0]-xicc)*sin(w*deltat)+(self.position[1]-yicc)*cos(w*deltat)+yicc
                self.heading +=(w*deltat*180/pi)
                if self.heading>=360:
                    self.heading = 360- self.heading
                elif self.heading<0:
                    self.heading = 360+self.heading
            else:
                self.position[0]+=(l_vel*deltat)*cos(self.heading)
                self.position[1]+=(r_vel*deltat)*sin(self.heading)
            prev_error=angle_error
            sum_error+=angle_error
            #if np.linalg.norm(waypoint-self.position)<0.1:
                #print("Hooooooorayyyyyyyyy reached")

            #trial.append(self.position[0])
            #trial1.append(self.position[1])
        #return trial,trial1
            
            

        #print(self.position,self.velocity,self.name)




# Initialise particle objects for each particle
#alpha_r_list=[0.36,0.32]
#alpha_r_list=[0.14]
alpha_r_list=[0.02]
for o in range(16,21):
    for alpha_r in alpha_r_list:
        particle_list=[]
        position_list=[]
        particle_x=[-1.5,-1.0,-0.5,0.,0.5,1.0,1.5,2.0,2.5,3.0]
        particle_y=[0.0,-0.5,0.5,-1.0,1.0,-1.5,1.5,-2.0,2.0,2.5]
        for i in particle_y:
            for j in particle_x:
                position_list.append([j,i])

        #print(particle_list)
        for i in range(n_particles):
            particle = Particle('Particle%02d' % i, search_func= search_space,k=n_neighbours, max_speed=max100, d=d, alpha_r=alpha_r,
                                repulsion=repulsion, rep_weight=weight)
            particle.position=np.array(position_list[i])

            particle_list.append(particle)


        iteration = 0
        if rule == 'local':
            pos_history=[]
            #for particle in particle_list:
                #print(particle.position,particle.name)
            # Particle goes towards personal best and best current value in the neighbourhood
            #f= open('PSOsimulationlog'+str(n_neighbours)+str(weight)+'.txt','a+')
            #for i in range(iter_max):
            #   pos_history.append([])
            f=open("/home/hari/RealPso/n_particles"+str(n_particles)+"/Trial"+str(o)+"/RealPSOlog"+"k="+str(n_neighbours)+"alpha_r="+str(alpha_r)+".txt","a+")
            while iteration <= iter_max:
                pos_current = []
                f.write("Iteration:"+str(iteration)+"\r\n")
                #print(pos_history,"pos_history")
                for particle in particle_list:
                    particle.fitness()
                    particle.set_pbest()
                    #print (particle.name,iteration)
                    neighbour_pos, neighbour_vals = particle.get_neighbours(particle_list,iteration)
                    particle.set_nbest(neighbour_pos, neighbour_vals)
                    particle.update_pso_velocity()
                    particle.update_repulsion_velocity(neighbour_pos)
                    particle.update_repulsion_boundary()
                    f.write("{Name:"+str(particle.name)+"; position:"+str(particle.position)+"; velocity:"+str(particle.velocity)+"}"+"\r\n")
                    # Report to me
                    if particle.fit < gbest_val:
                        gbest_prev = gbest_val
                        gbest_val = particle.fit
                        gbest_pos = particle.position
                        print('gbest update')
                        print('value: ', gbest_val)
                        print('position: ', gbest_pos)
                        print('Iteration: ', iteration)
                        print('-------------------------------------------------')
                    a=list(particle.position)
                    pos_current.append(a[:])
                #pos_history1.append(pos_current)
                #print(pos_history1,"pos_history1",iteration)
                    #print(pos_history,"pos_history",particle.name)    
                pos_history.append(pos_current)
                #print(pos_current,iteration,"pos_current")
                
                #print(pos_history,"pos_history",iteration)
                for particle in particle_list:
                    
                    waypoint = particle.get_waypoint()
                    #waypoint = np.array([1.,1.])
                    particle.position_update(waypoint)
                    #print(pos_current,"pos_current")
                #distances = []
                #rms_dist=0.
                #centroid = np.array([0.0,0.0])
                #for particle in particle_list:
                #    distances.append(np.linalg.norm(particle.position-np.array([-2.90354,-2.90354])))
                #    centroid+=particle.position
                #distances = np.array(distances)
                #centroid = centroid/len(particle_list)
                #centroid_dist = np.linalg.norm(centroid-np.array([-2.90354,-2.90354])) 
                #for i in distances:
                #    rms_dist+=i**2
                #rms_dist=rms_dist/len(particle_list)

                
                #f.write("iteration:"+str(iteration)+";"+"average_distance:"+str(np.average(distances))+";"+"centroid distance:"+str(centroid_dist)+"\r\n")
                #f.write("minimum_distance:"+str(min(distances))+";"+"maximum_distance:"+str(max(distances))+"RMS distance:"+str(rms_dist)+"\r\n")

                if iteration == iter_max:
                    print('Solution: ', gbest_pos)
                    print('gbest: ', gbest_val)
                    print('Iterations: ', iteration)
                    #print(pos_history)
                    f.close()
                    break

                iteration += 1
                #f.close()

        # Plotting
        #print (trial,trial1,"trials")
        fig = plt.figure()
        if search_space == 'easom':
            ax = plt.axes(xlim=(-10, 10), ylim=(-10, 10), title='Modified nBest PSO \n K = 7')
        else:
            ax = plt.axes(xlim=(-11.0, 11.0), ylim=(-12.5,12.5), title='Modified nBest PSO \n K = 10')
        xdata, ydata = [], []
        points, = ax.plot([], [], 'wx')

        #lines = ax.plot([], [], 'bs',linestyle='-',color="r", markersize=3,linewidth=2)[0]

        # Rastrigin search space
        if search_space == 'rastrigin':
            x = np.linspace(-5.12, 5.12, 100)
            y = np.linspace(-5.12, 5.12, 100)
            X, Y = np.meshgrid(x, y)

            Z = (X ** 2 - 10 * np.cos(2 * np.pi * X)) + \
                (Y ** 2 - 10 * np.cos(2 * np.pi * Y)) + 20

        elif search_space == 'st':
            x = np.linspace(-11.0, 11.0, 100)
            y = np.linspace(-12.5, 12.5, 100)
            X, Y = np.meshgrid(x, y)
            X=X*(10.0/16)
            Y=Y*(10.0/25)
            Z = 0.5 * ((X ** 4 - 16 * X ** 2 + 5 * X) + (Y ** 4 - 16 * Y ** 2 + 5 * Y))

        elif search_space == 'easom':
            x = np.linspace(-10, 10, 200)
            y = np.linspace(-10, 10, 200)
            X, Y = np.meshgrid(x, y)

            Z = -np.cos(X) * np.cos(Y) * np.exp(-((X - np.pi) ** 2 + (Y - np.pi) ** 2))+2*(-np.cos(X+5) * np.cos(Y) * np.exp(-((X+5 - np.pi) ** 2 + (Y - np.pi) ** 2)))

        cont = ax.contourf(x, y, Z,30)

        X_pos = []
        Y_pos = []

        """

        f= open("psotest.txt","a+")
        j=0
        for i in pos_history:
            f.write(str(j)+"\r\n")
            j+=1
            f.write(str(i)+"\r\n")
        f.close()
        """



        for timestep in pos_history:
            X_pos = []
            Y_pos = []
            for position in timestep:
                X_pos.append(position[0])
                Y_pos.append(position[1])
            xdata.append(X_pos)
            ydata.append(Y_pos)
        average_arr=[]
        
        f=open("/home/hari/RealPso/n_particles"+str(n_particles)+"/Trial"+str(o)+"/RealPSO"+"k="+str(n_neighbours)+"alpha_r="+str(alpha_r)+".txt","a+")
        for i in range(len(xdata)):
            distarr=[]
            for j in range(len(xdata[i])):
                dist = np.array([xdata[i][j],ydata[i][j]])-np.array([-4.6456,-7.25875])
                dist1=np.linalg.norm(dist)
                distarr.append(dist1)
            average_distance=np.average(np.array(distarr))
            average_arr.append(average_distance)
            f.write("Iteration:"+str(i)+"\r\n")
            f.write("average_distance:"+str(average_distance)+"\r\n")
        f.close()
        

        """
        for i in range(len(xdata[-1])):
            dist = np.array([xdata[-1][i],ydata[-1][i]])-np.array([-2.90354,-2.90354])
            dist1= np.linalg.norm(dist)
            distarr.append(dist1)
        average_distance = np.average(np.array(distarr))
        minimum_distance = min(distarr)
        maximum_distance = max(distarr)

        print(average_distance,minimum_distance,maximum_distance)

        avg_distancearr.append(average_distance)
        min_distarr.append(minimum_distance)
        max_distarr.append(maximum_distance)
        """

        #avg_distancereparr.append(avg_distancearr)
        #min_distreparr.append(min_distarr)
        #max_distreparr.append(max_distarr)



        def init():
            points.set_data([], [])
            #lines.set_xdata([])
            #lines.set_ydata([])
            return points,


        def animate(frame):
            x = xdata[frame]
            y = ydata[frame]
            points.set_data(x, y)
            #lines.set_xdata(trial)
            #lines.set_ydata(trial1)
            return points,

        Writer = animation.writers['ffmpeg']
        if show_ani is True:
            anim = animation.FuncAnimation(fig, animate, init_func=init, frames=iteration, blit=True)
            #plt.scatter([-0.5],[0.0])
            #plt.plot(trial,trial1)
            #plt.show()

        if save is True:
            anim.save('/home/hari/RealPso/n_particles'+str(n_particles)+'/Trial'+str(o)+'/RealPSO'+"k="+str(n_neighbours)+"alpha_r="+str(alpha_r)+'.mp4', writer="ffmpeg")
        figure_save=False

        if figure_save is True:
            plt.clf()
            plt.plot(range(iter_max+1),average_arr,label="Average distance to optimum")
            fig_22=plt.gcf()
            fig_22.savefig('/home/hari/RealPSo/Trial5'+str(o)+'/RealPSO'+"k="+str(n_neighbours)+"alpha_r="+str(alpha_r)+".png")
        """
        if alpha_r<1.0:
            alpha_r+=0.02
            alpha_r+=0.02
        else:
            alpha_r+=0.02

    alpha_r=0.02
    """
