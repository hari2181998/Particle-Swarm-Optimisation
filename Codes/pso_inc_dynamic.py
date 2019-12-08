
"""
PSO dynamic case simulation with kinematic constraints for all the particles and initial position of those 
used in real experiment
"""
from collections import deque
import random
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib as mpl
from math import pi
from math import sin,cos,atan2
from matplotlib.gridspec import GridSpec
import itertools
mpl.rc('figure', max_open_warning = 0)#To prevent matplotlib warning which shows up when more than 20 figures are created without clearing 

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
n_neighbours=49

weight=1

w = 0.5
c1 = 0.
c2 = 1.0
chi = 1
max100 = 10  # Number of "units" a particle can move in 100 iterations
runs = np.linspace(1, 100, 100)



rule = 'local'
rule_list = ['pso', 'local', 'gbest rumour', 'gbest conf']
assert rule in rule_list, 'Invalid rule'

search_space = 'easom'
search_spaces = ['rastrigin', 'st', 'easom','onemin']

movement = "Rotate"
movements=["Rotate","Oscillate"]
frequency=75

show_ani = False
save = False
save_data = False

iter_max = 10
n_particles = 50

repulsion = True
d = 1
rep_radius = 3
S = np.pi * rep_radius ** 2
alpha_r = 0.14

gbest_val = 90  #random initialisation of gbest_val
gbest_prev = 190


pos_current = []
pos_history = []


class Particle:
    def __init__(self, name,search_func,k=7, max_speed=5, memory=float('inf'), d=6, alpha_r=0.25, repulsion=True, rep_weight=1.,movement="None",frequency="0"):
        self.name = name
        self.nearest_neighbours = k
        self.search_space = search_func
        self.repulsion_neighbours = 15  #Number of neighbours to be chosen for repulsion factor
        self.heading = 90    #Initial heading assumed to be facing Y axes for including the robot kinematics
        self.d = d     
        self.alpha_r = alpha_r
        self.repulsion = repulsion
        self.rep_weight = rep_weight
        self.it_no = 0
        self.frequency=frequency
        self.movement=movement

        if self.movement == "None":
            if self.search_space == 'rastrigin':
                self.position = np.array([random.uniform(-5.12, 5.12), random.uniform(-5.12, 5.12)])
                self.fit = 20 + (self.position[0] ** 2 - 10 * np.cos(2*np.pi*self.position[0])) + \
                           (self.position[1] ** 2 - 10 * np.cos(2*np.pi*self.position[1]))
            elif self.search_space == 'st':
                self.position = np.array([random.uniform(-5, 5), random.uniform(-5, 5)])
                self.fit = 0.5 * ((self.position[0] ** 4 - 16 * self.position[0] ** 2 + 5 * self.position[0]) +
                                  (self.position[1] ** 4 - 16 * self.position[1] ** 2 + 5 * self.position[1]))
            elif self.search_space == 'easom':
                self.position = np.array([random.uniform(-10, 10), random.uniform(-10, 10)])
                self.fit = -np.cos(self.position[0]) * np.cos(self.position[1]) * np.exp(-((self.position[0] - np.pi) ** 2 +
                                                                                           (self.position[1] - np.pi) ** 2))



        if self.movement=="Rotate":

            if self.search_space == "st":
                self.position = np.array([random.uniform(0, 5), random.uniform(-5, 5)])
                self.fitness()

            if self.search_space == "onemin":
                self.position = np.array([random.uniform(-8, 8), random.uniform(-10, 10)])
                self.fitness()

            if self.search_space =="easom":
                self.position = np.array([random.uniform(-8, 8), random.uniform(-10, 10)])
                self.fitness()


        self.velocity = np.array([0., 0.])
        self.max_speed = max_speed/100
        self.pbest_position = self.position
        self.pbest_value = float('inf')
        self.nbest_pos = self.position
        self.nbest_value = float('inf')

    def fitness(self):
        # a is the angle in radians the field has to be rotated recalculated at every iteration.Frequency is the 
        # number of iterations required for one complete rotation.
        a= (self.it_no%self.frequency)*(2*np.pi/self.frequency)
        x1 = self.position[0]
        x2 = self.position[1]
        if search_space == "onemin":
            """Initial field is assumed to be from -5 to 5 in both axes.To have the same field in -8 to 8 in X 
            and -10 to 10 in Y axes, it is multiplied by the respective factor.The increase in dimensions of the field
            are chosen such that the particle density(Number of particles/area of the field) remains constant.
            """
            x1=x1*5.0/8    
            x2=x2*0.5
        """
        The two lines below is to incorporate the rotation movement of the field.As when the field moves
        """
        xx=x1*cos(a)-x2*sin(a)
        yy=x1*sin(a)+x2*cos(a)

        x1=xx
        x2=yy
        if self.search_space == 'onemin':
            f=(x1+3.5)**2+(x2+3.5)**2
        if self.search_space == 'rastrigin':
            f = 20 + (x1 ** 2 - 10 * np.cos(2*np.pi*x1)) + (x2 ** 2 - 10 * np.cos(2*np.pi*x2))
        elif self.search_space == 'st':
            f = 0.5 * ((x1 ** 4 - 16 * x1 ** 2 + 5 * x1) + (x2 ** 4 - 16 * x2 ** 2 + 5 * x2))
        elif self.search_space == 'easom':
            f = -np.cos(x1) * np.cos(x2) * np.exp(-((x1 - np.pi) ** 2 + (x2 - np.pi) ** 2))
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
                continue
            if neighbour_number < self.nearest_neighbours:
                neighbour_number += 1
                neighbour_pos.append(p.position)
                neighbour_vals.append(p.fit)
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
        return neighbour_pos, neighbour_vals

    def get_repulsion_neighbours(self, PList,iteration):

        # To get the neighbours list for repulsion calculation
        neighbour_number = 0
        neighbour_pos = []
        neighbour_dist = []
        for p in PList:
            if self.name == p.name:
                continue
            if neighbour_number < self.repulsion_neighbours:
                neighbour_number += 1
                neighbour_pos.append(p.position)
                neighbour_dist.append(distance.euclidean(self.position, p.position))
            elif neighbour_number == self.repulsion_neighbours:
                dist = distance.euclidean(self.position, p.position)
                # If shorter distance, remove furthest neighbour and replace
                if dist < max(neighbour_dist):
                    index = neighbour_dist.index(max(neighbour_dist))
                    neighbour_dist.pop(index)
                    neighbour_pos.pop(index)
                    neighbour_pos.append(p.position)
                    neighbour_dist.append(distance.euclidean(self.position, p.position))
        return neighbour_pos, neighbour_vals


    
    # Sets nbest as the best current neighbouring position
    def set_nbest(self, neighbour_pos, neighbour_vals):
        self.nbest_value = min(neighbour_vals)
        index = neighbour_vals.index(self.nbest_value)
        self.nbest_pos = neighbour_pos[index]

    def update_repulsion_velocity(self, neighbour_pos):
        # For updating the repulsion velocity
        for particle in neighbour_pos:
            vector = self.position - particle
            dist = distance.euclidean(self.position, particle)
            rep_vel = ((self.alpha_r / dist) ** d * (vector / dist)) * self.rep_weight
            self.velocity += rep_vel

    def update_pso_velocity(self):
        self.velocity = chi*((w * self.velocity) + (c1 * random.random() * (self.pbest_position - self.position)) +
                             (c2 * random.random() * (self.nbest_pos - self.position)))
    def update_repulsion_boundary(self):
        # For adding repulsion from the boundaries to make sure they dont go out of the field.The positions has to be changed
        # based on the field dimensions
        if self.search_space == "onemin":
            plist = [np.array([8,self.position[1]]),np.array([-8,self.position[1]]),np.array([self.position[0],-10]),np.array([self.position[0],10])]
        elif self.search_space == "easom":
            plist = [np.array([10,self.position[1]]),np.array([-10,self.position[1]]),np.array([self.position[0],-10]),np.array([self.position[0],10])]

        for particle in plist:
            vector = self.position - particle
            dist = distance.euclidean(self.position, particle)
            rep_vel = ((0.1/ dist) ** d * (vector / dist)) * 1
            self.velocity += rep_vel

        

    def get_waypoint(self):
        # Getting the waypoint from the velocity calculated to be used by the position update function.
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

           Implementing P controller for go to goal problem assuming particle as a differential drive robot.
           The linear speed for both the wheels is fixed and does not vary based on the waypoint location.
           The angle error in degrees is calculated based on the heading and target position.A value proportional to 
           the error is subtracted from one wheel and added to other wheel based on the error.
           The loop runs for 1000 times with delta time =0.0001 for every iteration.So for every particle iteration in 
           the main loop, the time is assumed to be 0.1 seconds.

        """
        prev_error=0. # not used currently but for implementing PID control.
        sum_error=0.
        for i in range(0,1000):
            linearspeed = 2.5
            kp=1.5
            kd=2.0
            ki=0.001
            deltat=0.0001
            angle_error = self.heading - 180/pi*atan2(waypoint[1]-self.position[1],waypoint[0]-self.position[0])
            """
            To find the direction the bot needs to turn based on the shortest angle.
            Example, the error is usually calculated with respect to positive X axis so it might go beyond 180 degrees.
            But for error greater than 180 degrees, turning in the opposite direction is the best.
            """
            if angle_error<=180:
                l_vel = linearspeed+(kp*angle_error)
                r_vel = linearspeed-(kp*angle_error)
            elif angle_error>180:
                angle_error= 360-angle_error
                l_vel = linearspeed-(kp*angle_error)
                r_vel = linearspeed+(kp*angle_error)
            
            """
            Position update equations based on Kinematic model of diff drive robot.If the wheel velocities
            for both the wheels are equal or very less, another set of equation assuming straight line motion
            are used.
            The heading of the robot is also updated 
            """
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

    def reset(self):
        #Resetting the values every iteration
        self.pbest_position=self.position
        self.pbest_value=self.fit
        self.nbest_value=float('inf')



# Initialise particle objects for each particle

for run in runs:
        particle_list = []
        for i in range(n_particles):
            particle = Particle('Particle%02d' % i, search_func= search_space,k=n_neighbours, max_speed=max100, d=d, alpha_r=alpha_r,
                                repulsion=repulsion, rep_weight=weight,frequency=frequency,movement="Rotate")

            particle_list.append(particle)


        iteration = 0
        if rule == 'local':
            pos_history=[]
            vel_history=[]
            #Writing to a text file, the position and velocity of each particle at every iteration
            f= open("/home/hari/DynamicPsocorrelation"+"/DynamicPSOlog"+str(n_neighbours)+"freq"+str(frequency)+"rep"+str(alpha_r)+"Trial"+str(run)+'.txt','a+')
            while iteration <= iter_max:
                pos_current = []
                vel_current = []
                f.write("Iteration:"+str(iteration)+"\r\n")
                for particle in particle_list:
                    particle.fitness()
                    particle.set_pbest()
                    neighbour_pos, neighbour_vals = particle.get_neighbours(particle_list,iteration)
                    rep_pos,rep_vals = particle.get_repulsion_neighbours(particle_list,iteration)
                    particle.set_nbest(neighbour_pos, neighbour_vals)
                    particle.update_pso_velocity()
                    particle.update_repulsion_velocity(rep_pos)
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
                pos_history.append(pos_current)

                for particle in particle_list:
                    
                    waypoint = particle.get_waypoint()
                    b=list(waypoint)
                    vel_current.append(particle.velocity)
                    particle.position_update(waypoint)
                    particle.it_no+=1
                    particle.reset()
                vel_history.append(vel_current)
                    

                if iteration == iter_max:
                    print('Solution: ', gbest_pos)
                    print('gbest: ', gbest_val)
                    print('Iterations: ', iteration)
                    
                    break

                iteration += 1
            
            f.close()    

        # Plotting
        
        fig = plt.figure()
        if search_space == 'easom':
            ax = plt.axes(xlim=(-10, 10), ylim=(-10, 10), title='Modified nBest PSO \n K = 7')
        else:
            ax = fig.add_subplot(1, 1, 1, aspect='equal')
            ax = plt.axes(xlim=(-8, 8), ylim=(-10,10), title='DynamicPSO \n K ='+str(n_neighbours)+'f='+str(frequency))
            
        xdata, ydata = [], []
        points, = ax.plot([], [], 'wx')
        lines=[]

        colors=['b','g','r','c','m','y','k','w','b','g']

        for i in range(n_particles):
            lines.append(ax.plot([], [], 'bs',linestyle='-',color=colors[i%10], markersize=2,linewidth=1.0)[0])

        """
        To calculate the global minimum position at every iteration
        """
        if search_space == "onemin":
            min_x=-5.6
            min_y=-7
            minloc=[]
            for i in range(iter_max+1):
                a=(i%frequency)*(2*np.pi/frequency)
                x_min=min_x*np.cos(a)-min_y*np.sin(a)
                y_min = min_x*np.sin(a)+min_y*np.cos(a)

                aa = np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
                bb= np.array([min_x,min_y])
                minloc.append(np.linalg.solve(aa,bb))
        if search_space == "easom":
            min_x= np.pi
            min_y= np.pi
            minloc=[]
            for i in range(iter_max+1):
                a=(i%frequency)*(2*np.pi/frequency)
                x_min=min_x*np.cos(a)-min_y*np.sin(a)
                y_min = min_x*np.sin(a)+min_y*np.cos(a)

                aa = np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
                bb= np.array([min_x,min_y])
                minloc.append(np.linalg.solve(aa,bb))







        X_pos = []
        Y_pos = []
        for timestep in pos_history:
            X_pos = []
            Y_pos = []
            for position in timestep:
                X_pos.append(position[0])
                Y_pos.append(position[1])
            xdata.append(X_pos)
            ydata.append(Y_pos)
        average_arr=[]
        #TO calculate the average distance at every iteration and store it in a text file
        f=open("/home/hari/DynamicPsocorrelation"+"/DynamicPSOK="+str(n_neighbours)+"f="+str(frequency)+"rep"+str(alpha_r)+"Trial"+str(run)+".txt","a+")
        for i in range(len(xdata)):
            distarr=[]
            for j in range(len(xdata[i])):
                dist = np.array([xdata[i][j],ydata[i][j]])-np.array(minloc[i])
                dist1=np.linalg.norm(dist)
                distarr.append(dist1)
            average_distance=np.average(np.array(distarr))
            average_arr.append(average_distance)
            f.write("Iteration:"+str(i)+"\r\n")
            f.write("average_distance:"+str(average_distance)+"\r\n")
        f.close()

        #To animate for the first two runs
        if run<3:
            def init():
                points.set_data([], [])
                return points,


            # TO calculate the field values at every iteration
            Z=[]
            for i in range(iter_max+1):
                if search_space == "onemin":
                    x = np.linspace(-8, 8, 100)
                    y = np.linspace(-10, 10, 100)
                    X, Y = np.meshgrid(x, y)
                    X=X*5/8
                    Y=Y*5/10
                    #frequency=10
                    a=(i%frequency)*(2*np.pi/frequency)
                    #print(a,i)
                    aa=X*np.cos(a)-Y*np.sin(a)
                    bb=X*np.sin(a)+Y*np.cos(a)

                    X=aa
                    Y=bb
                    if search_space =='st':
                        Z .append(0.5 * ((X ** 4 - 16 * X ** 2 + 5 * X) + (Y ** 4 - 16 * Y ** 2 + 5 * Y)))
                    elif search_space == 'onemin':
                        Z.append((X+3.5)**2+(Y+3.5)**2)
                elif search_space == "easom":
                    x = np.linspace(-10, 10, 100)
                    y = np.linspace(-10, 10, 100)
                    X, Y = np.meshgrid(x, y)
                    X=X
                    Y=Y
                    #frequency=10
                    a=(i%frequency)*(2*np.pi/frequency)
                    #print(a,i)
                    aa=X*np.cos(a)-Y*np.sin(a)
                    bb=X*np.sin(a)+Y*np.cos(a)

                    X=aa
                    Y=bb
                    Z.append(-np.cos(X ) * np.cos(Y) * np.exp(-((X- np.pi) ** 2 + (Y-np.pi) ** 2)))


            #To initialise deque for plotting the trail of each particle
            xposes=[]
            yposes=[]
            for i in range(n_particles):
                xxx=deque(maxlen=20)
                yyy=deque(maxlen=20)
                xposes.append(xxx)
                yposes.append(yyy)
            

            def animate(frame):
                """
                Clearing the figure and reinitialising every iteration because if the plot is
                not cleared at every iteration,the contour plot is just overwritten which consumes a lot of 
                space and takes a very long time when saving the animation.
                """
                plt.clf()
                ax = fig.add_subplot(1, 1, 1, aspect='equal')
                if search_space == "onemin":
                    ax = plt.axes(xlim=(-8, 8), ylim=(-10,10), title='DynamicPSO \n K ='+str(n_neighbours)+'f='+str(frequency)+"rep"+str(alpha_r))
                elif search_space == "easom":
                    ax = plt.axes(xlim=(-10, 10), ylim=(-10,10), title='DynamicPSO \n K ='+str(n_neighbours)+'f='+str(frequency)+"rep"+str(alpha_r))

                lines=[]
                points, = ax.plot([], [], 'wx')
                for i in range(n_particles):
                    color=next(palette)
                    lines.append(ax.plot([], [], 'bs',linestyle='-',color=colors[i%10], markersize=0.5,linewidth=0.1)[0])
                

                xx = xdata[frame]
                yy = ydata[frame]
                points.set_data(xx, yy)
                
                for i in range(n_particles):
                    if frame<1:
                        xposes[i].append(xx[i])
                        yposes[i].append(yy[i])
                    else:
                        xposes[i].append(xx[i])
                        yposes[i].append(yy[i])
                        if frame!=(iter_max-1):
                            lines[i].set_xdata(xposes[i])
                            lines[i].set_ydata(yposes[i])
                cont = ax.contourf(x, y, Z[frame],30,cmap="RdGy")
                return points,

            Writer = animation.writers['ffmpeg']
            if show_ani is True:
                anim = animation.FuncAnimation(fig, animate, init_func=init, frames=iteration)
                

            if save is True:
                anim.save("/home/hari/DynamicPsocorrelation"+"/DynamicPSOK="+str(n_neighbours)+"f="+str(frequency)+"rep"+str(alpha_r)+"Trial"+str(run)+".mp4", writer="ffmpeg")
            figure_save=False

            if figure_save is True:
                plt.clf()
                plt.plot(range(iter_max+1),average_arr,label="Average distance to optimum")
                fig_22=plt.gcf()
                fig_22.savefig('/home/hari/RealPSo/Trial5'+str(o)+'/RealPSO'+"k="+str(n_neighbours)+"alpha_r="+str(alpha_r)+".png")


        if save_data:
            import os

            path = 'logs/' + movement + '_k' + str(n_neighbours) +"Freq"+str(frequency)+ '/trial%02d' % run

            if os.path.exists(path):
                continue
            else:
                os.makedirs(path, 0o777)
                print('Directory Created')

            for i in range(iter_max):
                mean_velocity = np.mean(vel_history[i], axis=0)
                ms = 0
                x_min = float('inf')
                x_max = -float('inf')
                y_min = float('inf')
                y_max = -float('inf')
                for j in range(n_particles):
                    ms += np.linalg.norm(vel_history[i][j] - mean_velocity) ** 2

                    # For scaling purposes
                    if pos_history[i][j][0] >= x_max:
                        x_max = pos_history[i][j][0]
                    if pos_history[i][j][0] <= x_min:
                        x_min = pos_history[i][j][0]
                    if pos_history[i][j][1] >= y_max:
                        y_max = pos_history[i][j][1]
                    if pos_history[i][j][1] <= y_min:
                        y_min = pos_history[i][j][1]

                x_range = x_max - x_min
                y_range = y_max - y_min
                norm_factor = max(x_range, y_range)

                for j in range(n_particles):
                    dphi = (vel_history[i][j] - mean_velocity) / np.sqrt(ms/n_particles)
                    if x_range >= y_range:
                        x_pos = (pos_history[i][j][0] - x_min) / x_range
                        y_pos = (pos_history[i][j][1] - y_min) / y_range
                    else:
                        x_pos = (pos_history[i][j][0] - x_min) / x_range
                        y_pos = (pos_history[i][j][1] - y_min) / y_range

                    data = (x_pos, y_pos, vel_history[i][j][0],
                            vel_history[i][j][1], dphi[0], dphi[1])

                    file_name = os.path.join(path, movement + str(i) + '.csv')
                    new_file = False
                    if not os.path.isfile(file_name):
                        new_file = True
                    with open(file_name, 'a') as fd:
                        if new_file:
                            header = "x,y,vx,vy,dpx,dpy"
                            fd.write(header)
                        fd.write('\n' + ','.join(str(d) for d in data))

        print('Run %01d Done' % run)