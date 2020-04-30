
# coding: utf-8

# # Assignment 6

# *Zuzanna Materny Magdalena Wolniaczyk*

# Reynold’s  boids  flocking  model  

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random
import math
import imageio
import os
import shutil
import glob


# In[10]:


def normalize(vector, length):
    n = np.sqrt(vector[0] ** 2 + vector[1] ** 2) / length
    return vector[0] / n, vector[1] / n

def neighborhood(boid, flock):
    view_range = boid.view_range
    radius = boid.radius
    neighbors = []
    direction = normalize([boid.vx, boid.vy], 1)
    my_angle = math.atan2(direction[1], direction[0])
    start_angle = (my_angle - view_range / 2)   # actual view range
    end_angle = (my_angle + view_range / 2)
    for neighbor in flock:
        if neighbor != boid:
            angle = (math.atan2(neighbor.y - boid.y, neighbor.x - boid.x))  # angle between my direction and neighbor
            polar_radius = math.sqrt((neighbor.x - boid.x) ** 2 + (neighbor.y - boid.y) ** 2)
            if start_angle < angle < end_angle and polar_radius < radius:
                neighbors.append(neighbor)  # if boid is in my view and is close enough, i treat it like my neighbor
    return neighbors

def separation(boid, neighbors, separation_strength = 0.9):
    safety_zone = boid.safety_zone
    resultant_x = 0
    resultant_y = 0
    counter = 0
    for neighbor in neighbors:
        difference_x = neighbor.x - boid.x   # (q_i_x) checking distance to my neighbor 
        difference_y = neighbor.y - boid.y
        distance = math.sqrt(difference_x ** 2 + difference_y ** 2)
        if distance < safety_zone:   # if neighbor is in my safety zone
            counter += 1
            resultant_x -= difference_x / distance   # dividing by norm, creating a vector of separation
            resultant_y -= difference_y / distance
    if counter != 0:
        resultant_x /= counter   # divided by number of neighbors in boid's safety zone 
        resultant_y /= counter
    vs_x = separation_strength * resultant_x   # scaling using separation coefficient
    vs_y = separation_strength * resultant_y
    return vs_x, vs_y

def cohesion(boid, neighbors, cohesion_strength = 0.9):
    resultant_x = 0
    resultant_y = 0
    counter = 0
    for neighbor in neighbors:
        counter += 1
        resultant_x += neighbor.x   # taking sum of neighbors positions
        resultant_y += neighbor.y
    if counter != 0:
        resultant_x /= counter    # dividing by number of neighbors
        resultant_y /= counter
        vc_x = cohesion_strength * (resultant_x - boid.x)   # facing to the center and scaling using cohesion coefficient
        vc_y = cohesion_strength * (resultant_y - boid.y)
    else:
        vc_x, vc_y = 0, 0
    return vc_x, vc_y

def alignment(neighbors, alignment_strength = 0.9):
    resultant_vx = 0
    resultant_vy = 0
    counter = 0
    for neighbor in neighbors:
        counter += 1
        resultant_vx += neighbor.vx   # taking sum of neighbors velocities directions
        resultant_vy += neighbor.vy
    if counter != 0:
        resultant_vx /= counter   # dividing by number of neighbors
        resultant_vy /= counter
    va_x = alignment_strength * resultant_vx   # scaling using alignment coefficient
    va_y = alignment_strength * resultant_vy
    return va_x, va_y 


# In[11]:


class Boid:
    def __init__(self, view_range = math.pi, radius = 20, safety_zone = 2, v_max = 1, grid_x = 200, grid_y = 200):
        self.grid_x = grid_x 
        self.grid_y = grid_y
        self.x = random.uniform(0, self.grid_x)
        self.y = random.uniform(0, self.grid_y)
        self.vx = random.uniform(-1, 1)
        self.vy = random.uniform(-1, 1)
        self.radius = radius
        self.view_range = view_range
        self.safety_zone = safety_zone
        self.v_max = v_max


# In[12]:


def visualization(steps=10, num_of_boids=50, view_range=math.pi, radius=20, safety_zone=10, v_max=1, grid_x=200, grid_y=200, 
                  alignment_strength = 0.9, cohesion_strength = 0.9, separation_strength = 0.9):
    plt.figure(figsize=(15,15))
    flock = []
    folder_path = "Boids_flocking_model"

    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)

    frames_path = folder_path + "/{i}.png"

    for i in range(num_of_boids):
        flock.append(Boid(view_range, radius, safety_zone, v_max, grid_x, grid_y))
   
    for k in range(steps):
        plt.clf()
        for boid in flock:
            neighbors = neighborhood(boid, flock)
            ali = alignment(neighbors, alignment_strength)
            coh = cohesion(boid, neighbors, cohesion_strength)
            sep = separation(boid, neighbors, separation_strength)
            new_vx = boid.vx + coh[0] + ali[0] + sep[0]           # update velocity
            new_vy = boid.vy + coh[1] + ali[1] + sep[1]
            if np.sqrt(new_vx ** 2 + new_vy ** 2) > boid.v_max:
                new_v = normalize([new_vx, new_vy], boid.v_max)   # (and normalize if necessary)
                boid.vx = new_v[0]
                boid.vy = new_v[1]
            else:
                boid.vx = new_vx
                boid.vy = new_vy

            new_x = boid.x + boid.vx             # update position
            new_y = boid.y + boid.vy
            if new_x < 0:                        # periodic boundary condition
                new_x = boid.grid_x + new_x
            if new_x > boid.grid_x:
                new_x = new_x - boid.grid_x
            if new_y < 0:
                new_y = boid.grid_y + new_y
            if new_y > boid.grid_y:
                new_y = new_y - boid.grid_y
            boid.x = new_x
            boid.y = new_y

            direction = normalize([boid.vx, boid.vy], 1)
            plt.quiver(boid.x, boid.y, direction[0], direction[1],angles='xy', scale=80, norm=True, color='DeepPink')
            plt.xlim(0, boid.grid_x)
            plt.ylim(0, boid.grid_y)
            plt.title('Boids flocking model simulation\nview range={a}, radius={b}, safety zone={c}, separation={d}, cohesion={e}, alignment={f}'.format(a=round(view_range,2), b=radius, c=safety_zone, d=separation_strength, e=cohesion_strength, f=alignment_strength),         
                      fontsize=15)
        plt.savefig(frames_path.format(i=k))
        #plt.show()
        
    animation_path = 'boids_flocking_simulation_view_range={a}_radius={b}_safety_zone={c}_separation={d}_cohesion={e}_alignment={f}.gif'.format(a=round(view_range,2), b=radius, c=safety_zone, d=separation_strength, e=cohesion_strength, f=alignment_strength)
    with imageio.get_writer(animation_path, mode='I') as writer:
        for i in range(steps):
            writer.append_data(imageio.imread(frames_path.format(i=i)))
        


# In[17]:


visualization(100, 300, view_range=math.pi, radius=5, safety_zone=4, 
              separation_strength=0.8, cohesion_strength = 0.3, alignment_strength = 0.2,
             grid_x=50, grid_y=50)

