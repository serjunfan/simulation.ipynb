#!/usr/bin/env python
# coding: utf-8

# In[20]:


import random
from math import dist
import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
import math
np.random.seed(1)
simulation_time = 3*60*60
area_width = 500.
area_height = 500.
cluster_num = 5
cluster_R = 1.
node_num = 1
alpha = 1.75
alpha_min = 1.
alpha_max = 500
beta = 1.8
beta_min = 1
beta_max = 60*60*3
side_for_clus = min(area_width, area_height)
TPL = lambda ALPHA, MIN, MAX, SAMPLES: ((MAX ** (ALPHA+1.) -1.) * rand(*SAMPLES.shape) +1.) ** (1./(ALPHA+1.))
def generate_angle():
    return np.random.uniform(0,360)
def power_law(k,a,m):
    return k*(m**a)
def prob_explore( node_visited_num):
    return 0.6 * (node_visited_num ** -0.21 )
def set_pos_of_clus(n = cluster_num , r = cluster_R, side = side_for_clus):
    output = []
    i = 0
    while i < n:
        qualified = True
        x = np.random.uniform(0+r,side-r)
        y = np.random.uniform(0+r,side-r)
        point = [x,y]
        for node in output:
            if dist(point,node) < r:
                qualified = False
        if qualified:
            i += 1
            output.append(point)
    return np.array(output)
def set_clus_prob(n = cluster_num):
    output = np.random.dirichlet(np.ones(n),size=1)
    return output

class Cluster():
    def __init__(self):
        self.data = set_pos_of_clus()
        self.prob = set_clus_prob()
        self.r = [cluster_R for _ in range(cluster_num)]
    def getr(self):
        return self.r
    def getdata(self):
        return self.data
    def getclus_prob(self):
        return self.prob
def assign_cluster( prob_list, nn = node_num, cn = cluster_num):
    return(np.random.choice( cn, nn, replace=True, p=prob_list[0]) )
    
def set_node_data(clusdata, clusprob, nn = node_num, r = cluster_R ): #data[node_index] = [x,y] 
    clus_index = assign_cluster(clusprob)
    output = []
    validrange = r/2
    for i in range(nn):
        cx,cy = clusdata[ clus_index[i] ]
        x = np.random.uniform(-validrange,validrange) + cx
        y = np.random.uniform(-validrange,validrange) + cy
        output.append([x,y])
    return (np.array(output),clus_index)
def set_init_status(nn = node_num):   #status[node_index] = 0(ready) or 1(moving) or 2(pausing)
    output = []
    for i in range(nn):
        output.append(0)
    return np.array(output)
def set_init_movingtime(nn = node_num):
    output = []
    for i in range(nn):
        output.append(0)
    return np.array(output)
def set_init_pausingtime(nn = node_num):
    output = []
    for i in range(nn):
        output.append(0)
    return np.array(output)
def set_init_velocity(nn = node_num):
    output = []
    for i in range(nn):
        output.append([0,0])
    return np.array(output)
def set_init_des(nn = node_num):
    output = []
    for i in range(nn):
        output.append([0,0])
    return np.array(output)
#def set_init_cluster( clus_index ):
#    nn = len(clus_index)
#    init_cluster = []
#    for i in range(nn):
#        init_cluster.append( {clus_index[i] : 1})
#    return init_cluster
def set_init_visited(data): # visited[node_index][0] = node_index's [x,y,total_visited_times]
    visited = []
    nn = len(data)
    for i in range(nn):
        x, y = data[i]
        visited_data= [x,y,1]
        visited.append( [visited_data] )
    return visited
def explore_or_revisit(visited, node_index):
    prob_ex = prob_explore( len(visited[node_index]) )
    prob_re = 1 - prob_ex
    prob = [prob_re, prob_ex]
    output = [0,1]
    result = np.random.choice( output, 1, replace=True, p=prob)
    if result == 1:
        return True
    else:
        return False
def revisit(visited,node_index):
    total_visited = 0
    prob = []
    total_locations = len(visited[node_index])
    for i in range( total_locations):
        total_visited += visited[node_index][i][2]
    for j in range( total_locations ):
        prob.append( visited[node_index][j][2] / total_visited )
    revisited_index = np.random.choice( [i for i in range(total_locations)], 1, replace=True,p=prob )
    revisited_index = revisited_index[0] # change[ int ] to int because np.random.choice return [int]
    visited[node_index][revisited_index][2] += 1
    x,y,_ = visited[node_index][revisited_index]
    return np.array([x,y])
def explore(data,node_index):
    angle =generate_angle()
    flight = TPL(alpha,alpha_min,alpha_max,np.arange(1))
    flight = flight[0]
    #print(f" during explore flight = {flight} ")
    co = math.cos(angle)
    si = math.sin(angle)
    #print(f" angle = {angle}, sin ={ si} , cos = {co}")
    #print(f" data[0][0] = {data[node_index][0]} , data[0][1] = {data[node_index][1]}")
    x2 = data[node_index][0] + (flight * co )
    #print(f"x2 = {data[node_index][0]} + {flight * co}")
    y2 = data[node_index][1] + (flight * si )
    #print(f"y2 = {data[node_index][1]} + {flight * si}")
    #x2 = x2[0] # x2 is a nparray ,change to int
    #y2 = y2[0] # y2 is a nparray ,change to int
    if x2 > area_width :
        x2 = area_width
    if x2 < 0:
        x2 = 0
    if y2 > area_height:
        y2 = area_height
    if y2 < 0:
        y2 = 0
    #print(f"x2 = {x2} , y2 = {y2}")
    return np.array([x2,y2])
def next_destination(data, visited, node_index):
    result = explore_or_revisit(visited, node_index)
    #print(f"result = {result}")
    if result :
        x,y = explore(data, node_index)
        #print(f" x= {x} , y = {y}")
        visited_before = False
        #print(f"visited_before = {visited_before}")
        #print(f" len of visited[0] = {len(visited[node_index])}")
        for i in range( len(visited[node_index]) ):
            x1, y1, _ = visited[node_index][i]
            #print(f"x1 = {x1},y1 = {y1}")
            if x1 == x and y1 == y :
                visited[node_index][i][2] += 1
                visited_before = True
            break
        if visited_before == False:
            value = [x,y,1]
            visited[node_index].append(value)
        return np.array([x,y])
    else:
        x,y = revisit(visited, node_index)
        return np.array([x,y])
def get_pausingtime():
    pause_time = TPL(beta,beta_min,beta_max,np.arange(1))
    return pause_time[0]  # pause_time = [int]
def compute_velocity_and_time(data, destination, node_index):
    flight = math.dist(data[node_index], destination)
    x2, y2 = destination
    x1, y1 = data[node_index]
    if flight == 0 :
        return (np.array([0,0]), 0 )
    if flight >= 500:
        time = (1.37 * flight ** (1-0.36))
        speed = ( flight / (1.37 * flight ** (1-0.36)))
        velocity = np.array( [speed * ( x2 - x1 )/flight, speed * (y2 - y1)/flight ] )
        return velocity, time
    else:
        time = (18.72 * flight ** (1-0.79))
        speed = ( flight / (18.72 * flight ** (1-0.79)) )
        velocity = np.array( [speed * ( x2 - x1 )/flight, speed * (y2 - y1)/flight ] )
        return velocity, time
class Node():
    def __init__(self, clusdata, clusprob):
        self.data, clus_index = set_node_data(clusdata, clusprob)
        self.visited = set_init_visited(self.data)
        self.status = set_init_status()
        self.des = set_init_des()
        self.velocity = set_init_velocity()
        self.movingtime = set_init_movingtime()
        self.pausingtime = set_init_pausingtime()
       # self.cluster_list = set_init_cluster(clus_index)
    def get_data(self):
        return self.data
    def get_visited(self):
        return self.visited
    def get_status(self):
        return self.status
    def get_des(self):
        return self.des
    def get_velocity(self):
        return self.velocity
    def get_movingtime(self):
        return self.movingtime
    def get_pausingtime(self):
        return self.pausingtime
    def set_status(self, value, node_index):
        self.status[node_index] = value
    def set_velocity(self, value, node_index):
        self.velocity[node_index] = value
    def set_des(self, value, node_index):
        self.des[node_index] = value
    def set_movingtime(self, value, node_index):
        self.movingtime[node_index] = value
    def set_pausingtime(self, value , node_index):
        self.pausingtime[node_index] = value
    def set_data(self, value , node_index):
        x, y = value
        self.data[node_index][0] += x
        self.data[node_index][1] += y
cluster = Cluster()
print(f"cluster = {cluster.getdata()}")
node = Node(cluster.getdata(), cluster.getclus_prob())
print(f"node.data = {node.get_data()}")
print(f"node.visited = {node.get_visited()}")
print(f"node.status = {node.get_status()}")
print(f"visited = {node.get_visited()}")
print(f"velocity = {node.get_velocity()}")
print(f"pausingtime = {node.get_pausingtime()}")
print(f"status = {node.get_status()}")
print( "#########simulation Start######") 
time = 0
while (time < simulation_time):
    for i in range(node_num):
        print(f" time = {time}, node[0] = {node.get_data()[i]},des ={node.get_des()[i]},vi = {node.get_visited()[i]}")
        if (node.get_status())[i] == 0:
            des = next_destination(node.get_data(), node.get_visited(), i)
            #print(f"des = {des}")
            if des[0] != node.get_data()[i][0] or des[1] != node.get_data()[i][1]:
                node.set_des(des,i)
                node.set_status(1,i)
                velocity, movingtime = compute_velocity_and_time(node.get_data(), des, i)
                movingtime = int(movingtime) #無條件進位+1後，更新位置所以後來要在-1等於沒+
                node.set_data(velocity, i) #update data
                if movingtime != 0:
                    node.set_velocity(velocity, i)
                    node.set_movingtime(movingtime,i)
                else:
                    node.set_des([0,0],i)
                    pausingtime = get_pausingtime()
                    pausingtime = int(pausingtime)
                    if pausingtime != 0:
                        node.set_status(2,i)
                        node.set_pausingtime(pause_time,i)
                    else:
                        node.set_status(0,i)
            else:
                pausingtime = get_pausingtime()
                print(f"pausingtime = {pausingtime}")
                pausingtime = int(pausingtime)
                if pausingtime != 0:
                    node.set_status(2,i)
                    node.set_pausingtime(pausingtime,i)
                else:
                    node.set_status(0,i)
        elif (node.get_status())[i] == 1:
            if node.get_movingtime()[i] == 1:
                node.set_data( -node.get_data()[i],i)
                node.set_data(node.get_des()[i],i)
                node.set_velocity( np.array([0,0]),i)
                node.set_movingtime(0,i)
                node.set_des( np.array([0,0]), i)
                pausingtime = get_pausingtime()
                pausingtime = int(pausingtime)
                print(f"pausingtime = {pausingtime}")
                if pausingtime != 0:
                    node.set_status(2,i)
                    node.set_pausingtime(pausingtime,i)
                else:
                    node.set_status(0,i)
            else:
                node.set_data(velocity, i) #update data
                node.set_movingtime( node.get_movingtime()[i]-1, i)
        else:
            if node.get_pausingtime()[i] == 1:
                node.set_pausingtime(0,i)
                node.set_status(0,i)
            else:
                node.set_pausingtime( node.get_pausingtime()[i] -1 , i)
    time += 1
print(f" time = {time}, node[0] = {node.get_data()[i]} ")
print(f" node.visited[0] = {node.get_visited()[0]}")
#cluster_list = set_pos_of_clus()
#print(type(cluster_list))
#print(cluster_list)
#x = []
#y = []
#for i in range( cluster_num):
#    x0 , y0= zip(cluster_list[i])
#    x.append(x0)
#    y.append(y0)
    
#plt.plot(x,y,'ro')
#plt.show()

        


# In[41]:


P = lambda ALPHA, MIN, MAX, SAMPLES: ((MAX ** (ALPHA+1.) -1.) * rand(*SAMPLES.shape) +1.) ** (1./(ALPHA+1.))
import math
area_width = 500.
area_height = 500.
print(P(0.5, 10 ,20 , np.array([2])))
def set_clus_popu(n = cluster_num):
    output = np.random.dirichlet(np.ones(n),size=1)
    return output

def assign_cluster( prob_list, nn = node_num, cn = cluster_num):
    return (np.random.choice( cn, nn,replace=True, p = prob_list))
pl = set_clus_prob()
#print(pl)
a = assign_cluster(prob_list = pl[0])
#print(a)

print("##########")
def set_node_data(clusdata, clusprob,nn = node_num, r = cluster_R):
    clus_index = assign_cluster(clusprob[0])
    output = []
    validrange = r/2
    for i in range(nn):
        cx,cy = clusdata[ clus_index[i] ]
        print(f"cx = {cx}, cy = {cy}")
        x = np.random.uniform(-validrange,validrange) + cx
        y = np.random.uniform(-validrange,validrange) + cy
        print(f"x = {x}, y = {y} ")
        output.append([x,y])
    return (np.array(output),clus_index)
clusdata = set_pos_of_clus()
nl,cluster_list= set_node_data(clusdata=clusdata, clusprob=pl)
nx = []
ny = []
for i in range( node_num):
    x0 , y0= zip(nl[i])
    nx.append(x0)
    ny.append(y0)
x = []
y = []
for i in range( cluster_num):
    x0 , y0= zip(clusdata[i])
    x.append(x0)
    y.append(y0)
#print(f"x={x}, y={y}" )
#print(f"nx={nx}, ny = {ny}")
fig = plt.figure()
ax1 = fig.add_subplot(111)
#ax1.scatter(x,y,c='r',marker='s', label='first')
ax1.scatter(nx,ny,c='b',marker='o', label='second')
plt.legend(loc='upper left')
plt.show()

def set_init_cluster( clus_index ):
    nn = len(clus_index)
    init_cluster = []
    for i in range(nn):
        init_cluster.append( {clus_index[i] : 1})
    return init_cluster
dic = set_init_cluster(cluster_list)
print( cluster_list )
print(dic)
print(nl)
def set_init_visited(data): # visited[node_index][0] = node_index's [x,y,total_visited_times]
    visited = []
    nn = len(data)
    for i in range(nn):
        x, y = data[i]
        visited_data= [x,y,1]
        visited.append( [visited_data] )
    return visited
def explore_or_revisit(visited, node_index):
    prob_ex = prob_explore( len(visited[node_index]) )
    prob_re = 1 - prob_ex
    prob = [prob_re, prob_ex]
    output = [0,1]
    result = np.random.choice( output, 1, replace=True, p=prob)
    if result == 1:
        return True
    else:
        return False
def explore_or_revisit(visited, node_index):
    prob_ex = prob_explore( len(visited[node_index]) )
    prob_re = 1 - prob_ex
    prob = [prob_re, prob_ex]
    output = [0,1]
    result = np.random.choice( output, 1, replace=True, p=prob)
    if result == 1:
        return True
    else:
        return False
def revisit(visited,node_index):
    total_visited = 0
    prob = []
    total_locations = len(visited[node_index])
    for i in range( total_locations):
        total_visited += visited[node_index][i][2]
    for j in range( total_locations ):
        prob.append( visited[node_index][j][2] / total_visited )
    revisited_index = np.random.choice( [i for i in range(total_locations)], 1, replace=True,p=prob )
    print(f"revisited_index = {revisited_index}")
    print(f"revisited_index[0] = {revisited_index[0]}")
    x,y,_ = visited[node_index][revisited_index[0]]
    return np.array([x,y])

visited = set_init_visited(nl)
print(f" visited[0] = {visited[0]}")
r = explore_or_revisit(visited,0)
rx,ry = revisit(visited,0)
print(f"rx = {rx}, ry = {ry} ")
print(f"r = {r}")
print(f"prob_explore or 0= {prob_explore(len(visited[0]) + 100)} ")
vv = np.array([1,1090])
print(rand(*vv.shape))
def generate_angle():
    return np.random.uniform(0,360)
print(f"angle() = {generate_angle()}")
print(f"sin = {math.sin(generate_angle())}, cos = {math.cos(generate_angle())}")
alpha = 1.75
alpha_min = 1.
alpha_max = 500
def explore(data,node_index):
    angle =generate_angle()
    flight = TPL(alpha,alpha_min,alpha_max,np.arange(1))
    x2 = data[node_index][0] + (flight * math.cos(angle) )
    y2 = data[node_index][1] + (flight * math.sin(angle) )
    x2 = x2[0]
    y2 = y2[0]
    print(f" x2 = {x2} , y2 = {y2} " )
    if x2 > area_width :
        x2 = area_width
    if x2 < 0 :
        x2 = 0
    if y2 < 0:
        y2 = 0
    if y2 > area_height:
        y2 = area_height
    return np.array([x2,y2])
print(f" nl = {nl} " )
x0,y0 = explore(nl,0)
print(f"x0 = {x0}, y0 = {y0} ")
def set_init_status(nn = node_num):   #status[node_index] = 0(ready) or 1(moving) or 2(pausing)
    output = []
    for i in range(node_num):
        output.append(0)
    return np.array(output)
print(set_init_status(100))
def get_pause_time():
    pause_time = TPL(beta,beta_min,beta_max,np.arange(1)) 
    return pause_time[0]
print(get_pause_time())
def next_destination(data, visited, node_index):
    result = explore_or_revisit(visited, node_index) 
    if result :
        x,y = explore(data, node_index)
        for i in range( len(visited[node_index]) ):
            x1, y1, _= visited[node_index][i]
            if x1 == x and y1 == y :
                visited[node_index][i][2] += 1
            break
        return np.array([x,y])
    else:
        x,y = revisit(visited, node_index)
        return np.array([x,y])
dest = next_destination(nl, visited, 0)
print(f"dest = {dest}")
def compute_velocity(data, destination, node_index):
    flight = math.dist(data[node_index], destination)
    x2, y2 = destination
    x1, y1 = data[node_index]
    if flight == 0 :
        return np.array([0,0]),0
    if flight >= 500:
        time = (1.37 * (flight ** (1-0.36)) )
        speed = ( flight / (1.37 * (flight ** (1-0.36))  ))
        velocity = np.array([speed * ( x2 - x1 )/flight, speed * (y2 - y1)/flight ]) 
        print(f"speed = {speed}, flight ={flight}, x2 ={x2} ,x1 ={x1}, y2 = {y2}, y1 = {y1} ")
        return velocity, time
    else:
        time = (18.72 * (flight ** (1-0.79)) )
        speed = ( flight / (18.72 * (flight ** (1-0.79)) ) )
        velocity = np.array([speed * ( x2 - x1 )/flight, speed * (y2 - y1)/flight ] )
        print(f"speed = {speed}, flight ={flight}, x2 ={x2}, x1= {x1}, y2 = {y2} y1 = {y1} ")
        return velocity , time
velo_list, time = compute_velocity(nl,dest,0)
time = int(time) + 1
print(f"data[0] = {nl[0]} ")
print(f"velo_list = {velo_list}, time = {time}")
print(f"dest = {dest}, node[0] = {nl[0]} ")
count = 0
#while nl[0][0] < dest[0] or nl[0][1] < dest[1] :
#    nl[0][0] += velo_list[0]
#    nl[0][1] += velo_list[1]
#    count += 1
#print(f"count = {count}")
print(f" nl = {nl}, velo_list = {velo_list}")
for i in range(time):
    nl[0][0] += velo_list[0]
    nl[0][1] += velo_list[1]
print(f"node[0] = {nl[0]}")
print(f"visited = {visited}")
print(f"visited[0] = {visited[0]}")
def add_visited( des, visited, node_index):
    x,y = des
    value = np.array([x,y,1])
    a = np.vstack([visited[node_index],value])
    print(f"visited[0] = {visited[node_index]}")
    print(f"a = {a}")
    visited[node_index] = a
add_visited( dest,visited,0)
print(f"visited[0] = {visited[0]}")

