"""
This code considers: stochastic magnetic field and reflection at vascular boundaries
"""
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from particle_motion_magnetic import particle_motion
from vessel_net import vessel_net_3d as vessel_net

T, dt, record_dt = 200, 0.1, 0.2

# load vascular network data
net, ma1, ma2, ma3 = vessel_net('./vessel_net_3d.txt')

# calculate particle trajectory
t = particle_motion(network=net, boundary=[ma1,ma2,ma3], T=T, dt=dt)
t.run(strategy='levy', para1=0.1, para2=2.0)
traj = np.zeros((int(T/record_dt), 3), float)
for ti in range(int(T / record_dt)):
    traj[ti, :] = t.pos[int(ti/dt*record_dt),0], t.pos[int(ti/dt*record_dt),1], t.pos[int(ti/dt*record_dt),2]

# view results
st1, st2 = np.where(net==255), np.where(net==100)
fig = plt.figure()
ax = plt.gca(projection='3d')
ax.set_xlim(0, ma1)
ax.set_ylim(0, 250)
ax.set_zlim(0, ma3)
ax.scatter(st2[0], st2[1], st2[2], s=0.05, c='brown', marker='.', alpha=0.15)
ax.plot(traj[:,0], traj[:,1], traj[:,2], c='black')
ax.view_init(elev=60, azim=-60)
plt.show()


