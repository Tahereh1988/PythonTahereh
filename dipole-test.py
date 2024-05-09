#!/usr/bin/python3
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from dipole import dipole
import pandas as pd
# magnetic dipole moment in A*m**2
mx = 0
my = 0
mz = 1

# location of the dipole
x0 = 0
y0 = 0
z0 = 0

# initialize the dipole with this magnetic moment and location
d = dipole(x0, y0, z0, mx, my, mz)

# what's the field produced by this dipole at some location?
x = 1
y = 0
z = 0

print(d.bx(x, y, z), d.by(x, y, z), d.bz(x, y, z), d.b(x, y, z))

d.random(1.2)

xd = []
yd = []
zd = []
mx = []
my = []
mz = []
for i in range(100):
    d.random(1.2)
    xd.append(d.xd)
    yd.append(d.yd)
    zd.append(d.zd)
    mx.append(d.mx)
    my.append(d.my)
    mz.append(d.mz)
Dipole = {'xd' : xd, 'yd': yd, 'zd':zd, 'mx':mx, 'my': my, 'mz':mz}
columns = ('xd', 'yd', 'zd', 'mx', 'my', 'mz')
df = pd.DataFrame(data = Dipole)
df = df.to_csv('RandomDipole.csv')

df = pd.read_csv('RandomDipole.csv')
type(df)
pd.set_option("display.max.columns", None)
df.head()
# df.plot(x='xd', y=['mx','my','mz'], kind = "hist")

fig = plt.figure()
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax.scatter(xd, yd, zd, marker='.')
ax2.scatter(mx, my, mz, marker='.')
plt.show()

d.set(1.2, 0, 0, mx[-1], my[-1], mz[-1])
print('Final d')
print(d.xd, d.yd, d.zd, d.bx(x, y, z), d.by(x, y, z), d.bz(x, y, z))

d.set(-1.2, 0, 0, mx[-1], my[-1], mz[-1])
print('Final d')
print(d.xd, d.yd, d.zd, d.bx(x, y, z), d.by(x, y, z), d.bz(x, y, z))

