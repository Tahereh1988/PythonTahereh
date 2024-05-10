#!/usr/bin/env python3

# Fri May 24 10:11:45 CDT 2019 Jeff added this line.

# Tue Feb 11 13:43:43 CST 2020 Jeff taking original patch.py and
# updating to solve the zero mode issue.  Will now update to use the
# patchlib submodule.

# Fri Feb 14 11:45:17 CST 2020 Jeff speeding up code by improving the
# sensorarray class to use numpy structures.

# Sat Aug 1 12:40:53 CDT 2020 Jeff added command line options and
# improved graphs

from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
# import random
from scipy.constants import mu_0, pi
import numpy as np
from patchlib.patch import *
from Pis.Pislib import *
from dipole import *
import pandas as pd
from optparse import OptionParser

parser = OptionParser()

parser.add_option("-l", "--ell", dest="l", default=2,
                  help="l for spherical harmonic")

parser.add_option("-m", "--em", dest="m", default=0,
                  help="m for spherical harmonic")

parser.add_option("-M", "--matrices", dest="matrices", default=False,
                  action="store_true",
                  help="show matrices")

parser.add_option("-d", "--dipole", dest="dipole", default=False,
                  action="store_true",
                  help="use dipole field")

parser.add_option("-t", "--traces", dest="traces", default=False,
                  action="store_true",
                  help="show 3D view of coils and sensors")

parser.add_option("-r", "--residuals", dest="residuals", default=False,
                  action="store_true",
                  help="show residuals")

parser.add_option("-z", "--zoom", dest="zoom", default=False,
                  action="store_true",
                  help="zoom to ROI")

parser.add_option("-a", "--axes", dest="axes", default=False,
                  action="store_true",
                  help="make graphs along axes")

parser.add_option("-i", "--incells", dest="incells", default=False,
                  action="store_true",
                  help="ROI for statistics is in EDM cells")

d=dipole()

# d = dipole(1.2, 0, 0, 0, 0, 100000)
# d.set(1.2,0,0,0,0,100000)  # dipole1
#d2 = dipole(1.2, 0, 0, 0, 0, 100)
#d2.random(1.2)
# d=dipole(0,0,1.2,0,0,1)  # dipole2
# d=dipole(0,0,1.2,1,0,0)  # dipole3

(options, args) = parser.parse_args()

l = int(options.l)
m = int(options.m)
sp = scalarpotential(l, m)
print("Sigma in spherical coordinates is %s" % sp.Sigma_spherical)
print("Sigma in cartesian coordinates is %s" % sp.Sigma)

print("Pix is %s" % sp.Pix)
print("Piy is %s" % sp.Piy)
print("Piz is %s" % sp.Piz)

#if(options.dipole):
bxtarget = d.bx
bytarget = d.by
bztarget = d.bz
#else:
#    bxtarget = sp.fPix
#    bytarget = sp.fPiy
#    bztarget = sp.fPiz

# Setup our coilset

myset = coilset()

# positions of faces (positive values -- faces will be at plus and
# minus of these)
a = 1.95
xface = a/2  # m
yface = a/2  # m
zface = a/2  # m

# set up the rear wall
y1 = (170/2 + 60)*.001
z1 = (400/2 + 60)*.001
x1 = xface
point1 = (x1, y1, z1)
y2 = y1
z2 = z1 + 400 * 0.001  # guess
x2 = xface
point2 = (x2, y2, z2)
y3 = y1 + 500 * 0.001  # guess
z3 = z2
x3 = xface
point3 = (x3, y3, z3)
y4 = y3
z4 = z1
x4 = xface
point4 = (x4, y4, z4)
points_ur = (point1, point2, point3, point4)
points_ur = np.array(points_ur)
myset.add_coil(points_ur)

# Now add mirror images of these
point1 = (x1, -y1, z1)
point2 = (x4, -y4, z4)
point3 = (x3, -y3, z3)
point4 = (x2, -y2, z2)
points_ul = (point1, point2, point3, point4)
points_ul = np.array(points_ul)
myset.add_coil(points_ul)

point1 = (x1, -y1, -z1)
point2 = (x2, -y2, -z2)
point3 = (x3, -y3, -z3)
point4 = (x4, -y4, -z4)
points_ll = (point1, point2, point3, point4)
points_ll = np.array(points_ll)
myset.add_coil(points_ll)

point1 = (x1, y1, -z1)
point2 = (x4, y4, -z4)
point3 = (x3, y3, -z3)
point4 = (x2, y2, -z2)
points_lr = (point1, point2, point3, point4)
points_lr = np.array(points_lr)
myset.add_coil(points_lr)

# now the central coil
x1 = xface
y1 = 530/2*.001
z1 = 400/2*.001
point1 = (x1, y1, z1)
point2 = (x1, y1, -z1)
point3 = (x1, -y1, -z1)
point4 = (x1, -y1, z1)
points_c = (point1, point2, point3, point4)
points_c = np.array(points_c)
myset.add_coil(points_c)

# now the right side coil
x1 = xface
y1 = 1190/2*.001
z1 = 400/2*.001
point1 = (x1, y1, z1)
x2 = xface
y2 = y1+420*.001  # guess
z2 = z1
point2 = (x2, y2, z2)
x3 = xface
y3 = y2
z3 = -z2
point3 = (x3, y3, z3)
x4 = xface
y4 = y1
z4 = -z1
point4 = (x4, y4, z4)
points_mr = (point1, point2, point3, point4)
points_mr = np.array(points_mr)
myset.add_coil(points_mr)

# now the left side coil -- reflect and wind in same direction
point1 = (x1, -y1, z1)
point2 = (x4, -y4, z4)
point3 = (x3, -y3, z3)
point4 = (x2, -y2, z2)
points_ml = (point1, point2, point3, point4)
points_ml = np.array(points_ml)
myset.add_coil(points_ml)

# now reflect them all to the other face: xface -> -xface


def reflect_x(points):
    newpoints = np.copy(points)
    newpoints[:, 0] = -newpoints[:, 0]
    newpoints = np.flip(newpoints, 0)  # wind them in the opposite direction
    return newpoints


oside_ur = reflect_x(points_ur)
myset.add_coil(oside_ur)
oside_ul = reflect_x(points_ul)
myset.add_coil(oside_ul)
oside_ll = reflect_x(points_ll)
myset.add_coil(oside_ll)
oside_lr = reflect_x(points_lr)
myset.add_coil(oside_lr)
oside_c = reflect_x(points_c)
myset.add_coil(oside_c)
oside_ml = reflect_x(points_ml)
myset.add_coil(oside_ml)
oside_mr = reflect_x(points_mr)
myset.add_coil(oside_mr)

# Phew -- now onto the sides

z1 = (400/2+60)*.001
x1 = (105/2+60+255+60)*.001
y1 = -yface
point1 = (x1, y1, z1)
z2 = z1+420*.001  # guess
x2 = x1
y2 = -yface
point2 = (x2, y2, z2)
z3 = z2
x3 = x2+500*.001  # guess
y3 = -yface
point3 = (x3, y3, z3)
z4 = z1
x4 = x3
y4 = -yface
point4 = (x4, y4, z4)
side_ur = (point1, point2, point3, point4)
side_ur = np.array(side_ur)
myset.add_coil(side_ur)

# now reflect around
point1 = (-x1, y1, z1)
point2 = (-x4, y4, z4)
point3 = (-x3, y3, z3)
point4 = (-x2, y2, z2)
side_ul = np.array((point1, point2, point3, point4))
myset.add_coil(side_ul)

point1 = (-x1, y1, -z1)
point2 = (-x2, y2, -z2)
point3 = (-x3, y3, -z3)
point4 = (-x4, y4, -z4)
side_ll = np.array((point1, point2, point3, point4))
myset.add_coil(side_ll)

point1 = (x1, y1, -z1)
point2 = (x4, y4, -z4)
point3 = (x3, y3, -z3)
point4 = (x2, y2, -z2)
side_lr = np.array((point1, point2, point3, point4))
myset.add_coil(side_lr)

# central coil
z1 = 400/2*.001
y1 = -yface
x1 = (105/2+60+255)*.001
point1 = (x1, y1, z1)
point2 = (x1, y1, -z1)
point3 = (-x1, y1, -z1)
point4 = (-x1, y1, z1)
side_c = np.array((point1, point2, point3, point4))
myset.add_coil(side_c)

# middle right coil
x1 = (105/2+60+255+60)*.001
y1 = -yface
z1 = 400/2*.001
point1 = (x1, y1, z1)
x2 = x1+500*.001  # same guess as above
y2 = -yface
z2 = z1
point2 = (x2, y2, z2)
point3 = (x2, y2, -z2)
point4 = (x1, y1, -z1)
side_mr = np.array((point1, point2, point3, point4))
myset.add_coil(side_mr)

# reflect it to middle left coil
point1 = (-x1, y1, z1)
point2 = (-x1, y1, -z1)
point3 = (-x2, y2, -z2)
point4 = (-x2, y2, z2)
side_ml = np.array((point1, point2, point3, point4))
myset.add_coil(side_ml)

# middle top
z1 = (400/2+60)*.001
x1 = (105/2+60+255)*.001
y1 = -yface
point1 = (x1, y1, z1)
z2 = z1
x2 = -x1
y2 = -yface
point2 = (x2, y2, z2)
z3 = z2+420*.001  # same guess as above
x3 = x2
y3 = -yface
point3 = (x3, y3, z3)
z4 = z3
x4 = x1
y4 = -yface
point4 = (x4, y4, z4)
side_mt = np.array((point1, point2, point3, point4))
myset.add_coil(side_mt)

# mirror to middle bottom
point1 = (x1, y1, -z1)
point2 = (x4, y4, -z4)
point3 = (x3, y3, -z3)
point4 = (x2, y2, -z2)
side_mb = np.array((point1, point2, point3, point4))
myset.add_coil(side_mb)

# now reflect them all to the other face: -yface -> yface


def reflect_y(points):
    newpoints = np.copy(points)
    newpoints[:, 1] = -newpoints[:, 1]
    newpoints = np.flip(newpoints, 0)  # wind them in the opposite direction
    return newpoints


oside_side_ur = reflect_y(side_ur)
oside_side_ul = reflect_y(side_ul)
oside_side_ll = reflect_y(side_ll)
oside_side_lr = reflect_y(side_lr)
oside_side_c = reflect_y(side_c)
oside_side_ml = reflect_y(side_ml)
oside_side_mr = reflect_y(side_mr)
oside_side_mt = reflect_y(side_mt)
oside_side_mb = reflect_y(side_mb)

myset.add_coil(oside_side_ur)
myset.add_coil(oside_side_ul)
myset.add_coil(oside_side_lr)
myset.add_coil(oside_side_ll)
myset.add_coil(oside_side_c)
myset.add_coil(oside_side_ml)
myset.add_coil(oside_side_mr)
myset.add_coil(oside_side_mt)
myset.add_coil(oside_side_mb)


# Double phew, now on to the top side

x1 = (400/2+60)*.001  # picture frame of 400x400's separated by 60's
y1 = (400/2+60)*.001
z1 = zface
point1 = (x1, y1, z1)
x2 = x1
y2 = y1+400*.001
z2 = zface
point2 = (x2, y2, z2)
x3 = x2+400*.001
y3 = y2
z3 = zface
point3 = (x3, y3, z3)
x4 = x3
y4 = y1
z4 = zface
point4 = (x4, y4, z4)
top_ur = (point1, point2, point3, point4)
top_ur = np.array(top_ur)
myset.add_coil(top_ur)

# now reflect around
point1 = (-x1, y1, z1)
point2 = (-x4, y4, z4)
point3 = (-x3, y3, z3)
point4 = (-x2, y2, z2)
top_ul = np.array((point1, point2, point3, point4))
myset.add_coil(top_ul)

point1 = (-x1, -y1, z1)
point2 = (-x2, -y2, z2)
point3 = (-x3, -y3, z3)
point4 = (-x4, -y4, z4)
top_ll = np.array((point1, point2, point3, point4))
myset.add_coil(top_ll)

point1 = (x1, -y1, z1)
point2 = (x4, -y4, z4)
point3 = (x3, -y3, z3)
point4 = (x2, -y2, z2)
top_lr = np.array((point1, point2, point3, point4))
myset.add_coil(top_lr)

# central coil
z1 = zface
y1 = 400/2*.001
x1 = 400/2*.001
point1 = (x1, y1, z1)
point2 = (x1, -y1, z1)
point3 = (-x1, -y1, z1)
point4 = (-x1, y1, z1)
top_c = np.array((point1, point2, point3, point4))
myset.add_coil(top_c)

# middle right coil
x1 = (400/2+60)*.001
y1 = 400/2*.001
z1 = zface
point1 = (x1, y1, z1)
x2 = x1+400*.001
y2 = y1
z2 = zface
point2 = (x2, y2, z2)
point3 = (x2, -y2, z2)
point4 = (x1, -y1, z1)
top_mr = np.array((point1, point2, point3, point4))
myset.add_coil(top_mr)

# reflect it to middle left coil
point1 = (-x1, y1, z1)
point2 = (-x1, -y1, z1)
point3 = (-x2, -y2, z2)
point4 = (-x2, y2, z2)
top_ml = np.array((point1, point2, point3, point4))
myset.add_coil(top_ml)

# middle top
x1 = 400/2*.001
y1 = (400/2+60)*.001
z1 = zface
point1 = (x1, y1, z1)
x2 = -x1
y2 = y1
z2 = zface
point2 = (x2, y2, z2)
x3 = x2
y3 = y2+400*.001
z3 = zface
point3 = (x3, y3, z3)
x4 = x1
y4 = y3
z4 = zface
point4 = (x4, y4, z4)
top_mt = np.array((point1, point2, point3, point4))
myset.add_coil(top_mt)

# mirror to middle bottom
point1 = (x1, -y1, z1)
point2 = (x4, -y4, z4)
point3 = (x3, -y3, z3)
point4 = (x2, -y2, z2)
top_mb = np.array((point1, point2, point3, point4))
myset.add_coil(top_mb)

# now reflect them all to the other face: zface -> -zface


def reflect_z(points):
    newpoints = np.copy(points)
    newpoints[:, 2] = -newpoints[:, 2]
    newpoints = np.flip(newpoints, 0)  # wind them in the opposite direction
    return newpoints


bott_ur = reflect_z(top_ur)
bott_ul = reflect_z(top_ul)
bott_ll = reflect_z(top_ll)
bott_lr = reflect_z(top_lr)
bott_c = reflect_z(top_c)
bott_ml = reflect_z(top_ml)
bott_mr = reflect_z(top_mr)
bott_mt = reflect_z(top_mt)
bott_mb = reflect_z(top_mb)

myset.add_coil(bott_ur)
myset.add_coil(bott_ul)
myset.add_coil(bott_lr)
myset.add_coil(bott_ll)
myset.add_coil(bott_c)
myset.add_coil(bott_ml)
myset.add_coil(bott_mr)
myset.add_coil(bott_mt)
myset.add_coil(bott_mb)


class sensor:
    def __init__(self, pos):
        self.pos = pos

class sensorarray:
    def __init__(self):
        self.sensors = []
        self.numsensors = len(self.sensors)

    def add_sensor(self, pos):
        self.sensors.append(sensor(pos))
        self.numsensors = len(self.sensors)

    def draw_sensor(self, number, ax):
        x = self.sensors[number].pos[0]
        y = self.sensors[number].pos[1]
        z = self.sensors[number].pos[2]
        c = 'r'
        m = 'o'
        ax.scatter(x, y, z, c=c, marker=m)

    def draw_sensors(self, ax):
        for number in range(self.numsensors):
            self.draw_sensor(number, ax)

    def vec_b(self, bxtarget, bytarget, bztarget):
        # makes a vector of magnetic fields in the same ordering as
        # the_matrix class below
        the_vector = np.zeros((self.numsensors*3))
        for j in range(myarray.numsensors):
            r = myarray.sensors[j].pos
            b = np.array([bxtarget(r[0], r[1], r[2]),
                          bytarget(r[0], r[1], r[2]),
                          bztarget(r[0], r[1], r[2])])
            for k in range(3):
                the_vector[j*3+k] = b[k]
        return the_vector


# set up array of sensors
a_sensors = 0.5
myarray = sensorarray()
# random positions for sensors
# in each run we have different positions
# f = open("SP.text", "w+")
# for i in range(24):
#     x = (round(random.uniform(-0.5, 0.5), 2))
#     y = (round(random.uniform(-0.5, 0.5), 2))
#     z = (round(random.uniform(-0.5, 0.5), 2))
#     f.write('%f %f %f\n' % (x, y, z))
# f.close()

with open('SP.text') as f:
    for line in f:
        pos = np.array([float(u) for u in line.split()])
        print(pos)
        myarray.add_sensor(pos)


mpl.rcParams['legend.fontsize'] = 10


class the_matrix:
    def __init__(self, myset, myarray):
        self.m = np.zeros((myset.numcoils, myarray.numsensors*3))
        # self.fill(myset,myarray)
        self.fillspeed(myset, myarray)
        self.condition = np.linalg.cond(self.m)

        # for some reason I chose to create the transpose of the usual
        # convention, when I first wrote the fill method
        self.capital_M = self.m.T  # M=s*c=sensors*coils Matrix

        # Do the svd
        self.U, self.s, self.VT = np.linalg.svd(self.capital_M)

        print('s is', self.s)
        # s is just a list of the diagonal elements, rather than a matrix
        # You can make the matrix this way:
        self.S = np.zeros(self.capital_M.shape)
        self.S[:self.capital_M.shape[1],
               :self.capital_M.shape[1]] = np.diag(self.s)
        # Or, I've seen people use "full_matrices=True" in the svd command

        # Start to calculate the inverse explicitly
        # list of reciprocals
        d = 1./self.s
        self.D = np.zeros(self.capital_M.shape)
        # matrix of reciprocals
        self.D[:self.capital_M.shape[1], :self.capital_M.shape[1]] = np.diag(d)

        # inverse of capital_M
        self.Minv = self.VT.T.dot(self.D.T).dot(self.U.T)
        # self.Minv=np.linalg.pinv(self.capital_M)

        # now gets to fixin'
        # remove just the last mode
        n_elements = myset.numcoils-1

        self.Dp = self.D[:, :n_elements]
        self.VTp = self.VT[:n_elements, :]
        self.Minvp = self.VTp.T.dot(self.Dp.T).dot(self.U.T)

    def fill(self, myset, myarray):
        for i in range(myset.numcoils):
            myset.set_independent_current(i, 1.0)
            for j in range(myarray.numsensors):
                r = myarray.sensors[j].pos
                b = myset.b(r)
                for k in range(3):
                    self.m[i, j*3+k] = b[k]
            myset.set_independent_current(i, 0.0)

    def fillspeed(self, myset, myarray):
        myset.set_common_current(1.0)
        for i in range(myset.numcoils):
            print("Doing coil %d" % i)
            for j in range(myarray.numsensors):
                r = myarray.sensors[j].pos
                bx, by, bz = myset.coil[i].b_prime(r[0], r[1], r[2])
                b = [bx, by, bz]
                for k in range(3):
                    self.m[i, j*3+k] = b[k]
        myset.zero_currents()

    def check_field_graphically(self, myset, myarray):
        # test each coil by graphing field at each sensor
        for i in range(myset.numcoils):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            myset.draw_coil(i, ax)
            myset.coil[i].set_current(1.0)
            for j in range(myarray.numsensors):
                r = myarray.sensors[j].pos
                b = myset.b(r)
                bhat = b*5.e4
                points = []
                points.append(r)
                points.append(r+bhat)
                xs = ([p[0] for p in points])
                ys = ([p[1] for p in points])
                zs = ([p[2] for p in points])
                ax.plot(xs, ys, zs)
            myset.coil[i].set_current(0.0)
            ax.legend()
            plt.show()

    def show_matrices(self):
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()
        fig3, ax3 = plt.subplots()
        fig4, ax4 = plt.subplots()
        fig5, ax5 = plt.subplots()
        fig6, ax6 = plt.subplots()
        fig7, ax7 = plt.subplots()
        fig8, ax8 = plt.subplots()
        fig9, ax9 = plt.subplots()

        ax1.imshow(self.capital_M, cmap=cm.bwr)
        ax2.imshow(self.U, cmap=cm.bwr)
        ax3.imshow(self.S, cmap=cm.bwr)
        ax4.imshow(self.VT, cmap=cm.bwr)
        ax5.imshow(self.D, cmap=cm.bwr)
        ax6.imshow(self.Minv, cmap=cm.bwr)
        ax7.imshow(self.Dp, cmap=cm.bwr)
        ax8.imshow(self.VTp, cmap=cm.bwr)
        ax9.imshow(self.Minvp, cmap=cm.bwr)

        ax1.set_title('M')
        ax2.set_title('U')
        ax3.set_title('S')
        ax4.set_title('VT')
        ax5.set_title('D')
        ax6.set_title('Minv')
        ax7.set_title('Dp')
        ax8.set_title('VTp')
        ax9.set_title('Minvp')

        plt.show()


mymatrix = the_matrix(myset, myarray)

print('The condition number is %f' % mymatrix.condition)
if(options.matrices):
    mymatrix.show_matrices()

# cell geometry setup
rcell = 0.3  # m, cell radius
hcell = 0.1602  # m, cell height
dcell = 0.0799  # m, bottom to top distance of cells

# studies over an ROI
# x,y,z=np.mgrid[-.5:.5:101j,-.5:.5:101j,-.5:.5:101j]
x, y, z = np.mgrid[-.5:.5:21j, -.5:.5:21j, -.5:.5:21j]

mask = (abs(z) >= dcell/2) & (abs(z) <= dcell/2+hcell) & (x**2+y**2 < rcell**2)
mask_upper = (abs(z) >= dcell/2) & (abs(z) <= dcell/2 +
                                    hcell) & (x**2+y**2 < rcell**2) & (z > 0)
mask_lower = (abs(z) >= dcell/2) & (abs(z) <= dcell/2 +
                                    hcell) & (x**2+y**2 < rcell**2) & (z < 0)


def find_delta_sigma():
    # Set up vector of desired fields
    vec_i = mymatrix.Minvp.dot(myarray.vec_b(bxtarget, bytarget, bztarget))

    # Assign currents to coilcube
    myset.set_currents(vec_i)

    bx_roi,by_roi,bz_roi=myset.b_prime(x,y,z)
    bx_target = bxtarget(x, y, z)
    by_target = bytarget(x, y, z)
    bz_target = bztarget(x, y, z)
    bx_residual = bx_roi-bx_target
    by_residual = by_roi-by_target
    bz_residual = bz_roi-bz_target

    print('Statistics on the ROI')
    # print

    bz_ave = np.average(bz_target)
    #print('The unmasked average Bz prior to correction is %e'%bz_ave)
    bz_max = np.amax(bz_target)
    bz_min = np.amin(bz_target)
    bz_delta = bz_max-bz_min
    #print('The unmasked max/min/diff Bz are %e %e %e'%(bz_max,bz_min,bz_delta))
    #print('We normalize this to 3 nT max-min')
    # print

    print('Upper cell')
    bz_mask_max = np.amax(bz_target[mask_upper])
    bz_mask_min = np.amin(bz_target[mask_upper])
    bz_mask_delta = bz_mask_max-bz_mask_min
    print('The max/min/diff Bz masks are %e %e %e' %
          (bz_mask_max, bz_mask_min, bz_mask_delta))
    print('Normalizing to 3 nT gives a delta of %f nT' %
          (bz_mask_delta/bz_delta*3))
    bz_std = np.std(bz_target[mask_upper])
    print('The masked standard deviation of Bz is %e' % bz_std)
    print('Normalizing to 3 nT gives a standard deviation of %f nT' %
          (bz_std/bz_delta*3))
    before_correction_delta_upper=bz_mask_delta/bz_delta*3
    before_correction_sigma_upper=bz_std/bz_delta*3
    print

    bz_residual_max = np.amax(bz_residual[mask_upper])
    bz_residual_min = np.amin(bz_residual[mask_upper])
    bz_residual_delta = bz_residual_max-bz_residual_min
    print('The max/min/diff Bz residuals are %e %e %e' %
          (bz_residual_max, bz_residual_min, bz_residual_delta))
    print('Normalizing to 3 nT gives a delta of %f nT' %
          (bz_residual_delta/bz_delta*3))
    bz_residual_std = np.std(bz_residual[mask_upper])
    print('The standard deviation of Bz residuals is %e' % bz_residual_std)
    print('Normalizing to 3 nT gives a standard deviation of %f nT' %
          (bz_residual_std/bz_delta*3))
    after_correction_delta_upper=bz_residual_delta/bz_delta*3
    after_correction_sigma_upper=bz_residual_std/bz_delta*3
    unnormalized_sigma_upper = bz_residual_std
    unnormalized_delta_upper = bz_residual_delta
    print

    print('Lower cell')
    bz_mask_max = np.amax(bz_target[mask_lower])
    bz_mask_min = np.amin(bz_target[mask_lower])
    bz_mask_delta = bz_mask_max-bz_mask_min
    print('The max/min/diff Bz masks are %e %e %e' %
          (bz_mask_max, bz_mask_min, bz_mask_delta))
    print('Normalizing to 3 nT gives a delta of %f nT' %
          (bz_mask_delta/bz_delta*3))
    bz_std = np.std(bz_target[mask_lower])
    print('The masked standard deviation of Bz is %e' % bz_std)
    print('Normalizing to 3 nT gives a standard deviation of %f nT' %
          (bz_std/bz_delta*3))
    before_correction_delta_lower=bz_mask_delta/bz_delta*3
    before_correction_sigma_lower=bz_std/bz_delta*3
    print

    bz_residual_max = np.amax(bz_residual[mask_lower])
    bz_residual_min = np.amin(bz_residual[mask_lower])
    bz_residual_delta = bz_residual_max-bz_residual_min
    print('The max/min/diff Bz residuals are %e %e %e' %
          (bz_residual_max, bz_residual_min, bz_residual_delta))
    print('Normalizing to 3 nT gives a delta of %f nT' %
          (bz_residual_delta/bz_delta*3))
    bz_residual_std = np.std(bz_residual[mask_lower])
    print('The standard deviation of Bz residuals is %e' % bz_residual_std)
    print('Normalizing to 3 nT gives a standard deviation of %f nT' %
          (bz_residual_std/bz_delta*3))
    after_correction_delta_lower=bz_residual_delta/bz_delta*3
    after_correction_sigma_lower=bz_residual_std/bz_delta*3
    unnormalized_sigma_lower = bz_residual_std
    unnormalized_delta_lower = bz_residual_delta
    print

    bt2_target = bx_target**2+by_target**2+bz_target**2
    bt2_ave = np.average(bt2_target[mask])
    print('The BT2 prior to correction is %e' % bt2_ave)
    bt2_ave_norm = bt2_ave*3**2/bz_delta**2
    print('Normalized is %f nT^2' % (bt2_ave_norm))

    bt2_residual = bx_residual**2+by_residual**2+bz_residual**2
    bt2_residual_ave = np.average(bt2_residual[mask])
    print('The BT2 after correction is %e' % bt2_residual_ave)
    bt2_residual_ave_norm = bt2_residual_ave*3**2/bz_delta**2
    print('Normalized is %f nT^2' % (bt2_residual_ave_norm))
    print

    # Tara's nice printing

    print()

    print('***Tara\'s nice printing***')
    print()
    print('Delta(B_z) upper before %f after %f' % (before_correction_delta_upper,after_correction_delta_upper))
    print('Delta(B_z) lower before %f after %f' % (before_correction_delta_lower,after_correction_delta_lower))
    print('Sigma(B_z) upper before %f after %f' % (before_correction_sigma_upper,after_correction_sigma_upper))
    print('Sigma(B_z) lower before %f after %f' % (before_correction_sigma_lower,after_correction_sigma_lower))
    print('BT2 before correction %f nT^2' % (bt2_ave_norm))
    print('BT2 after correction %f nT^2' % (bt2_residual_ave_norm))
    
    print()
    print('***end of Tara\'s nice printing***')

    print()
    # end Tara's nice printing

    print('The normalized currents are:')
    vec_i = vec_i*3e-9/bz_delta
    print(vec_i)
    print('The maximum current is %f A' % np.amax(vec_i))
    print('The minimum current is %f A' % np.amin(vec_i))

    return unnormalized_sigma_upper,unnormalized_delta_upper,unnormalized_sigma_lower,unnormalized_delta_lower, before_correction_delta_upper,after_correction_delta_upper,before_correction_delta_lower,after_correction_delta_lower,before_correction_sigma_upper,after_correction_sigma_upper,before_correction_sigma_lower,after_correction_sigma_lower, bt2_ave_norm,bt2_residual_ave_norm


df=pd.DataFrame(columns=['xd','yd','zd','mx','my','mz','sbzu','sbzl','dbzu','dbzl','unsbzu', 'unsbzl','undbzl', 'undbzu'])



for trial in range(3):
    d.random(1.2)
    bc_dbzu,dbzu,bc_dbzl,dbzl,bc_sbzu,sbzu,bc_sbzl,sbzl,bt2_ave_norm,bt2_residual_ave_norm,unsbzu,unsbzl,undbzl,undbzu =find_delta_sigma()
    df.loc[len(df)]=[d.xd,d.yd,d.zd,d.mx,d.my,d.mz,sbzu,sbzl,dbzu,dbzl,unsbzu,unsbzl,undbzl,undbzu]

df = df.to_csv('RandomDipole.csv')
