from __future__ import division
from __future__ import print_function

print("Loading libraries...")
import numpy as np
from numpy import tan, pi
from eig_symm import eig_symm

"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm

from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
"""

FOV_Y = 43 # degrees
FOV_X = 57

SCALE_Y = tan((FOV_Y/2)*pi/180)
SCALE_X = tan((FOV_X/2)*pi/180)

POINT_STRIDE = 4
NORMAL_STRIDE = 8

ARROW_LENGTH = 0.08

RAD_WIN = 4
RAD_NN = 0.05
MIN_NN = int((2*RAD_WIN+1)*0.5)

if __name__ == '__main__':
    size_x = 640
    size_y = 480
    # load RGBD data and convert to float
    rgb_data = np.fromfile('1.rgb', dtype=np.uint8).reshape([size_y, size_x, 3])
    rgb_data = np.float32(rgb_data)/255 # convert to 0-1
    depth_data = np.fromfile('1.depth', dtype=np.uint16).reshape([size_y, size_x])
    depth_data = np.float32(depth_data)/1000
    # build point cloud
    point_cloud = np.zeros((size_y, size_x, 3), dtype=np.float32)
    point_colors = np.zeros((size_y, size_x, 3), dtype=np.float32)
    print("Building point cloud...")
    for r in range(size_y):
        for c in range(size_x):
            if depth_data[r][c] > 0.0:
                u = (c-(size_x-1)/2+1) / (size_x-1)
                v = ((size_y-1)/2-r) / (size_y-1)
                Z = depth_data[r][c]
                point_cloud[r, c, :] = [u*Z*SCALE_X, v*Z*SCALE_Y, Z]
                point_colors[r, c, :] = rgb_data[r][c]
    # calculate point normals
    print("Calculating normals...")
    point_normals = np.zeros((size_y, size_x, 3), dtype=np.float32)
    print_condition = False
    for r in range(0, size_y, NORMAL_STRIDE):
        for c in range(0, size_x, NORMAL_STRIDE):
            #if (not r % 10) and (not c % 10): print(r, c)
            #print_condition = (220 < r < 250) and (200 < c < 220)
            if print_condition: print(r, c)
            window = np.s_[r-RAD_WIN if r-RAD_WIN>=0 else 0:r+RAD_WIN+1 if r+RAD_WIN+1<=size_y else size_y,
                           c-RAD_WIN if c-RAD_WIN>=0 else 0:c+RAD_WIN+1 if c+RAD_WIN+1<=size_x else size_x,
                           :]
            window_points = point_cloud[window].copy()
            window_points[r-window[0].start, c-window[1].start, :] = 0
            window_points = window_points.reshape(window_points.shape[0]*window_points.shape[1], 3)
            window_points = window_points[(window_points>0).any(1)] # initial filtering
            center = point_cloud[r, c, :]
            #print_condition = (5 < len(window_points) < 10)
            #print_condition = 1
            if print_condition: print("center:", center)
            if print_condition: print("window_points:")
            if print_condition: print(window_points)
            if (center>0).any():
                if len(window_points): # if NN > 0
                    # filter by distances from center
                    distances = np.sqrt(np.sum((window_points-center)**2, 1))
                    if (distances<=RAD_NN).any():
                        window_points = window_points[distances<=RAD_NN, :]
                    else:
                        window_points = np.empty(0)
                else:
                    distances = np.empty(0)# for printing
                if print_condition: print("distances:", distances)
                if len(window_points) >= MIN_NN:
                    differences = window_points-center
                    if print_condition: print("differences:", differences)
                    C = differences[:,:,np.newaxis]*differences[:,np.newaxis,:]
                    if print_condition: print("covariances:", C)
                    C_avg = np.mean(C, 0)
                    if print_condition: print("mean covariance:", C_avg)
                    eigvals, eigvecs = np.linalg.eigh(C_avg) # Hermitian assumption valid
                    #eigvals, eigvecs = eig_symm(C_avg) # Hermitian assumption valid
                    if print_condition: print(eigvals, eigvecs)
                    normal = eigvecs[:, np.argmin(eigvals)]
                    if print_condition: print(normal)
                    if center.dot(normal) > 0: # facing away from camera
                        normal = -normal
                    point_normals[r, c, :] = normal
            if print_condition: print("NN:", len(window_points))
    # thin out data for plotting
    point_normal_centers = point_cloud[::NORMAL_STRIDE, ::NORMAL_STRIDE, :]
    point_normals = point_normals[::NORMAL_STRIDE, ::NORMAL_STRIDE, :]
    point_cloud = point_cloud[::POINT_STRIDE, ::POINT_STRIDE, :]
    point_colors = point_colors[::POINT_STRIDE, ::POINT_STRIDE, :]
    # flatten data
    point_normal_centers = np.reshape(point_normal_centers, (point_normal_centers.shape[0]*point_normal_centers.shape[1], 3))
    point_normals = np.reshape(point_normals, (point_normals.shape[0]*point_normals.shape[1], 3))
    point_cloud = np.reshape(point_cloud, (point_cloud.shape[0]*point_cloud.shape[1], 3))
    point_colors = np.reshape(point_colors, (point_colors.shape[0]*point_colors.shape[1], 3))
    # filter bad normal values for plotting
    normal_filter_mask = (point_normals>0).any(1)
    point_normal_centers = point_normal_centers[normal_filter_mask]
    point_normals = point_normals[normal_filter_mask]
    # filter bad depth values for plotting
    point_filter_mask = (point_cloud>0).any(1)
    point_cloud = point_cloud[point_filter_mask]
    point_colors = point_colors[point_filter_mask]

    ### MAYAVI PLOTTING (NO COLORS) ###
    print("Plotting with Mayavi...")
    import mayavi.mlab as mlab
    print(point_cloud)
    print(point_cloud.shape)
    mlab.points3d(point_cloud[:,0], point_cloud[:,1], point_cloud[:,2], scale_factor=0.01, scale_mode='none')
    mlab.quiver3d(point_normal_centers[:,0], point_normal_centers[:,1], point_normal_centers[:,2], point_normals[:,0], point_normals[:,1], point_normals[:,2], scale_factor=ARROW_LENGTH, scale_mode='none', opacity=0.5, line_width=1.0, mode='arrow')
    mlab.show()

    ### MATPLOTLIB PLOTTING (WITH COLORS) ###

    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.scatter(point_cloud[:,0], point_cloud[:,1], point_cloud[:,2], 'o', c=point_colors, s=5, lw=0)

    ax.invert_yaxis()
    ax.elev = -90
    ax.azim = -90

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    #imgplot = plt.imshow(depth_data>>5, cmap=cm.Greys_r)
    #plt.show()

    plt.show()
    """
