import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
import math

def make_hex(n1, n2):
    '''
    make a hexagonal lattice of particles, n1 and n2 decides how big it is
    '''

    pointx = np.concatenate((np.arange(n1), np.arange(-0.5, n1 + 0.5, 1)), axis=0)
    pointx = np.tile(pointx, n2)
    pointx = np.concatenate((pointx, np.arange(n1)), axis=0)

    pointy = np.zeros(n1)

    for i in range(1, n2 * 2 + 1):
        if i % 2 == 1:
            pointy = np.concatenate((pointy, np.ones(n1 + 1) * np.sqrt(1 - 0.5 ** 2) * i), axis=0)
        else:
            pointy = np.concatenate((pointy, np.ones(n1) * np.sqrt(1 - 0.5 ** 2) * i), axis=0)

    i_points = np.array([pointx - np.mean(pointx), pointy[:len(pointx)]]).T
    return i_points


def make_movie(dat_hd_r, movname=[]):
    '''make movies, need to install ffmpeg'''
    import subprocess

    imgname = dat_hd_r
    paths = dat_hd_r.split('/')
    datedir = ''
    for ii in range(len(paths) - 2):
        pn = paths[ii]
        datedir += pn + '/'
    if movname == []:
        movname = datedir + dat_hd_r.split('/')[-2] + '_xy0_1'

    print(imgname)
    print(movname)
    subprocess.call(['/Users/jiayiwu/simujupyter/lepm/ffmpeg', '-i',
                     imgname + '%*.png', '-filter:v', 'setpts=1*PTS',
                     '-vb', '20M', movname + '.mov', '-vcodec', 'libx264',
                     '-profile:v',
                     'main',
                     '-crf',
                     '1',
                     '-threads',
                     '0', '-r',
                     '100', '-pix_fmt', 'yuv420p'])


def plot_circles(fig, ax, x, y, r, c='r', cmap=matplotlib.cm.rainbow, ticks=[0, 1]):
    patches = []
    for x1, y1 in zip(x, y):
        circle = plt.Circle((x1, y1), r)
        patches.append(circle)

    if isinstance(c, str):
        p = PatchCollection(patches, facecolor="None", edgecolor=c, alpha=1, linewidths=2)
        ax.add_collection(p)
        return ax

    else:
        p = PatchCollection(patches, cmap=cmap, alpha=0.8)
        p.set_array(np.array(c))

    ax.add_collection(p)

    # cbar = fig.colorbar(p, ax=ax,ticks=ticks)
    p.set_clim(min(ticks), max(ticks))
    return fig


def get_d_txt(a):
    times_go_thr = []
    forces = []
    c_gs = []
    j = 0
    for i in a:
        if j != 0:
            print(i.split(",")[0])
            times_go_thr += [float(i.split(",")[0])]
            forces += [float(i.split(",")[1])]
            c_gs += [float(i.split(",")[2])]
        j += 1
    return forces, times_go_thr

def PointsInCircum(a,b,n=100):
    return np.array([[math.cos(2*np.pi/n*x)*a,math.sin(2*np.pi/n*x)*b] for x in range(0,n+1)])
