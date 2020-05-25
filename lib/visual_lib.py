import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
from matplotlib.collections import PatchCollection


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

    p.set_clim(min(ticks), max(ticks))
    return fig, p


def plot_active_force(c_active, alp, c_g):
    plt.plot(np.arange(0, 10, 0.1), c_active / c_g * (alp * np.arange(0, 10, 0.1) / (1 + alp * np.arange(0, 10, 0.1))))
    plt.ylabel('active velocity ($1/c_g$)', fontsize=20)
    plt.xlabel('total velocity from last time v(t-1)', fontsize=20)
    plt.show()

    plt.close()


def visualization(if_adap, rp, R, actf, rw, RW, FWfeel, tot_Fwall, force, time,
                  c_active=0, alp=None, ad_f=None, pas_force=None, n_vec_theta=None, c_g=None):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect='equal')

    plot_circles(fig, ax, rp[:, 0], rp[:, 1], R, actf)

    fig, p = plot_circles(fig, ax, rw[:, 0], rw[:, 1], RW, la.norm(FWfeel, axis=1), cmap=matplotlib.cm.coolwarm,
                          ticks=[0, 100])
    fig.colorbar(p, ax=ax, orientation="horizontal", fraction=0.07, anchor=(1.0, 0.0))

    plt.xlabel('force on wall = %d' % tot_Fwall, fontsize=15)
    ax.quiver(rp[:, 0], rp[:, 1], force[:, 0], force[:, 1], linewidth=25, headwidth=10, units='xy', scale=10)
    ax.quiver(rw[:, 0], rw[:, 1], FWfeel[:, 0], FWfeel[:, 1],
              linewidth=25, headwidth=10, units='xy', scale=10, color='r')
    if c_g:
        plt.title('time at %0.2f' % time + ' with active = %0.1f c_g' % (c_active / c_g), fontsize=15)
    else:
        plt.title('time at %0.2f' % time + ' with active = %0.1f ' % c_active, fontsize=15)
    plt.xlim(np.amin(rw[:, 0] - 2 * RW), np.amax(rw[:, 0] + 2 * RW))
    plt.ylim(np.amin(rw[:, 1] - 2 * RW), np.amax(rw[:, 1] + 4 * RW))

    plt.yticks([])

    if if_adap and c_active != 0:
        plt.axes([0.8, 0.3, .2, .2])
        xx = np.arange(0, 20, 0.5)
        plt.plot(xx, (alp * xx / (1 + alp * xx)))

        plt.scatter(ad_f, actf, c=actf, cmap=matplotlib.cm.rainbow,
                    vmin=0, vmax=1, alpha=1.)
        plt.colorbar()
        plt.ylabel('active velocity ', fontsize=15)
        plt.xlabel('$\|f(t-1)*\hat{n}\|$', fontsize=15)
        plt.yticks([])

    if len(force[:, 0]) == 1:
        plt.axes([0.6, 0.6, .2, .2])
        plt.annotate("", xy=(pas_force[:, 0], pas_force[:, 1]), xytext=(0, 0),
                     arrowprops=dict(facecolor='b', shrink=0.05), alpha=0.5)

        plt.annotate("", xy=(c_active * actf * n_vec_theta[:, 0], c_active * actf * n_vec_theta[:, 1]), xytext=(0, 0),
                     arrowprops=dict(facecolor='r', shrink=0.05))

        if c_active != 0:
            plt.ylim(-c_g / 2. - c_active, c_g / 2. + c_active)
            plt.xlim(-c_g / 2. - c_active, c_g / 2. + c_active)
        else:
            plt.ylim(-2 * c_g, 2 * c_g)
            plt.xlim(-2 * c_g, 2 * c_g)
        plt.xticks([])
        plt.yticks([])
        plt.title('passive(b) vs active(r)', fontsize=15)

    return fig
