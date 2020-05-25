import argparse
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la

import lib.visual_lib as vl
from lib import func as func

parser = argparse.ArgumentParser(description='Active robot model')
parser.add_argument('--save_path', metavar='data path', type=str, help='save path',
                    default='/Users/jiayiwu/projects/softrobot/experiment/')
parser.add_argument('--dt', type=float, help='time step', default=0.005)
parser.add_argument('--tau', type=int, help='Max iteration time (number of iteration/dt)', default=40)

parser.add_argument('--eta', type=float, help='noise level', default=0.1)
parser.add_argument('--R', type=float, help='Radius of beads', default=0.5)
parser.add_argument('--RW', type=float, help='Radius of wall',
                    default=3)

parser.add_argument('--FR', type=int, help='cutoff range of force between beads (repulsion if surface distance < R)',
                    default=0)
parser.add_argument('--c_LJ', type=float, help='L-J force parameter', default=1.)
parser.add_argument('--alp', type=float, help='ratio between active and non-active suggesting: [0-5]', default=0.5)
parser.add_argument('--if_adap', type=bool, help='if active or not', default=True)

parser.add_argument('--c_active', type=float, help='active amplitude', default=2)
parser.add_argument('--c_g', type=float, help='travel downward velocity ', default=2)
parser.add_argument('--las', type=float, help='between [0,1] when las=1 the change of theta is instantaneous ', default=0.1)

def write_par():
    '''write parameters in the path named Datapath '''
    f = open(Datapath + "/parameters_run.txt", "w+")
    f.write('intrinsic velocity of each active particle: \r\n' + str(c_g) + '\n')

    f.write('time step dt : \r\n' + str(dt) + '\n')
    f.write('Max iteration time : \r\n' + str(tau) + '\n')

    f.write('Number of the active particle :' + '\r\n' + str(N_p) + '\n')
    f.write('Number of the Wall  :' + '\r\n' + str(N_w) + '\n')

    f.write('Diameter of the active particle :' + '\r\n' + str(2 * R) + '\n')
    f.write('Diameter of the wall : \r\n' + str((DW - R) * 2) + '\n')
    f.write('noise level : \r\n' + str(eta) + '\n')

    f.write('LJ strength : \r\n' + str(c_LJ) + '\n')
    f.write('MAX cutoff dis for the LJ between particle and wall : \r\n' + str(DW + FWR) + '\n')
    f.write('MIN cutoff dis for the LJ between particle and wall : \r\n' + str(0.8 * DW) + '\n')

    f.write('Kspring : \r\n' + str(kspring) + '\n')
    f.write('cutoff dis from equilibrium dis for the spring between particle and particle : \r\n' + str(FR) + '\n')

    if c_active == 0:
        f.write('NOT active''\n')
    else:
        f.write('active''\n')
        f.write('alpha(las in the code), continuous changing for theta:\r\n' + str(las) + '\n')
        if if_adap:
            f.write('parameters describing the adaptive force(A*(alp * ad_f / (1 + alp * ad_f))), A and alp:\r\n' + str(
                c_active) + ',' + str(alp) + '\n')
    f.close()


def initialize():
    '''
    initialize positions of wall and particles
    :return:
    rb: initial positions of particles of robot, 2d array (x,y)
    s_id: index of the skin particles corresponding to positions in rb
    rw: position of walls, 2d array (x,y)
    thetas_vec: initialize direction of movement of each particle inside the robot
    ad_f: use for calculating actf (for plotting only)
    actf: active force
    n_b: number of body particles
    n_w: number of wall particles
    '''

    rb, _, s_id = make_s_b()
    n_b = len(rb)

    rw = func.make_hex(3, 2)
    rw[:, 0] = rw[:, 0].copy() * (max(rb[:, 0]) * 1.5 + 2.5 * RW) - DW
    rw[:, 1] = rw[:, 1].copy() * (max(rb[:, 1]) * 1.5 + 2.5 * RW)
    rw[:, 1] += -max(rw[:, 1].copy()) - DW * np.max(rb[:, 1])

    n_w = len(rw)

    thetas_vec = np.zeros((n_b, 2))
    thetas_vec[s_id, 1] = np.tile([-1], len(s_id))

    ad_f = np.zeros(n_b)
    ad_f[s_id] = 1
    if if_adap:
        actf = (alp * ad_f / (1 + alp * ad_f))

    else:
        actf = ad_f

    return rb, s_id, rw, thetas_vec, ad_f, actf, n_b, n_w


def make_s_b():
    '''
    make skin and body particles (hex packing)

    :return:
    tot: all particles
    skin: skin particles
    s_id: index of the skin particles corresponding to positions in skin
    '''
    body = func.make_hex(2, 1)
    body[:, 1] = body[:, 1] - np.mean(body[:, 1])
    body = np.round(body, 3)

    ra = max(body[:, 0]) + 2 * R
    rb = max(body[:, 1]) + 2 * R
    big_r = max(ra, rb)
    cir = 2 * np.pi * big_r

    skin = func.PointsInCircum(ra, rb, n=int(cir))[:int(cir)]  # max(int(cir),len(body)))
    tot = np.concatenate((body, skin), axis=0)
    s_id = np.arange(len(body), len(body) + len(skin))

    return tot, skin, s_id


def make_path(save_path):
    Timenow = str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    Datapath = save_path + Timenow
    if not os.path.exists(Datapath): os.mkdir(Datapath)
    return Datapath


if __name__ == "__main__":
    args = parser.parse_args()
    dt = args.dt

    tau = args.tau
    eta = args.eta
    sqdt = np.sqrt(dt)  # noise

    R = args.R  # radius of beads
    D = 2 * R
    RW = args.RW  # radius of Wall
    DW = RW + R  # distance between center of bead and center of the Wall
    if not args.FR:
        FR = R  # cutoff range of force between beads(repulsion if surface distance < R)
    else:
        FR = args.FR
    FWR = 2 * R  # cutoff range of force between bead and wall (repulsion if surface distance < 2*R)
    c_LJ = args.c_LJ

    c_cutoff = c_LJ * (12 * (DW / (0.8 * DW)) ** (13))

    alp = args.alp  # [0,5]

    if_adap = args.if_adap
    c_active = args.c_active
    c_g = args.c_g

    las = args.las
    tot_Fwall = 0  # total force acting on wall
    kspring = 5 * (c_g + c_active)
    kskin = 10 * (c_g + c_active)

    AVEFORCES = []
    TIMES = []

    if not if_adap:
        print('its not adaptive')
    if las != 1:
        print('not instantaneous')

    rp, s_id, rw, n_vec_theta, ad_f, actf, N_p, N_w = initialize()
    vec_theta = np.zeros(np.shape(n_vec_theta))

    # if c_active != 0 and if_adap:
    #     vl.plot_active_force(c_active, alp, c_g)

    tot_Fwall = 0
    Datapath = make_path(args.save_path)

    write_par()

    for time in np.arange(0, tau, dt):

        # initialize with the thermal noise force
        force = eta * (np.random.rand(N_p, 2) - 0.5) / sqdt
        FWfeel = np.zeros((N_w, 2))

        # loop around all movable particles:
        # calculate repulsive force
        for p_id in range(0, N_p):

            xy = rp[p_id, :]
            # %%%force with wall
            for i in range(0, N_w):
                vec_n = -rw[i, :] + xy
                dis = la.norm(vec_n)  # calculate distance

                if dis - DW < FR:  # only count when within range

                    strength = np.min([c_cutoff, c_LJ * (12 * (DW / dis) ** (13))])

                    force[p_id, :] += strength * vec_n / dis
                    FWfeel[i, :] += strength * vec_n / dis

            tot_Fwall += sum(la.norm(FWfeel, axis=1))
            # force with skin+skin particle
            if p_id in s_id:
                for i in s_id:  # all movable particles

                    if i != p_id:  # but can't interact with self

                        vec_n = -rp[i, :] + xy
                        dis = la.norm(vec_n)  # calculate distance
                        if dis - 2 * R < FR / 2.:
                            strength = np.round(-kskin * (dis - D), 2)
                            force[p_id, :] = force[p_id, :] + strength * vec_n / dis

            # force with other body+skin particle
            for i in range(N_p):  # all movable particles
                if i != p_id:  # but can't interact with self

                    vec_n = -rp[i, :] + xy
                    dis = la.norm(vec_n)  # calculate distance
                    if dis - 2 * R < FR / 2.:
                        strength = np.round(-kspring * (dis - D), 2)
                        force[p_id, :] = force[p_id, :] + strength * vec_n / dis

        force[:, 1] += -c_g
        pas_force = force.copy()

        force[:, 0][s_id] += c_active * (actf * n_vec_theta[:, 0])[s_id]
        force[:, 1][s_id] += c_active * (actf * n_vec_theta[:, 1])[s_id]

        rp += dt * force

        # data-saving and plotting
        if (time / dt) % 20 == 0:

            savedir = Datapath + '/rp_force_data/'

            if not os.path.exists(savedir): os.mkdir(savedir)
            np.save(savedir + '/%04d' % (time / dt), np.array([rp, force]))
            plt.clf()

            fig = vl.visualization(if_adap, rp, R, actf, rw, RW, FWfeel, tot_Fwall, force, time,
                                   c_active, alp, ad_f, pas_force, n_vec_theta, c_g)

            plotdir = Datapath + '/images/'
            if not os.path.exists(plotdir): os.mkdir(plotdir)

            plt.savefig(plotdir + '/%04d.png' % (time / dt), bbox_inches="tight")
            plt.close()

        # update active forces if c_active != 0
        if c_active != 0:
            vec_theta[s_id, 0] = c_active * actf[s_id] * n_vec_theta[s_id, 0] * (1 - las) + force[s_id, 0] * las
            vec_theta[s_id, 1] = c_active * actf[s_id] * n_vec_theta[s_id, 1] * (1 - las) + force[s_id, 1] * las
            n_vec_theta[s_id, 0] = vec_theta[s_id, 0] / la.norm(vec_theta, axis=1)[s_id]
            n_vec_theta[s_id, 1] = vec_theta[s_id, 1] / la.norm(vec_theta, axis=1)[s_id]

            if if_adap:
                ad_f[s_id] = abs(np.sum(force * n_vec_theta, axis=1))[s_id]
                actf = (alp * ad_f / (1 + alp * ad_f))

        # check if it goes through the maze
        if max(rp[:, 1]) < min(rw[:, 1]) - 5:
            print('all particles go through at time', time)
            print('with the total force exerted on wall', tot_Fwall)
            print('c_g is', c_g)

            AVEFORCES.append(tot_Fwall / time)
            TIMES.append(time)
            print(TIMES)
            func.make_movie(plotdir + '/', plotdir[:-8])

            break
