import numpy as np
import matplotlib.pyplot as plt
import genericio as gio
import pdb
import glob
import matplotlib.pyplot as plt
import itertools
import dtk

def fof_sod_comp(minStep=401):

    p = '/projects/DarkUniverse_esp/jphollowed/outerRim/lightcone_halos_octant_matchup/'
    
    f = np.array(glob.glob('{}/lcHalos*/*'.format(p)))
    f_header_mask = ['#' not in ff for ff in f]
    f = f[f_header_mask]
    f_step_mask = [int(ff.split('.')[-1]) >= minStep for ff in f]
    f = f[f_step_mask]
    
    print('reading')
    fof = np.array(list(itertools.chain.from_iterable([list(np.squeeze(gio.gio_read(ff, 'fof_mass'))) for ff in f])))
    sod = np.array(list(itertools.chain.from_iterable([list(np.squeeze(gio.gio_read(ff, 'sod_mass'))) for ff in f])))

    print('plotting')
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    
    ax.plot(sod, fof, '.m')
    ax.plot([np.min(fof), np.max(fof)], [np.min(fof), np.max(fof)], '--k')
    ax.set_xlabel('M_200', fontsize=14)
    ax.set_ylabel('FOF Mass', fontsize=14)
    #ax1.set_xlim([])

    ax2.plot(sod, fof, '.m')
    ax2.plot([np.min(fof), np.max(fof)], [np.min(fof), np.max(fof)], '--k')
    ax2.set_xlabel('M_200', fontsize=14)
    ax2.set_ylabel('FOF Mass', fontsize=14)
    ax2.set_xlim([-.25e13, 2e13])
    ax2.set_ylim([-.25e13, 2e13])

    plt.tight_layout()
    plt.show()


def velocity_check(minStep=442):

    old = '/projects/DarkUniverse_esp/jphollowed/outerRim/lightcone_halos_octant'
    new = '/projects/DarkUniverse_esp/jphollowed/outerRim/lightcone_halos_octant_matchup/'

    oldf = np.array(glob.glob('{}/lcHalos*/*'.format(old)))
    oldf_header_mask = ['#' not in f for f in oldf]
    oldf = oldf[oldf_header_mask]
    oldf_step_mask = [int(f.split('.')[-1]) >= minStep for f in oldf]
    oldf = oldf[oldf_step_mask]

    newf = np.array(glob.glob('{}/lcHalos*/*'.format(new)))
    newf_header_mask = ['#' not in f for f in newf]
    newf = newf[newf_header_mask]
    newf_step_mask = [int(f.split('.')[-1]) >= minStep for f in newf]
    newf = newf[newf_step_mask]

    print('reading old')
    old_tags = np.array(list(itertools.chain.from_iterable([list(np.squeeze(gio.gio_read(f, 'id'))) for f in oldf])))
    old_vx = np.array(list(itertools.chain.from_iterable([list(np.squeeze(gio.gio_read(f, 'vx'))) for f in oldf])))
    old_vy = np.array(list(itertools.chain.from_iterable([list(np.squeeze(gio.gio_read(f, 'vy'))) for f in oldf])))
    old_vz = np.array(list(itertools.chain.from_iterable([list(np.squeeze(gio.gio_read(f, 'vz'))) for f in oldf])))

    first_tag_old = 1073741793593
    idx = np.where(old_tags==1073741793593)[0][0]
    print('first old vel is {}, {}, {} for {}'.format(old_vx[idx], old_vy[idx], old_vz[idx], first_tag_old))

    print('reading new')
    new_tags = np.array(list(itertools.chain.from_iterable([list(np.squeeze(gio.gio_read(f, 'id'))) for f in newf])))
    new_vx = np.array(list(itertools.chain.from_iterable([list(np.squeeze(gio.gio_read(f, 'vx'))) for f in newf])))
    new_vy = np.array(list(itertools.chain.from_iterable([list(np.squeeze(gio.gio_read(f, 'vy'))) for f in newf])))
    new_vz = np.array(list(itertools.chain.from_iterable([list(np.squeeze(gio.gio_read(f, 'vz'))) for f in newf])))

    first_tag_new = 1073741793593
    idx = np.where(new_tags==1073741793593)[0][0]
    print('first new vel is {}, {}, {} for {}'.format(new_vx[idx], new_vy[idx], new_vz[idx], first_tag_new))

    print('removing duplicates')
    old_uniq, old_counts = np.unique(old_tags, return_counts=True)
    bad_tags = old_uniq[old_counts > 1]
    dupl_mask_old = np.array([t not in bad_tags for t in old_tags])
    old_tags = old_tags[dupl_mask_old]

    new_uniq, new_counts = np.unique(new_tags, return_counts=True)
    bad_tags = new_uniq[new_counts > 1]
    dupl_mask_new = np.array([t not in bad_tags for t in new_tags])
    new_tags = new_tags[dupl_mask_new]

    print('matching')
    new_srt = np.argsort(new_tags) 
    old_srt = np.argsort(old_tags) 

    sort_left_new = new_tags[new_srt].searchsorted(old_tags[old_srt], side='left')
    sort_right_new = new_tags[new_srt].searchsorted(old_tags[old_srt], side='right')

    sort_left_old = old_tags[old_srt].searchsorted(new_tags[new_srt], side='left')
    sort_right_old = old_tags[old_srt].searchsorted(new_tags[new_srt], side='right')

    inds_old = (sort_right_new-sort_left_new > 0).nonzero()[0]
    inds_new = (sort_right_old-sort_left_old > 0).nonzero()[0]

    new_tags = new_tags[new_srt[inds_new]]
    new_vx = new_vx[dupl_mask_new][new_srt[inds_new]]
    new_vy = new_vy[dupl_mask_new][new_srt[inds_new]]
    new_vz = new_vz[dupl_mask_new][new_srt[inds_new]]

    old_tags = old_tags[old_srt[inds_old]]
    old_vx = old_vx[dupl_mask_old][old_srt[inds_old]]
    old_vy = old_vy[dupl_mask_old][old_srt[inds_old]]
    old_vz = old_vz[dupl_mask_old][old_srt[inds_old]]

    print('first old vel is {}, {}, {} for {}'.format(old_vx[np.where(old_tags==first_tag_old)[0][0]], 
                                                      old_vy[np.where(old_tags==first_tag_old)[0][0]],
                                                      old_vz[np.where(old_tags==first_tag_old)[0][0]],
                                                      old_tags[np.where(old_tags==first_tag_old)[0][0]]))
    print('first new vel is {}, {}, {} for {}'.format(new_vx[np.where(new_tags==first_tag_new)[0][0]], 
                                                      new_vy[np.where(new_tags==first_tag_new)[0][0]],
                                                      new_vz[np.where(new_tags==first_tag_new)[0][0]],
                                                      new_tags[np.where(new_tags==first_tag_new)[0][0]]))


    print('plotting')
    fig = plt.figure(0)
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    h = ax1.hist((new_vx/old_vx)[~np.isnan(new_vx/old_vx)], bins=100, histtype='step', lw=1.5, color='r')
    ax1.hist((new_vx/old_vx)[~np.isnan(new_vx/old_vx)], bins=h[1], lw=0, color='m', alpha=0.5)
    ax1.set_xlabel('vx', fontsize=14)
    ax1.set_xlim([0.85, 1.10])
    ax1.set_ylabel('New/DC2')
    ax1.grid()

    h = ax2.hist((new_vy/old_vy)[~np.isnan(new_vy/old_vy)], bins=100, histtype='step', lw=1.5, color='r')
    ax2.hist((new_vy/old_vy)[~np.isnan(new_vy/old_vy)], bins=h[1], lw=0, color='m', alpha=0.5)
    ax2.set_xlabel('vy', fontsize=14)
    ax2.set_xlim([0.85, 1.10])
    ax2.grid()

    h = ax3.hist((new_vz/old_vz)[~np.isnan(new_vz/old_vz)], bins=100, histtype='step', lw=1.5, color='r')
    ax3.hist((new_vz/old_vz)[~np.isnan(new_vz/old_vz)], bins=h[1], lw=0, color='m', alpha=0.5)
    ax3.set_xlabel('vz', fontsize=14)
    ax3.set_xlim([0.85, 1.10])
    ax3.grid()

    plt.tight_layout
    plt.show()

