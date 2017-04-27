import unittest

import numpy as np
import healpy as H
from mpi4py import MPI
import pylab as pl

import cmb_analysis.powerspectrum.modemixing_matrices as mm
import cmb_analysis.powerspectrum.healpy_ext as H_ext
	
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#window_scal = H.read_map('testwindow.fits')
#window_scal = H.ud_grade(window_scal, 32)

true_cls = np.array(H.read_cl('testcl.fits'))

def construct_window(nside, center, width):

    nside = 32
    npix = H.nside2npix(nside)

    center = np.pi/2.0 - np.radians(center)
    width = np.radians(width)

    hpmap = np.zeros(npix)

    theta, phi = H.pix2ang(nside, np.arange(npix))

    hpmap[np.abs(theta - center) < width] = 1.0

    delta_c = np.radians(15.0)
    theta_max = np.max(theta[hpmap == 1])
    theta_min = np.min(theta[hpmap == 1])

    theta_diff_min = theta - theta_min
    theta_diff_max = theta_max - theta

    apodized_mask = np.ones_like(hpmap)

    apodized_mask[theta_diff_min < 0] = 0
    apodized_mask[theta_diff_max < 0] = 0

    idx1 = np.where(theta_diff_min >= 0)[0]
    idx2 = np.where(theta_diff_min <= delta_c)[0]
    idx = np.intersect1d(idx1,idx2)
    delta_i = theta_diff_min[idx]
    apodized_mask[idx] = -1.0 / (2*np.pi) * np.sin(2*np.pi*delta_i / delta_c) + delta_i / delta_c
	
    idx1 = np.where(theta_diff_max >= 0)[0]
    idx2 = np.where(theta_diff_max <= delta_c)[0]
    idx = np.intersect1d(idx1,idx2)
    delta_i = theta_diff_max[idx]
    apodized_mask[idx] = -1.0 / (2*np.pi) * np.sin(2*np.pi*delta_i / delta_c) + delta_i / delta_c

    return apodized_mask

def construct_cls(cltype, window_scal):

    niter = 100

    nside = H.npix2nside(len(window_scal))

    mask = np.ones_like(window_scal)
    mask[window_scal == 0] = 0.0
    
    cls_avg = None
    for i in range(rank, niter, size):
        print("Iter: ", i)
        cmbmap = H.synfast(true_cls, nside, new=True, verbose=False)
        
        if cltype is 'pseudo':
            cmbmap = (cmbmap[0]*window_scal, cmbmap[1]*window_scal, 
                      cmbmap[2]*window_scal)

            cls = H_ext.anafast(cmbmap)
        elif cltype is 'pure':
            cls = H_ext.pureanafast(cmbmap, window_scal, mask=mask)
        elif cltype is 'hybrid':
            cls = H_ext.hybridanafast(cmbmap, window_scal, mask=mask)

        if cls_avg is None:
            cls_avg = np.array(cls)
        else:
            cls_avg += np.array(cls)
            
    cls_avg = comm.allreduce(cls_avg)
    cls_avg /= niter

    return cls_avg

class TestModeMixing(unittest.TestCase):
    
    def test_compare_purecls(self):

        print("Testing pure")
    
        nside = 32
        window_scal = construct_window(nside, -28.0, 57.0)

        cls = construct_cls('pure', window_scal)
        
        nell = np.shape(cls)[1]
        cls_fs = np.zeros_like(cls)
        cls_fs[:4, :] = true_cls[:, :nell]

        mout = mm.calc_modemixing(comm, window_scal, cl_type='pure',
                                  verbose=False)

        cl_ps = mm.apply_coupling(cls_fs, mout[0], mout[1], mout[2],
                                    inverse=False)

        if comm.rank == 0:
            pl.figure()
            pl.loglog(cls[0], 'b')
            pl.loglog(cls[1], 'b')
            pl.loglog(cls[2], 'b')
            pl.loglog(cls[3], 'b')
            pl.loglog(cl_ps[0], 'r')
            pl.loglog(cl_ps[1], 'r')
            pl.loglog(cl_ps[2], 'r')
            pl.loglog(cl_ps[3], 'r')
            pl.xlabel(r'$\ell$')
            pl.ylabel(r'$C_{\ell}$')
            pl.title('Pure Cls')
            pl.savefig('purecls_compare_32.png')
            pl.show()

        
#    def test_compare_pseudocls(self):
#        
#        print("Testing pseudo")
#
#        cls = construct_cls('pseudo')
#
#        nell = np.shape(cls)[1]
#        cls_fs = np.zeros_like(cls)
#        cls_fs[:4, :] = true_cls[:, :nell]
#      
#        mout = mm.calc_modemixing(comm, window_scal, cl_type='pseudo',
#                                  verbose=False)
#       
#        cl_ps = mm.apply_coupling(cls_fs, mout[0], mout[1], mout[2],
#                                    inverse=False)
#      
#        if comm.rank == 0:
#            pl.figure()
#            pl.loglog(cls[0], 'b')
#            pl.loglog(cls[1], 'b')
#            pl.loglog(cls[2], 'b')
#            pl.loglog(cls[3], 'b')
#            pl.loglog(cl_ps[0], 'r')
#            pl.loglog(cl_ps[1], 'r')
#            pl.loglog(cl_ps[2], 'r')
#            pl.loglog(cl_ps[3], 'r')
#            pl.xlabel(r'$\ell$')
#            pl.ylabel(r'$C_{\ell}$')
#            pl.title('Pseudo Cls')
#            pl.savefig('pseudocls_compare_32.png')
#
#         
#    def test_compare_hybridcls(self):
#        
#        print("Testing hybrid")
#
#        cls = construct_cls('hybrid')
#        
#        nell = np.shape(cls)[1]
#        cls_fs = np.zeros_like(cls)
#        cls_fs[:4, :] = true_cls[:, :nell]
#        
#        mout = mm.calc_modemixing(comm, window_scal, cl_type='hybrid',
#                                  verbose=False)
#        
#        cl_ps = mm.apply_coupling(cls_fs, mout[0], mout[1], mout[2],
#                                    inverse=False)
#        
#        if comm.rank == 0:
#            pl.figure()
#            pl.loglog(cls[0], 'b')
#            pl.loglog(cls[1], 'b')
#            pl.loglog(cls[2], 'b')
#            pl.loglog(cls[3], 'b')
#            pl.loglog(cl_ps[0], 'r')
#            pl.loglog(cl_ps[1], 'r')
#            pl.loglog(cl_ps[2], 'r')
#            pl.loglog(cl_ps[3], 'r')
#            pl.xlabel(r'$\ell$')
#            pl.ylabel(r'$C_{\ell}$')
#            pl.title('Hybrid Cls')
#            pl.savefig('hybridcls_compare.png')


if __name__ == '__main__':
    unittest.main()
