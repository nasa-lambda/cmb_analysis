'''Test code to see whether the modemixing code is working. This just
compares specific elements of each matrix to values I have calculated.
At least one element of each block is tested. 

These matrices have also been tested by comparing plots of expected Cls
(mode-mixing applied to input Cls) to measured Cls (calculated
pseudo/pure/hybrid Cls) and not seeing a bias.
'''

import unittest

import numpy as np
import pylab as pl
import healpy as H

from mpi4py import MPI

import cmb_analysis.powerspectrum.modemixing_matrices as mm
import cmb_analysis.powerspectrum.healpy_ext as H_ext
	
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

window_scal = H.read_map('testwindow.fits')
window_scal = H.ud_grade(window_scal, 16)

class TestModeMixing(unittest.TestCase):

    def test_pseudo_scalar(self):
        mm_scal = mm.calc_modemixing(comm, window_scal, cl_type='pseudo',
                                     pol=False, cross=False,
                                     verbose=False)[0]

        np.testing.assert_almost_equal(mm_scal[30, 4], 1.83739217175e-06)
        np.testing.assert_almost_equal(mm_scal[4, 8], 0.00292934623597)
        np.testing.assert_almost_equal(mm_scal[29, 5], 9.3980091378e-06)
        np.testing.assert_almost_equal(mm_scal[28, 41], 0.000352022057911)
        np.testing.assert_almost_equal(mm_scal[16, 47], 1.40748553658e-06)
        np.testing.assert_almost_equal(mm_scal[41, 36], 0.00368015553055)
        np.testing.assert_almost_equal(mm_scal[47, 20], 3.1203993093e-06)
    
    def test_pseudo_pol(self):
        mm_pol = mm.calc_modemixing(comm, window_scal, cl_type='pseudo',
                                    scal=False, cross=False, verbose=False)[0]
         
        np.testing.assert_almost_equal(mm_pol[41, 12], 1.5635136702e-06)
        np.testing.assert_almost_equal(mm_pol[28, 45], 8.15495138582e-05)
        np.testing.assert_almost_equal(mm_pol[45, 16], 1.83295683132e-06)
        np.testing.assert_almost_equal(mm_pol[40, 37], 0.00448854304032)
        np.testing.assert_almost_equal(mm_pol[20, 10], 0.000172198520876)
        np.testing.assert_almost_equal(mm_pol[25, 20], 0.00322469605068)
        np.testing.assert_almost_equal(mm_pol[40, 44], 0.00190728298179)
        np.testing.assert_almost_equal(mm_pol[65, 11], 0.000260619008503)
        np.testing.assert_almost_equal(mm_pol[85, 7], 2.42224507982e-07)
        np.testing.assert_almost_equal(mm_pol[77, 28], 0.000756936665183)
        np.testing.assert_almost_equal(mm_pol[17, 65], 0.0023636268951)
        np.testing.assert_almost_equal(mm_pol[12, 12], 0.464851591327)
        np.testing.assert_almost_equal(mm_pol[75, 18], 9.88603062258e-05)
        np.testing.assert_almost_equal(mm_pol[134, 5], -0.0)
        np.testing.assert_almost_equal(mm_pol[11, 79], 2.85444772107e-05)
        np.testing.assert_almost_equal(mm_pol[84, 84], 0.467545290364)
        np.testing.assert_almost_equal(mm_pol[139, 86], 0.0)
        np.testing.assert_almost_equal(mm_pol[46, 124], 0.0)
        np.testing.assert_almost_equal(mm_pol[58, 136], -0.0)
        np.testing.assert_almost_equal(mm_pol[133, 130], 0.00424061130075)

    def test_pseudo_cross(self):
        mm_cross = mm.calc_modemixing(comm, window_scal, cl_type='pseudo',
                                      pol=False, scal=False,
                                      verbose=False)[0]
        
        np.testing.assert_almost_equal(mm_cross[16, 46], 1.40681352501e-06)
        np.testing.assert_almost_equal(mm_cross[15, 40], 1.26088840863e-05)
        np.testing.assert_almost_equal(mm_cross[2, 27], 8.78711650258e-06)
        np.testing.assert_almost_equal(mm_cross[46, 5], 3.15438544537e-09)
        np.testing.assert_almost_equal(mm_cross[5, 18], 0.000284025893773)
        np.testing.assert_almost_equal(mm_cross[42, 15], 1.97009670125e-06)
        np.testing.assert_almost_equal(mm_cross[27, 10], 2.96741369019e-05)
        np.testing.assert_almost_equal(mm_cross[43, 40], 0.00461645748156)
        np.testing.assert_almost_equal(mm_cross[65, 26], -0.0)
        np.testing.assert_almost_equal(mm_cross[23, 81], 0.0)
        np.testing.assert_almost_equal(mm_cross[83, 68], 3.12688926846e-05)
    
    def test_pure_scalar(self):
        mm_scal = mm.calc_modemixing(comm, window_scal, cl_type='pure',
                                     pol=False, cross=False,
                                     verbose=False)[0]
        
        np.testing.assert_almost_equal(mm_scal[12, 34], 2.84087393862e-05)
        np.testing.assert_almost_equal(mm_scal[37, 41], 0.00206036765534)
        np.testing.assert_almost_equal(mm_scal[36, 27], 0.000523346173846)
        np.testing.assert_almost_equal(mm_scal[18, 23], 0.00443742959833)
        np.testing.assert_almost_equal(mm_scal[11, 33], 2.93599786354e-05)
        np.testing.assert_almost_equal(mm_scal[34, 29], 0.00363090478666)
        np.testing.assert_almost_equal(mm_scal[45, 9], 1.16988045161e-07)
    
    def test_pure_pol(self):
        mm_pol = mm.calc_modemixing(comm, window_scal, cl_type='pure',
                                    scal=False, cross=False,
                                    verbose=False)[0]
        
        np.testing.assert_almost_equal(mm_pol[36, 36], 0.468043898244)
        np.testing.assert_almost_equal(mm_pol[32, 2], 2.20756688177e-11)
        np.testing.assert_almost_equal(mm_pol[21, 15], 0.000476616710767)
        np.testing.assert_almost_equal(mm_pol[21, 14], 9.37935582769e-05)
        np.testing.assert_almost_equal(mm_pol[27, 7], 7.46812040813e-08)
        np.testing.assert_almost_equal(mm_pol[25, 41], 0.00104278390722)
        np.testing.assert_almost_equal(mm_pol[34, 11], 6.02486266177e-08)
        np.testing.assert_almost_equal(mm_pol[37, 31], 0.000945580570887)
        np.testing.assert_almost_equal(mm_pol[57, 45], 5.57274213948e-09)
        np.testing.assert_almost_equal(mm_pol[134, 6], 2.44464239686e-27)
        np.testing.assert_almost_equal(mm_pol[31, 73], 2.9988511577e-10)
        np.testing.assert_almost_equal(mm_pol[51, 69], 0.225736386389)
        np.testing.assert_almost_equal(mm_pol[102, 52], -7.57535653275e-23)
        np.testing.assert_almost_equal(mm_pol[30, 128], -3.18790737965e-23)
        np.testing.assert_almost_equal(mm_pol[89, 121], 4.32050755775e-24)
        np.testing.assert_almost_equal(mm_pol[137, 113], 4.1216329779e-07)
    
    def test_pure_cross(self):
        mm_cross = mm.calc_modemixing(comm, window_scal, cl_type='pure',
                                      pol=False, scal=False,
                                      verbose=False)[0]

        np.testing.assert_almost_equal(mm_cross[46, 18], 3.23037295968e-07)
        np.testing.assert_almost_equal(mm_cross[6, 11], 0.0172056255349)
        np.testing.assert_almost_equal(mm_cross[38, 31], 0.000328486236927)
        np.testing.assert_almost_equal(mm_cross[46, 5], -1.31016755125e-10)
        np.testing.assert_almost_equal(mm_cross[36, 37], 0.05520788714)
        np.testing.assert_almost_equal(mm_cross[15, 32], 0.000503794496037)
        np.testing.assert_almost_equal(mm_cross[35, 41], 0.0030642343253)
        np.testing.assert_almost_equal(mm_cross[13, 8], 0.00124872421814)
        np.testing.assert_almost_equal(mm_cross[80, 45], 5.35657547012e-25)
        np.testing.assert_almost_equal(mm_cross[20, 78], 1.6696281983e-24)
        np.testing.assert_almost_equal(mm_cross[63, 75], 0.000703420779726)

    def test_hybrid_scalar(self):
        mm_scal = mm.calc_modemixing(comm, window_scal, cl_type='hybrid',
                                     pol=False, cross=False,
                                     verbose=False)[0]

        np.testing.assert_almost_equal(mm_scal[35, 44], 0.00067367809385)
        np.testing.assert_almost_equal(mm_scal[11, 13], 0.0270362160258)
        np.testing.assert_almost_equal(mm_scal[4, 11], 0.000970983202322)
        np.testing.assert_almost_equal(mm_scal[41, 44], 0.00506132682973)
        np.testing.assert_almost_equal(mm_scal[41, 31], 0.000325425735452)
        np.testing.assert_almost_equal(mm_scal[39, 11], 1.75798736859e-06)
        np.testing.assert_almost_equal(mm_scal[30, 43], 0.000348206051517)
    
    def test_hybrid_pol(self):
        mm_pol = mm.calc_modemixing(comm, window_scal, cl_type='hybrid',
                                    scal=False, cross=False,
                                    verbose=False)[0]
         
        np.testing.assert_almost_equal(mm_pol[41, 10], 1.32552171566e-07)
        np.testing.assert_almost_equal(mm_pol[36, 7], 1.18139414103e-06)
        np.testing.assert_almost_equal(mm_pol[32, 24], 0.000501776385597)
        np.testing.assert_almost_equal(mm_pol[37, 31], 0.0017734712428)
        np.testing.assert_almost_equal(mm_pol[24, 4], 7.80795825724e-06)
        np.testing.assert_almost_equal(mm_pol[14, 9], 0.00255576684036)
        np.testing.assert_almost_equal(mm_pol[13, 7], 0.00105695372429)
        np.testing.assert_almost_equal(mm_pol[20, 12], 0.000374762672575)
        np.testing.assert_almost_equal(mm_pol[51, 3], 1.72225632388e-07)
        np.testing.assert_almost_equal(mm_pol[116, 19], 1.1788580868e-23)
        np.testing.assert_almost_equal(mm_pol[30, 64], 2.66133576119e-05)
        np.testing.assert_almost_equal(mm_pol[66, 67], 0.0654840598492)
        np.testing.assert_almost_equal(mm_pol[117, 67], 7.39784036002e-24)
        np.testing.assert_almost_equal(mm_pol[22, 126], 0.0)
        np.testing.assert_almost_equal(mm_pol[59, 104], 1.50373284165e-23)
        np.testing.assert_almost_equal(mm_pol[109, 135], 3.12975982554e-05)

    def test_hybrid_cross(self):
        mm_cross = mm.calc_modemixing(comm, window_scal, cl_type='hybrid',
                                      pol=False, scal=False,
                                      verbose=False)[0]
         
        np.testing.assert_almost_equal(mm_cross[46, 46], 0.467876088348)
        np.testing.assert_almost_equal(mm_cross[29, 25], 0.00166319815048)
        np.testing.assert_almost_equal(mm_cross[31, 30], 0.0504595237057)
        np.testing.assert_almost_equal(mm_cross[14, 43], 5.56989806163e-06)
        np.testing.assert_almost_equal(mm_cross[3, 3], 0.451031280482)
        np.testing.assert_almost_equal(mm_cross[34, 4], -2.93480604569e-07)
        np.testing.assert_almost_equal(mm_cross[2, 10], 7.71939346299e-05)
        np.testing.assert_almost_equal(mm_cross[18, 22], 0.00183444442182)
        np.testing.assert_almost_equal(mm_cross[80, 26], -1.0874660055e-23)
        np.testing.assert_almost_equal(mm_cross[42, 59], 0.0)
        np.testing.assert_almost_equal(mm_cross[89, 78], 7.85910067304e-05)

if __name__ == '__main__':
    unittest.main()

