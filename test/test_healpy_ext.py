import unittest

import numpy as np
import pylab as pl
import healpy as H

import cmb_analysis.powerspectrum.healpy_ext as H_ext

testmap = H.read_map('testmap.fits', field=(0, 1, 2, 3))
window_scal = H.read_map('testwindow.fits')

class TestPureCl(unittest.TestCase):

    def test_purecl(self):

        cls_good = np.loadtxt('purecls.txt')

        mask = np.ones_like(window_scal)
        mask[window_scal == 0] = 0.0

        idx = np.isnan(testmap[1])
        testmap[1][idx] = 0.0
        testmap[2][idx] = 0.0
        testmap[3][idx] = 0.0

        cls_pure = H_ext.pureanafast(testmap, window_scal, mask=mask, iter=0)

        np.testing.assert_almost_equal(cls_pure, cls_good)
    

if __name__ == '__main__':
    unittest.main()
    #test_window()
