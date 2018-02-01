from __future__ import print_function

import unittest

import numpy as np
from astropy.coordinates import SkyCoord
import pylab as pl

from cmb_analysis.util.pixfunc import pix2coord, coord2pix

#All tests return ecliptic coordinates since we are not testing the ecliptic
#to other coordinate system transformations that are in astropy and those
#seem to introduce some differences where I can't compare the IDL and 
#Python outputs
class TestCobePix(unittest.TestCase):

    def test_pix2coord(self):
        
        decval = 5
        c = pix2coord(2698, res=6, coord='E')
        np.testing.assert_almost_equal(c.lon.value, 46.597906933090641, decimal=decval)
        np.testing.assert_almost_equal(c.lat.value, 24.654547941306923, decimal=decval)

        c = pix2coord(4315, res=6, coord='E')
        np.testing.assert_almost_equal(c.lon.value, 263.48454344246329, decimal=decval)
        np.testing.assert_almost_equal(c.lat.value, -11.706450756234322, decimal=decval)

        c = pix2coord([1903, 2302, 5895], res=6, coord='E')
        np.testing.assert_almost_equal(c[0].lon.value, 31.050688834849119, decimal=decval)
        np.testing.assert_almost_equal(c[0].lat.value, 17.461971132048095, decimal=decval)
        np.testing.assert_almost_equal(c[1].lon.value, 86.091536034965088, decimal=decval)
        np.testing.assert_almost_equal(c[1].lat.value, -1.2985060461313416, decimal=decval)
        np.testing.assert_almost_equal(c[2].lon.value, 67.027439872705656, decimal=decval)
        np.testing.assert_almost_equal(c[2].lat.value, -80.081092508758928, decimal=decval)


    def test_coord2pix(self):

        #One test for each of the different values for nface
        pix = coord2pix(1.47, 20.465, coord='geocentrictrueecliptic', res=6)
        np.testing.assert_equal(pix, 1834)
        pix = coord2pix(73.467, -45.245, coord='geocentrictrueecliptic', res=9)
        np.testing.assert_equal(pix, 384496)
        pix = coord2pix(180.1, -10.0, coord='E', res=6)
        np.testing.assert_equal(pix, 3488)
        pix = coord2pix(-90.1, -25.3, coord='E', res=6)
        np.testing.assert_equal(pix, 4221)
        pix = coord2pix(270.2, 82.7, coord='E', res=6)
        np.testing.assert_equal(pix, 251)
        pix = coord2pix(92.8, 27.3, coord='E', res=6)
        np.testing.assert_equal(pix, 2953)
        pix = coord2pix([270.2, 92.8], [82.7, 27.3], coord='E', res=6)
        np.testing.assert_equal(pix[0], 251)
        np.testing.assert_equal(pix[1], 2953)


