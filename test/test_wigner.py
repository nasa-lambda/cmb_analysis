import unittest

import numpy as np

import cmb_analysis.util.wignercoupling as wc

# Test cases for Wigner 3j and 6j symbols are taken from inputting random
# values into Wolfram Alpha and comparing results. Test cases for Wigner 9j
# symbols are taken from examples in 
# http://doc.sagemath.org/html/en/reference/functions/sage/functions/wigner.html


class TestWigner3j(unittest.TestCase):

    def test_special1(self):
        self.assertAlmostEqual(wc.wigner3j(2, 2, 0, 0, 0, 0),
                               -np.sqrt(1.0/3.0))
        self.assertAlmostEqual(wc.wigner3j(6*2, 4*2, 2*2, 0, 0, 0),
                               np.sqrt(5.0/143.0))

    def test_special2(self):
        self.assertAlmostEqual(wc.wigner3j(3, 9, 6, 1, 3, -4),
                               0.146385)

    def test_special3(self):
        self.assertAlmostEqual(wc.wigner3j(2*3, 2*3, 2*3, 1*2, -1*2, 0),
                               -0.1543033)

    def test_normal(self):
        self.assertAlmostEqual(wc.wigner3j(2*4, 2*5, 2*3, 2*2, -2*2, 0),
                               0.0215917)
        self.assertAlmostEqual(wc.wigner3j(5, 3, 4, 1, 1, -2),
                               -0.243975)
        self.assertAlmostEqual(wc.wigner3j(10, 12, 8, 2, -4, 2),
                               np.sqrt(143)/429)

    def test_clebschgordon(self):
        self.assertAlmostEqual(wc.clebsch_gordon(5, 1, 5, -1, 3*2, 2*2),
                               np.sqrt(1.0/6.0))

    def test_racahv(self):
        pass

class TestWigner6j(unittest.TestCase):

    def test_special1(self):
        self.assertAlmostEqual(wc.wigner6j(2*2, 2*0, 2*2, 2*2, 2*2, 2*2),
                               1.0 / 5)

    def test_special2(self):
        self.assertAlmostEqual(wc.wigner6j(10*2, 10*2, 6*2, 6*2, 9*2, 7*2),
                               -11*np.sqrt(49335)/111435)

    def test_special3(self):
        self.assertAlmostEqual(wc.wigner6j(2*2, 2*2, 2*2, 2*2, 2*2, 2*2),
                               -3.0/70)

    def test_normal(self):
        self.assertAlmostEqual(wc.wigner6j(2, 4, 6, 4, 2, 4),
                               np.sqrt(21)/105)
        self.assertAlmostEqual(wc.wigner6j(8*2, 10*2, 6*2, 8*2, 6*2, 4*2),
                               -635*np.sqrt(26)/176358)
        self.assertAlmostEqual(wc.wigner6j(8*2, 10*2, 6*2, 8*2, 9*2, 7*2),
                               -192529*np.sqrt(1430)/446185740)

    def test_racahw(self):
        self.assertAlmostEqual(wc.racah_w(8*2, 10*2, 6*2, 8*2, 6*2, 4*2),
                               -635*np.sqrt(26)/176358)

class TestWigner9j(unittest.TestCase):

    def test_9j(self):
        self.assertAlmostEqual(wc.wigner9j(2*1 ,2*2, 2*1, 2*2, 2*2, 2*2, 2*1, 2*2, 2*1),
                               -1.0/150)
        self.assertAlmostEqual(wc.wigner9j(2*1, 2*1, 2*1, 2*1, 2*1, 2*1, 2*1, 2*1, 2*0),
                               1.0/18)
        self.assertAlmostEqual(wc.wigner9j(2*1, 2*1, 2*1, 2*1, 2*1, 2*1, 2*1, 2*1, 2*1),
                               0.0)
        self.assertAlmostEqual(wc.wigner9j(2*3, 2*3, 2*2, 2*3, 2*3, 2*2, 2*3, 2*3, 2*2),
                               3221*np.sqrt(70)/(246960*np.sqrt(105)) - 365/(3528*np.sqrt(70)*np.sqrt(105)))

if __name__=='__main__':
    unittest.main()
