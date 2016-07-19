from unittest import TestCase

import py21cmsense as py21cm
import numpy as np
import os

class TestUtils(TestCase):

    def setUp(self):
        self.path = os.path.dirname(os.path.realpath(__file__))
        pass

    def tearDown(self):
        pass

    def test_load_no_files(self):
        out_f,_,_ = py21cm.utils.load_noise_files(None)
        self.assertEqual(0,out_f)

    def test_load_empty_list(self):
        out_f,_,_ = py21cm.utils.load_noise_files([])
        self.assertEqual(0,out_f)

    def test_load_ks(self):
        test_file = 'test_data/test_load_k_0.114.npz'
        ref_ks = np.linspace(0,1)*.5 + .01
        _,out_k,_ = py21cm.utils.load_noise_files(
                os.path.join(self.path,test_file))
        self.assertTrue(np.allclose(ref_ks,out_k))

    def test_load_freq(self):
        test_file = 'test_data/test_load_k_0.114.npz'
        ref_freq=114
        out_freq,_,_ = py21cm.utils.load_noise_files(
                os.path.join(self.path,test_file))
        self.assertEqual(ref_freq,out_freq)

    def test_load_noise(self):
        test_file= 'test_data/test_load_k_0.114.npz'
        ref_noise = [ 1.753, 1.54011634,  1.36467456,  1.22454969,
                1.11761678,  1.04175086,  0.99482696 , 0.97472012,
                0.97930537,  1.00645774,  1.05405228,  1.11996402,
                1.20206799,  1.29823923,  1.40635277,  1.52428365,
                1.6499069,   1.78109756, 1.91573066,  2.05168125,
                2.18682434,  2.31903499,  2.44618821 , 2.56615906,
                2.67682256,  2.77605374,  2.86172766,  2.93171933,
                2.98390379,  3.01615608, 3.02635124,  3.0123643,
                2.97207029,  2.90334424,  2.80406121,  2.67209621,
                2.50532429,  2.30162047,  2.0588598,   1.77491731,
                1.44766804,  1.07498701,  0.65474927,  0.18482985,
                -0.33689622, -0.91255389, -1.54426815, -2.23416394,
                -2.98436623, -3.797     ]
        _,_,out_noise = py21cm.utils.load_noise_files(
                os.path.join(self.path,test_file))
        self.assertTrue(np.allclose(ref_noise,out_noise))

class TestInterp(TestCase):

    def setup(self):
        pass

    def tearDown(self):
        pass

    def test_no_freqs(self):
        out = py21cm.utils.noise_interp2d(None,[.1,.2,.3],[1,2,3])
        self.assertEqual(0,out)

    def test_no_ks(self):
        out = py21cm.utils.noise_interp2d(114,None,[1,2,3])
        self.assertEqual(0,out)

    def test_no_noises(self):
        out = py21cm.utils.noise_interp2d(114,[.1,.2,.3],None)
        self.assertEqual(0,out)

    def test_one_ks(self):
        with self.assertRaises(ValueError):
            py21cm.utils.noise_interp2d(114,[.1],[1])

    def test_two_ks(self):
        with self.assertRaises(ValueError):
            py21cm.utils.noise_interp2d(114,[.1,.1],[1,2])

    def test_two_one_ks(self):
        out=py21cm.utils.noise_interp2d(114,[[.1],[.1]],[[1],[2]])
        self.assertEqual(0,out)

    def test_interp_linear(self):
        out_interp=py21cm.utils.noise_interp2d([114,115],[[.1,.2],[.1,.2]],[[1,1],[2,2]])
        ref_num = 1.5
        out_num = out_interp(.15,114.5)
        self.assertEqual(ref_num,out_num)

if __name__ == '__main__':
        unittest.main()
