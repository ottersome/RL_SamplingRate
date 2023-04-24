# This File is For Testing all Sampling Methods 
import unittest
import sys,os
import numpy as np
pardir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
print(os.getcwd())
sys.path.append(os.getcwd())
from sp_sims.statistics.statistics import quick_sample,simple_sample
from sp_sims.simulators.stochasticprocesses import RaceOfExponentials


class SamplingTests(unittest.TestCase):

    def __init__(self, algo, *args):
        # Algo will just by a  function for now 
        super().__init__(*args)
        self.algo = algo

    def setUp(self):
        assert self.algo != None, print('You must provide a sampling algorithm')

    def test_1(self):
        states = [1,0,1]
        times = [1, 1.1,2]
        # 0,1,2.1,4.1
        rate = 1/0.75
        result = [1,1,0,1,1,1]

        samples = self.algo(rate, states, times)
        for i in range(len(result)):
            assert result[i] == samples[i]

    def test_2(self):
        states = [0,1,0,1,0]
        times = [1,1,0.1,0.1,1]
        rate = 1/0.75
        result = [0,0,1,0,0]

        samples = self.algo(rate, states, times)

        for i in range(len(result)):
            assert result[i] == samples[i]
    def test_limit(self):
        states = [0,1]
        times = [1,1]
        rate = 1/0.25
        result = [0,0,0,0,0,1,1,1,1]

        samples = self.algo(rate, states, times)
        for i in range(len(result)):
            assert result[i] == samples[i]

    # Just for extra safety
    def test_random(self):
        rates = {"lam": 1,"mu":1.5}
        sp_rates = np.logspace(-3,6, 10, base=2)
        for rate in sp_rates:
            print("Testing with samp_rate ", rate)
            for i in range(100):
                roe = RaceOfExponentials(1000,list(rates.values()),state_limit=1)
                holdTimes_tape, state_tape = roe.generate_history(0)
                sample_quick = quick_sample(rate, state_tape, holdTimes_tape)
                simp_sample = simple_sample(rate, state_tape, holdTimes_tape)

                for i in range(len(sample_quick)):
                    assert sample_quick[i] == simp_sample[i]

    def test_random_limited(self):
        rates = {"lam": 1,"mu":1.5}
        sp_rates = np.logspace(-3,6, 10, base=2)
        for rate in sp_rates:
            print("Testing with samp_rate ", rate)
            for i in range(100):
                roe = RaceOfExponentials(1000,list(rates.values()),state_limit=1)
                holdTimes_tape, state_tape = roe.generate_history(0)
                sample_quick = quick_sample(rate, state_tape, holdTimes_tape)
                simp_sample = simple_sample(rate, state_tape, holdTimes_tape)

                for i in range(len(sample_quick)):
                    assert sample_quick[i] == simp_sample[i]


def suite():
    suite = unittest.TestSuite()
    suite.addTest(SamplingTests(simple_sample,'test_1'))
    suite.addTest(SamplingTests(simple_sample,'test_2'))
    suite.addTest(SamplingTests(simple_sample,'test_limit'))
    suite.addTest(SamplingTests(quick_sample,'test_limit'))
    suite.addTest(SamplingTests(simple_sample,'test_random'))
    return suite
if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
