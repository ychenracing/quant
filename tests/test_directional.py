"""Critical invariants for causal drawdown exits, rebounds and persistent units."""
import unittest
import numpy as np
from research.directional import directional_state, allocate

class DirectionalTest(unittest.TestCase):
    def test_peak_exit_and_fresh_reversal(self):
        c=np.array([100,105,110,120,109,108,114],float)[:,None]
        ready=np.ones_like(c,bool);initial=np.ones_like(c,bool);confirm=np.ones_like(c,bool)
        actual=directional_state(c,ready,initial,confirm,.08,.04)[:,0]
        np.testing.assert_array_equal(actual,[True,True,True,True,False,False,True])
    def test_prefix_invariance(self):
        c=np.array([100,105,110,120,109,108,114],float)[:,None]
        b=np.ones_like(c,bool)
        for end in range(1,len(c)):
            np.testing.assert_array_equal(directional_state(c,b,b,b,.08,.04)[:end],
                directional_state(c[:end],b[:end],b[:end],b[:end],.08,.04))
    def test_invalid_quote_cannot_enter(self):
        c=np.array([100,110,120],float)[:,None];b=np.ones_like(c,bool)
        ready=b.copy();ready[1]=False
        self.assertFalse(directional_state(c,ready,b,b,.08,.04)[1,0])
    def test_units_are_retained_and_cash_is_reused(self):
        np.testing.assert_allclose(allocate([.55,.449],[True,True],[True,True],[1,2],2,True),[.55,.449])
        np.testing.assert_allclose(allocate([.3,0,0],[True]*3,[True]*3,[3,2,1],2,False),[.6,.4,0])
    def test_broken_holding_exits_without_schedule(self):
        np.testing.assert_allclose(allocate([.7,0],[False,True],[False,False],[2,1],2,False),[0,0])
    def test_single_name_can_invest_without_leverage(self):
        np.testing.assert_allclose(allocate([0],[True],[True],[2],2,False),[1])
        with self.assertRaises(ValueError):allocate([1.1],[True],[True],[2],2,False)

if __name__=='__main__':unittest.main()
