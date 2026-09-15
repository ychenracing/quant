"""Joint funding must conserve every resource and original executable minimum."""
import unittest
import numpy as np
from research.joint_resources import progressive_funding, allocate_additions


class JointResourceTests(unittest.TestCase):
    def test_risk_direction_shares_resource_without_sequential_preemption(self):
        a=np.array([[1.,1.],[.1,.2],[1.,0.],[0.,1.]])
        result=progressive_funding(a,[1000.,100.,1000.,1000.],[2./.1,1./.2])
        np.testing.assert_allclose(result,[2000./3,500./3],atol=1e-10)
        np.testing.assert_allclose(a@result,[2500./3,100.,2000./3,500./3],atol=1e-10)

    def test_binding_symbol_redistributes_only_remaining_resource(self):
        a=np.array([[1.,1.],[1.,0.],[0.,1.]])
        result=progressive_funding(a,[100.,20.,100.],[2.,1.])
        np.testing.assert_allclose(result,[20.,80.],atol=1e-12)

    def test_sector_freeze_leaves_other_sector_fundable(self):
        a=np.array([[1.,1.,1.],[1.,1.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
        result=progressive_funding(a,[100.,40.,100.,100.,100.],[2.,1.,1.])
        np.testing.assert_allclose(result,[80./3,40./3,60.],atol=1e-12)

    def test_permutation_and_resource_units_do_not_select_winners(self):
        a=np.array([[1.,1.,1.],[.1,.2,.3],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
        b=np.array([100.,15.,25.,50.,100.]);w=np.array([2.,1.,3.]);p=[2,0,1]
        result=progressive_funding(a,b,w)
        changed=progressive_funding(a[:,p]*np.array([2,5,3,9,7])[:,None],b*np.array([2,5,3,9,7]),w[p])
        np.testing.assert_allclose(changed,result[p],rtol=1e-12,atol=1e-12)

    def inputs(self):
        return dict(scores=np.array([4.,3.,2.,1.]),density=np.full(4,.1),
                    headroom=np.full(4,100.),groups=('a','a','b','b'),
                    sector_room={'a':100.,'b':100.},held=np.zeros(4,dtype=bool),
                    eligible=np.ones(4,dtype=bool),symbols=('a','b','c','d'),
                    cash=100.,risk=10.,slots=2,nav=100.,held_band=.08)

    def test_original_priority_slots_and_materiality(self):
        kw=self.inputs();result=allocate_additions(**kw)
        self.assertEqual(set(np.flatnonzero(result)),{0,1})
        np.testing.assert_allclose(result[:2],[400./7,300./7])
        kw['headroom'][1]=.2
        result=allocate_additions(**kw)
        self.assertEqual(set(np.flatnonzero(result)),{0,2})
        self.assertTrue(np.all(result[result>0]>=1.))

    def test_no_held_addition_without_original_eligibility(self):
        kw=self.inputs();kw['held'][0]=True;kw['eligible'][0]=False;kw['slots']=1
        result=allocate_additions(**kw)
        self.assertEqual(set(np.flatnonzero(result)),{1})
        kw['slots']=0
        np.testing.assert_array_equal(allocate_additions(**kw),np.zeros(4))

    def test_small_held_leg_dropped_without_stealing_new_slot(self):
        kw=self.inputs();kw['held'][0]=True;kw['scores'][0]=.01;kw['slots']=1
        result=allocate_additions(**kw)
        self.assertEqual(set(np.flatnonzero(result)),{1})
        self.assertEqual(result.sum(),100.)

    def test_zero_resources_empty_and_invalid_are_explicit(self):
        kw=self.inputs()
        for field in ('cash','risk'):
            changed=kw.copy();changed[field]=0.
            np.testing.assert_array_equal(allocate_additions(**changed),np.zeros(4))
        kw['eligible'][:]=False
        np.testing.assert_array_equal(allocate_additions(**kw),np.zeros(4))
        for a,b,w in (([[1]],[-1],[1]),([[0]],[1],[1]),([[1]],[1],[0]),
                      ([[float('nan')]],[1],[1]),([[1,1]],[1],[1])):
            with self.assertRaises(ValueError):progressive_funding(a,b,w)
        kw=self.inputs();kw['slots']=True
        with self.assertRaises(ValueError):allocate_additions(**kw)

    def test_many_resources_are_conserved_without_mutating_inputs(self):
        rng=np.random.default_rng(19)
        for _ in range(80):
            a=np.vstack([np.ones(5),rng.random((3,5)),np.eye(5)])
            b=rng.random(len(a))*100.;w=rng.random(5)+.01;old=a.copy()
            value=progressive_funding(a,b,w)
            self.assertTrue(np.all(a@value<=b+1e-9));self.assertTrue(np.all(value>=0))
            np.testing.assert_array_equal(a,old)
            for j in range(5):
                self.assertTrue(np.any((a[:,j]>0)&(b-a@value<1e-8)))


if __name__=='__main__':unittest.main()
