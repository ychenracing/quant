from collections import Counter
import importlib, importlib.util, json
from pathlib import Path
import unittest


class FreshChallengerFormalValidationTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(
            importlib.util.find_spec('research.fresh_challenger_formal_validation'),
            'formal validation implementation is absent',
        )
        return importlib.import_module('research.fresh_challenger_formal_validation')

    def inputs(self):
        root=Path(__file__).parent
        catalog=json.loads((root/'catalog.json').read_text())
        protocol=json.loads((root/'protocol.json').read_text())
        full=list(catalog['pools']['union'])
        sectors=dict(catalog['sectors'])
        return catalog,protocol,full,sectors

    def test_fixed_plan_matches_preregistered_case_inventory(self):
        m=self.module(); catalog,protocol,full,sectors=self.inputs()
        plan=m.build_case_plan(full,sectors,catalog,protocol)
        self.assertEqual(len(plan),199)
        self.assertEqual(len({row['name'] for row in plan}),199)
        groups=Counter(row['group'] for row in plan)
        self.assertEqual(groups,Counter({
            'original_pool':17,
            'joint_removal':1,
            'single':34,
            'leave_one_out':34,
            'sector_removal':7,
            'sampled_subset':48,
            'common_five_exhaustive':31,
            'cardinality_coverage':24,
            'stress':3,
        }))
        self.assertEqual([row['index'] for row in plan],list(range(199)))
        self.assertEqual(plan,m.build_case_plan(full,sectors,catalog,protocol))
        self.assertEqual(m.plan_sha256(plan),m.plan_sha256(m.build_case_plan(full,sectors,catalog,protocol)))

    def test_shards_cover_every_case_once_without_reordering(self):
        m=self.module(); catalog,protocol,full,sectors=self.inputs(); plan=m.build_case_plan(full,sectors,catalog,protocol)
        shards=[m.shard_plan(plan,index,4) for index in range(4)]
        flattened=sorted((row['index'],index) for index,rows in enumerate(shards) for row in rows)
        self.assertEqual([case for case,_ in flattened],list(range(199)))
        self.assertEqual([case for case,index in flattened if case % 4 != index],[])
        self.assertEqual(sum(len(rows) for rows in shards),199)

    def test_original_pools_use_four_policies_and_other_cases_three(self):
        m=self.module(); catalog,protocol,full,sectors=self.inputs(); plan=m.build_case_plan(full,sectors,catalog,protocol)
        for row in plan:
            expected=['control','treatment','buy_hold','equal_weight'] if row['group']=='original_pool' else ['control','treatment','buy_hold']
            self.assertEqual(row['policies'],expected)
        self.assertEqual(sum(len(row['policies']) for row in plan),614)

    def test_no_numeric_candidate_neighborhood_is_synthesized(self):
        m=self.module(); catalog,protocol,full,sectors=self.inputs(); plan=m.build_case_plan(full,sectors,catalog,protocol)
        self.assertFalse(any(row['group']=='stability_not_reselection' for row in plan))
        self.assertEqual(sorted(row['name'] for row in plan if row['group']=='stress'),['delayed_open','double_cost','triple_cost'])


if __name__=='__main__': unittest.main()
