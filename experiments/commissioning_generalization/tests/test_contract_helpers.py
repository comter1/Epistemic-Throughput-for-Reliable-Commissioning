"""Executable tests of handoff utilities only, not of the not-yet-built main experiment."""
import importlib.util
from pathlib import Path
import struct
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('cgx_handoff_contract',ROOT/'contracts/interfaces.py')
mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)

class ContractTests(unittest.TestCase):
    def test_wire_sizes(self):
        self.assertEqual(struct.calcsize('<d16sIBBH14f40x'),128)
        self.assertEqual(struct.calcsize('<16d4Q'),160)
    def test_budget_dimensions_are_separate(self):
        b=mod.ResourceVector(executions=4,bytes=1000,calibrations=2,decisions=6)
        after=b.subtract(mod.ResourceVector(bytes=160,calibrations=1,decisions=1))
        self.assertEqual(after.executions,4)
        self.assertEqual(after.calibrations,1)
        self.assertEqual(after.bytes,840)
        self.assertEqual(after.decisions,5)
    def test_pathwise_overrun_rejected(self):
        b=mod.ResourceVector(executions=1,bytes=128,calibrations=0,decisions=1)
        with self.assertRaises(ValueError):
            b.subtract(mod.ResourceVector(bytes=129))
        with self.assertRaises(ValueError):
            b.subtract(mod.ResourceVector(calibrations=1))
    def test_negative_resources_rejected(self):
        with self.assertRaises(ValueError): mod.ResourceVector(decisions=-1)
    def test_public_types_exclude_truth(self):
        fields=set(mod.PublicContext.__dataclass_fields__)
        self.assertFalse({'label','fault_node','true_offset','private_case'} & fields)

if __name__=='__main__': unittest.main()
