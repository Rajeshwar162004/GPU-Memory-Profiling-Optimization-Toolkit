#!/usr/bin/env python3
"""
GPUOpt Rule Engine Unit Tests
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from parser.models import KernelProfileData, DeviceInfo
from analyzer.analyzer import Analyzer
from analyzer.bottleneck_classifier import BottleneckClassifier
from recommendations.engine import RecommendationEngine


class TestRules(unittest.TestCase):
    def test_bank_conflict_rule(self):
        k_data = KernelProfileData(
            name="transpose_shared",
            demangled_name="transpose_shared",
            device_info=DeviceInfo(name="Device 0", compute_capability="8.9"),
            memory_metrics={
                "shared_bank_conflicts": 128.0,
                "shared_load_throughput_pct": 85.0
            }
        )
        
        analyzer = Analyzer()
        classifier = BottleneckClassifier()
        rec_engine = RecommendationEngine()
        
        analysis = analyzer.analyze(k_data)
        self.assertGreater(analysis['issues_count'], 0)
        
        issue_ids = [i['id'] for i in analysis['issues']]
        self.assertIn("SHARED_MEMORY_BANK_CONFLICT", issue_ids)
        
        cls = classifier.classify(k_data, analysis)
        self.assertIn("MEMORY", cls['category'])
        
        recs = rec_engine.generate(analysis, cls)
        self.assertGreater(len(recs), 0)
        self.assertIn("Padding", recs[0]['title'])


if __name__ == '__main__':
    unittest.main()
