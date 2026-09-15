#!/usr/bin/env python3
"""
GPUOpt Parser Unit Tests
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from parser.csv_parser import CSVParser
from parser.metric_normalizer import MetricNormalizer

SAMPLE_NCU_CSV = """
"ID","Process ID","Process Name","Host Name","Kernel Name","Context","Stream","Block Size","Grid Size","Device","CC","Section Name","Metric Name","Metric Unit","Metric Value","Rule Name","Rule Type","Rule Description","Estimated Speedup Type","Estimated Speedup"
"0","1234","matmul","127.0.0.1","matmul_kernel(const float *, const float *, float *, int)","1","7","(32, 32, 1)","(32, 32, 1)","0","8.9","GPU Speed Of Light Throughput","Memory Throughput","%","78.50","","","","",""
"0","1234","matmul","127.0.0.1","matmul_kernel(const float *, const float *, float *, int)","1","7","(32, 32, 1)","(32, 32, 1)","0","8.9","GPU Speed Of Light Throughput","Compute (SM) Throughput","%","12.30","","","","",""
"0","1234","matmul","127.0.0.1","matmul_kernel(const float *, const float *, float *, int)","1","7","(32, 32, 1)","(32, 32, 1)","0","8.9","SpeedOfLight","","","","SOLBottleneck","OPT","Memory is more heavily utilized than Compute","",""
"""

class TestParser(unittest.TestCase):
    def test_csv_parser(self):
        parser = CSVParser()
        records = parser.parse(SAMPLE_NCU_CSV)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['kernel_name'], 'matmul_kernel(const float *, const float *, float *, int)')
        self.assertIn('Memory Throughput', records[0]['metrics'])

    def test_metric_normalizer(self):
        parser = CSVParser()
        normalizer = MetricNormalizer()
        records = parser.parse(SAMPLE_NCU_CSV)
        normalized = normalizer.normalize(records)
        self.assertEqual(len(normalized), 1)
        k_data = normalized[0]
        self.assertEqual(k_data.memory_metrics.get('memory_throughput_pct'), 78.50)
        self.assertEqual(k_data.compute_metrics.get('sm_throughput_pct'), 12.30)
        self.assertEqual(len(k_data.nsight_rules), 1)

if __name__ == '__main__':
    unittest.main()
