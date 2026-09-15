#!/usr/bin/env python3
"""
GPUOpt Analyzer Engine

Executes memory analysis rules against normalized kernel profiling data.
"""

from typing import Dict, Any, List
from parser.models import KernelProfileData
from analyzer.memory.coalescing import CoalescingAnalyzer
from analyzer.memory.bank_conflicts import BankConflictAnalyzer
from analyzer.memory.traffic import MemoryTrafficAnalyzer
from analyzer.memory.cache import CacheAnalyzer
from analyzer.memory.bandwidth import BandwidthAnalyzer


class Analyzer:
    """Main diagnostic analysis engine for GPU memory behavior."""
    
    def __init__(self):
        self.coalescing_analyzer = CoalescingAnalyzer()
        self.bank_conflict_analyzer = BankConflictAnalyzer()
        self.traffic_analyzer = MemoryTrafficAnalyzer()
        self.cache_analyzer = CacheAnalyzer()
        self.bandwidth_analyzer = BandwidthAnalyzer()
        
    def analyze(self, kernel_data: KernelProfileData) -> Dict[str, Any]:
        """
        Run rule engine across kernel data and return detected issues and diagnostic metrics.
        """
        issues: List[Dict[str, Any]] = []
        
        # Rule 1: Coalescing
        issue1 = self.coalescing_analyzer.analyze(kernel_data)
        if issue1:
            issues.append(issue1)
            
        # Rule 2: Shared Memory Bank Conflicts
        issue2 = self.bank_conflict_analyzer.analyze(kernel_data)
        if issue2:
            issues.append(issue2)
            
        # Rule 3: Excessive Memory Traffic
        issue3 = self.traffic_analyzer.analyze(kernel_data)
        if issue3:
            issues.append(issue3)
            
        # Rule 4: Cache Behavior
        issue4 = self.cache_analyzer.analyze(kernel_data)
        if issue4:
            issues.append(issue4)
            
        # Rule 5: Memory Bandwidth / Memory Bound
        issue5 = self.bandwidth_analyzer.analyze(kernel_data)
        if issue5:
            issues.append(issue5)
            
        return {
            "kernel_name": kernel_data.name,
            "issues_count": len(issues),
            "issues": issues,
            "device": kernel_data.device_info.name
        }
