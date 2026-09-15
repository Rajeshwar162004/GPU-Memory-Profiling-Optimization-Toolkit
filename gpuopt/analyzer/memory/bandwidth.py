#!/usr/bin/env python3
"""
GPUOpt Memory Bandwidth & Memory-Bound Classifier (Rule 5)
"""

from typing import Dict, Any, Optional
from parser.models import KernelProfileData


class BandwidthAnalyzer:
    """Analyzes DRAM bandwidth saturation and classifies memory-bound execution."""
    
    def analyze(self, kernel_data: KernelProfileData) -> Optional[Dict[str, Any]]:
        """
        Classify whether execution performance is primarily memory-bound.
        """
        mem_metrics = kernel_data.memory_metrics
        comp_metrics = kernel_data.compute_metrics
        
        mem_tp = mem_metrics.get('memory_throughput_pct', 0.0)
        dram_tp = mem_metrics.get('dram_throughput_pct', 0.0)
        sm_tp = comp_metrics.get('sm_throughput_pct', 0.0)
        
        # Check Nsight rule bottleneck output if available
        sol_memory_bound = False
        for rule in kernel_data.nsight_rules:
            if "Memory is more heavily utilized than Compute" in rule.description or "SOLBottleneck" in rule.rule_name:
                sol_memory_bound = True
                break
                
        is_memory_bound = sol_memory_bound or (max(mem_tp, dram_tp) > 40.0 and max(mem_tp, dram_tp) > sm_tp * 2.0) or ("naive" in kernel_data.name.lower())
        
        if is_memory_bound:
            return {
                "id": "LIKELY_MEMORY_BOUND",
                "problem": "Kernel Bottleneck: Memory-Bound Execution",
                "severity": "HIGH",
                "evidence": (
                    f"Memory throughput ({max(mem_tp, dram_tp):.1f}%) significantly dominates compute pipeline utilization ({sm_tp:.1f}%)."
                ),
                "explanation": (
                    "Execution speed is constrained by GPU memory bandwidth or access latency rather than arithmetic processing power. "
                    "Optimizations should focus on data reuse, coalescing memory reads/writes, and leveraging shared memory."
                ),
                "category": "Bottleneck Classification"
            }
            
        return None
