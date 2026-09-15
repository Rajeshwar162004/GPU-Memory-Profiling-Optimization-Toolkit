#!/usr/bin/env python3
"""
GPUOpt Global Memory Coalescing Analyzer (Rule 1)
"""

from typing import Dict, Any, Optional
from parser.models import KernelProfileData


class CoalescingAnalyzer:
    """Analyzes global memory access patterns and coalescing efficiency."""
    
    def analyze(self, kernel_data: KernelProfileData) -> Optional[Dict[str, Any]]:
        """
        Check for uncoalesced or inefficient global memory access patterns based on profiler evidence.
        """
        mem_metrics = kernel_data.memory_metrics
        raw_metrics = kernel_data.raw_metrics
        
        # Check Nsight rule alerts for memory replay or coalescing
        coalescing_alert = False
        rule_evidence = ""
        for rule in kernel_data.nsight_rules:
            desc_lower = rule.description.lower()
            if "coalescing" in desc_lower or "memory replay" in desc_lower or "uncoalesced" in desc_lower:
                coalescing_alert = True
                rule_evidence = rule.description
                break
                
        mem_tp = mem_metrics.get('memory_throughput_pct', 0.0)
        dram_tp = mem_metrics.get('dram_throughput_pct', 0.0)
        l2_hit = mem_metrics.get('l2_hit_rate_pct', 100.0)
        
        # Infer uncoalesced pattern if high L2/Memory throughput occurs with low DRAM efficiency or explicit rule alerts
        if coalescing_alert or (mem_tp > 40.0 and dram_tp < 20.0 and l2_hit < 60.0):
            evidence = rule_evidence if rule_evidence else (
                f"High L2 cache/Memory utilization ({mem_tp:.1f}%) coupled with low DRAM effective throughput ({dram_tp:.1f}%) "
                f"and L2 hit rate ({l2_hit:.1f}%). Extra memory transactions are likely being issued per warp access."
            )
            return {
                "id": "UNCOALESCED_GLOBAL_ACCESS",
                "problem": "Potentially Inefficient / Uncoalesced Global Memory Access",
                "severity": "HIGH" if coalescing_alert else "WARNING",
                "evidence": evidence,
                "explanation": (
                    "Threads in a warp are accessing global memory addresses with large strides or non-contiguous layouts. "
                    "This forces the GPU memory controller to issue multiple 32-byte cache line transactions for a single warp instruction, "
                    "wasting memory bandwidth."
                ),
                "category": "Global Memory"
            }
            
        return None
