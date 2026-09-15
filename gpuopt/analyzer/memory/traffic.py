#!/usr/bin/env python3
"""
GPUOpt Global Memory Traffic & Data Reuse Analyzer (Rule 3)
"""

from typing import Dict, Any, Optional
from parser.models import KernelProfileData


class MemoryTrafficAnalyzer:
    """Analyzes excessive global memory traffic and data reuse opportunities."""
    
    def analyze(self, kernel_data: KernelProfileData) -> Optional[Dict[str, Any]]:
        """
        Detect disproportionately high global memory traffic caused by lack of data reuse.
        """
        mem_metrics = kernel_data.memory_metrics
        comp_metrics = kernel_data.compute_metrics
        
        dram_bytes_read = mem_metrics.get('dram_bytes_read', 0.0)
        dram_bytes_write = mem_metrics.get('dram_bytes_write', 0.0)
        dram_tp = mem_metrics.get('dram_throughput_pct', 0.0)
        sm_tp = comp_metrics.get('sm_throughput_pct', 0.0)
        
        # Check if memory traffic is high relative to compute activity
        if dram_tp > 50.0 and sm_tp < 25.0:
            total_bytes = dram_bytes_read + dram_bytes_write
            bytes_str = f" ({total_bytes / 1e6:.2f} MB)" if total_bytes > 0 else ""
            
            return {
                "id": "EXCESSIVE_GLOBAL_MEMORY_TRAFFIC",
                "problem": "Excessive Global Memory Traffic",
                "severity": "WARNING",
                "evidence": (
                    f"DRAM throughput is high ({dram_tp:.1f}% of peak sustained){bytes_str} while compute utilization "
                    f"remains relatively low ({sm_tp:.1f}%)."
                ),
                "explanation": (
                    "The kernel is repeatedly loading matrix or array elements directly from high-latency DRAM "
                    "without reusing previously fetched data. Utilizing shared memory tiling or register caching can drastically reduce redundant DRAM requests."
                ),
                "category": "Memory Traffic"
            }
            
        return None
