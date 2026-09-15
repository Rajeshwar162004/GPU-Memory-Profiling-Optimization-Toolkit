#!/usr/bin/env python3
"""
GPUOpt Cache Utilization Analyzer (Rule 4)
"""

from typing import Dict, Any, Optional
from parser.models import KernelProfileData


class CacheAnalyzer:
    """Analyzes L1/L2 cache hit rate and cache-related inefficiencies."""
    
    def analyze(self, kernel_data: KernelProfileData) -> Optional[Dict[str, Any]]:
        """
        Detect low L2 cache hit rate or poor temporal/spatial locality.
        """
        mem_metrics = kernel_data.memory_metrics
        l2_hit_rate = mem_metrics.get('l2_hit_rate_pct')
        l1_hit_rate = mem_metrics.get('l1_hit_rate_pct')
        
        if l2_hit_rate is not None and l2_hit_rate < 50.0 and l2_hit_rate > 0.0:
            return {
                "id": "POOR_CACHE_BEHAVIOR",
                "problem": "Low L2 Cache Hit Rate / Reuse Behavior",
                "severity": "WARNING",
                "evidence": f"Measured L2 cache hit rate is only {l2_hit_rate:.1f}%, leading to frequent DRAM accesses.",
                "explanation": (
                    "Low cache reuse forces memory transactions to miss the L2 cache and travel all the way to DRAM. "
                    "This usually stems from strided accesses exceeding cache line limits, large working set sizes, or inefficient memory layout."
                ),
                "category": "Cache"
            }
            
        return None
