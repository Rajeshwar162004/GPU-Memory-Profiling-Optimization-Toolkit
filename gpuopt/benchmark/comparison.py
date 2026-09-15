#!/usr/bin/env python3
"""
GPUOpt Comparison Engine

Compares before and after benchmark and profiling results.
"""

from typing import Dict, Any, List


class ComparisonEngine:
    """Compares baseline and optimized performance metrics and execution timing."""
    
    def compare(self, before_data: Any, after_data: Any) -> Dict[str, Any]:
        """
        Compare two benchmark or analysis result structures.
        """
        # Extract execution times
        before_time = self._extract_time(before_data)
        after_time = self._extract_time(after_data)
        
        speedup = (before_time / after_time) if after_time > 0 else 1.0
        improvement_pct = ((before_time - after_time) / before_time * 100.0) if before_time > 0 else 0.0
        
        # Compare key metrics if present
        metric_diffs = {}
        before_metrics = self._extract_metrics(before_data)
        after_metrics = self._extract_metrics(after_data)
        
        for k, v1 in before_metrics.items():
            if k in after_metrics:
                v2 = after_metrics[k]
                change_pct = ((v2 - v1) / v1 * 100.0) if v1 != 0 else 0.0
                metric_diffs[k] = {
                    "before": float(v1),
                    "after": float(v2),
                    "change_pct": float(change_pct)
                }
                
        return {
            "before_time_ms": before_time,
            "after_time_ms": after_time,
            "speedup": float(speedup),
            "improvement_pct": float(improvement_pct),
            "metrics": metric_diffs,
            "validated": speedup > 1.05
        }
        
    @staticmethod
    def _extract_time(data: Any) -> float:
        if isinstance(data, dict):
            if 'median_ms' in data:
                return float(data['median_ms'])
            if 'mean_ms' in data:
                return float(data['mean_ms'])
            if isinstance(data.get('execution'), dict):
                return float(data['execution'].get('duration_ns', 0) / 1e6 or data['execution'].get('elapsed_cycles', 0))
            if isinstance(data, list) and len(data) > 0:
                return ComparisonEngine._extract_time(data[0])
        return 1.0

    @staticmethod
    def _extract_metrics(data: Any) -> Dict[str, float]:
        res = {}
        if isinstance(data, dict):
            if 'memory' in data and isinstance(data['memory'], dict):
                for k, v in data['memory'].items():
                    if isinstance(v, (int, float)):
                        res[k] = float(v)
            if 'execution' in data and isinstance(data['execution'], dict):
                for k, v in data['execution'].items():
                    if isinstance(v, (int, float)):
                        res[k] = float(v)
        elif isinstance(data, list) and len(data) > 0:
            return ComparisonEngine._extract_metrics(data[0])
        return res
