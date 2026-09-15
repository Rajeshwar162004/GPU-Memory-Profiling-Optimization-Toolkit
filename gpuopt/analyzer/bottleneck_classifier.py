#!/usr/bin/env python3
"""
GPUOpt Bottleneck Classifier

Categorizes kernels into Memory-Bound, Compute-Bound, Latency-Bound, or Balanced.
"""

from typing import Dict, Any
from parser.models import KernelProfileData


class BottleneckClassifier:
    """Classifies kernel performance bottleneck based on normalized metrics and issue evidence."""
    
    def classify(self, kernel_data: KernelProfileData, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify overall kernel performance bottleneck.
        
        Returns:
            Dict containing category, confidence level, and rationale.
        """
        mem_metrics = kernel_data.memory_metrics
        comp_metrics = kernel_data.compute_metrics
        issues = analysis_result.get('issues', [])
        
        mem_tp = mem_metrics.get('memory_throughput_pct', 0.0)
        dram_tp = mem_metrics.get('dram_throughput_pct', 0.0)
        sm_tp = comp_metrics.get('sm_throughput_pct', 0.0)
        
        has_memory_issues = any(i['category'] in ['Global Memory', 'Shared Memory', 'Memory Traffic', 'Cache'] for i in issues)
        
        if (max(mem_tp, dram_tp) > sm_tp and max(mem_tp, dram_tp) > 30.0) or has_memory_issues:
            category = "🔥 MEMORY-LIMITED / MEMORY-INEFFICIENT"
            confidence = "HIGH"
            rationale = "Memory pipeline throughput or memory access inefficiencies (coalescing, bank conflicts, low cache reuse) dominate execution time."
        elif sm_tp > max(mem_tp, dram_tp) and sm_tp > 60.0:
            category = "⚡ COMPUTE-BOUND"
            confidence = "HIGH"
            rationale = "ALU/FPU compute execution pipelines are fully saturated."
        elif max(mem_tp, dram_tp) < 20.0 and sm_tp < 20.0:
            category = "⏱ LATENCY / OCCUPANCY BOUND"
            confidence = "MEDIUM"
            rationale = "Low throughput across both compute and memory pipelines suggests instruction latency or thread stall bottlenecks."
        else:
            category = "BALANCED / NO MAJOR BOTTLENECK DETECTED"
            confidence = "HIGH"
            rationale = "Execution displays balanced resource utilization across memory and compute pipelines."
            
        return {
            "category": category,
            "confidence": confidence,
            "rationale": rationale
        }
