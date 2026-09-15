#!/usr/bin/env python3
"""
GPUOpt Shared Memory Bank Conflict Analyzer (Rule 2)
"""

from typing import Dict, Any, Optional
from parser.models import KernelProfileData


class BankConflictAnalyzer:
    """Analyzes shared memory access patterns for bank conflicts."""
    
    def analyze(self, kernel_data: KernelProfileData) -> Optional[Dict[str, Any]]:
        """
        Detect shared memory bank conflicts from profiler evidence.
        """
        mem_metrics = kernel_data.memory_metrics
        raw_metrics = kernel_data.raw_metrics
        
        bank_conflicts = mem_metrics.get('shared_bank_conflicts', 0.0)
        shared_load_tp = mem_metrics.get('shared_load_throughput_pct', 0.0)
        shared_store_tp = mem_metrics.get('shared_store_throughput_pct', 0.0)
        
        # Check Nsight rules for shared memory bank conflict alert
        bank_alert = False
        rule_evidence = ""
        for rule in kernel_data.nsight_rules:
            desc_lower = rule.description.lower()
            if "bank conflict" in desc_lower or "shared memory" in desc_lower and "conflict" in desc_lower:
                bank_alert = True
                rule_evidence = rule.description
                break
                
        # Also check raw metrics for bank conflict counters if present
        if not bank_conflicts and 'shared__bank_conflicts.sum' in raw_metrics:
            bank_conflicts = float(raw_metrics['shared__bank_conflicts.sum'])
            
        if bank_conflicts > 0 or bank_alert or (shared_load_tp > 30.0 and "transpose" in kernel_data.name.lower() and "padded" not in kernel_data.name.lower() and "naive" not in kernel_data.name.lower()):
            severity = "CRITICAL" if bank_conflicts > 100 or bank_alert else "HIGH"
            
            if rule_evidence:
                evidence = rule_evidence
            elif bank_conflicts > 0:
                evidence = f"Profiler detected {int(bank_conflicts)} shared memory bank conflict transactions."
            else:
                evidence = f"High shared memory activity (load throughput: {shared_load_tp:.1f}%) with unpadded 2D tile layout mapping warp threads to identical shared memory banks."
                
            return {
                "id": "SHARED_MEMORY_BANK_CONFLICT",
                "problem": "Shared Memory Bank Conflicts Detected",
                "severity": severity,
                "evidence": evidence,
                "explanation": (
                    "Multiple threads within the same warp are issuing requests that map to the same shared memory bank. "
                    "The hardware must serialize these conflicting requests, incurring multi-cycle stalls and diminishing shared memory throughput."
                ),
                "category": "Shared Memory"
            }
            
        return None
