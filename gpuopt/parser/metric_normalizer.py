#!/usr/bin/env python3
"""
GPUOpt Metric Normalizer

Normalizes raw Nsight Compute metric names and values into structured schema objects.
"""

from typing import List, Dict, Any
from parser.models import KernelProfileData, DeviceInfo, NsightRuleAdvisory


class MetricNormalizer:
    """Normalizes raw metric key-value dictionaries into structured KernelProfileData models."""
    
    def normalize(self, parsed_csv_records: List[Dict[str, Any]]) -> List[KernelProfileData]:
        """
        Convert parsed CSV records into normalized KernelProfileData objects.
        """
        normalized_list = []
        
        for k_rec in parsed_csv_records:
            kname = k_rec.get('kernel_name', 'Unknown')
            raw_metrics = k_rec.get('metrics', {})
            
            device_info = DeviceInfo(
                name=f"Device {k_rec.get('device', '0')}",
                compute_capability=str(k_rec.get('cc', '0.0')),
                device_id=int(k_rec.get('device', 0)) if str(k_rec.get('device')).isdigit() else 0
            )
            
            exec_m = {}
            mem_m = {}
            comp_m = {}
            occ_m = {}
            flat_raw = {}
            
            for m_name, m_info in raw_metrics.items():
                val = m_info['val']
                flat_raw[m_name] = val
                
                # Check metric names (handles exact names and substring matches)
                lower_m = m_name.lower()
                
                # DRAM
                if 'dram__throughput' in lower_m or ('dram' in lower_m and 'throughput' in lower_m):
                    mem_m['dram_throughput_pct'] = float(val) if isinstance(val, (int, float)) else 0.0
                elif 'dram__bytes_read' in lower_m or ('dram' in lower_m and 'read' in lower_m):
                    mem_m['dram_bytes_read'] = float(val) if isinstance(val, (int, float)) else 0.0
                elif 'dram__bytes_write' in lower_m or ('dram' in lower_m and 'write' in lower_m):
                    mem_m['dram_bytes_write'] = float(val) if isinstance(val, (int, float)) else 0.0
                
                # L1 / L2 Cache
                elif 'l1tex__t_sector_hit_rate' in lower_m or 'l1' in lower_m and 'hit' in lower_m:
                    mem_m['l1_hit_rate_pct'] = float(val) if isinstance(val, (int, float)) else 0.0
                elif 'l2__cache_hit_rate' in lower_m or 'l2' in lower_m and 'hit' in lower_m:
                    mem_m['l2_hit_rate_pct'] = float(val) if isinstance(val, (int, float)) else 0.0
                elif 'l2' in lower_m and 'throughput' in lower_m:
                    mem_m['l2_throughput_pct'] = float(val) if isinstance(val, (int, float)) else 0.0
                
                # Global Memory Throughput
                elif 'memory__throughput' in lower_m or m_name == 'Memory Throughput':
                    mem_m['memory_throughput_pct'] = float(val) if isinstance(val, (int, float)) else 0.0
                
                # Shared Memory & Bank Conflicts
                elif 'shared__bank_conflicts' in lower_m or 'bank conflict' in lower_m:
                    mem_m['shared_bank_conflicts'] = float(val) if isinstance(val, (int, float)) else 0.0
                elif 'shared__load_throughput' in lower_m or ('shared' in lower_m and 'load' in lower_m):
                    mem_m['shared_load_throughput_pct'] = float(val) if isinstance(val, (int, float)) else 0.0
                elif 'shared__store_throughput' in lower_m or ('shared' in lower_m and 'store' in lower_m):
                    mem_m['shared_store_throughput_pct'] = float(val) if isinstance(val, (int, float)) else 0.0
                
                # Compute & SM
                elif 'sm__throughput' in lower_m or m_name == 'Compute (SM) Throughput':
                    comp_m['sm_throughput_pct'] = float(val) if isinstance(val, (int, float)) else 0.0
                elif 'sm active cycles' in lower_m or 'sm__active_cycles' in lower_m:
                    comp_m['sm_active_cycles'] = float(val) if isinstance(val, (int, float)) else 0.0
                
                # Occupancy
                elif 'achieved_occupancy' in lower_m or 'achieved occupancy' in lower_m:
                    occ_m['achieved_occupancy'] = float(val) if isinstance(val, (int, float)) else 0.0
                elif 'theoretical_occupancy' in lower_m or 'theoretical occupancy' in lower_m:
                    occ_m['theoretical_occupancy'] = float(val) if isinstance(val, (int, float)) else 0.0
                
                # Execution
                elif 'duration' in lower_m:
                    exec_m['duration_ns'] = float(val) if isinstance(val, (int, float)) else 0.0
                elif 'elapsed cycles' in lower_m or 'elapsed_cycles' in lower_m:
                    exec_m['elapsed_cycles'] = float(val) if isinstance(val, (int, float)) else 0.0
                elif 'registers per thread' in lower_m or 'registers_per_thread' in lower_m:
                    exec_m['registers_per_thread'] = int(val) if isinstance(val, (int, float)) else 0
                elif 'dynamic shared memory' in lower_m:
                    exec_m['dynamic_shared_memory'] = int(val) if isinstance(val, (int, float)) else 0
                elif 'static shared memory' in lower_m:
                    exec_m['static_shared_memory'] = int(val) if isinstance(val, (int, float)) else 0
            
            # Rules
            rules_list = []
            for r in k_rec.get('rules', []):
                rules_list.append(NsightRuleAdvisory(
                    rule_name=r.get('rule_name', ''),
                    rule_type=r.get('rule_type', ''),
                    description=r.get('description', ''),
                    speedup_type=r.get('speedup_type'),
                    estimated_speedup=r.get('estimated_speedup')
                ))
                
            kp = KernelProfileData(
                name=kname,
                demangled_name=kname.split('(')[0],
                device_info=device_info,
                block_size=k_rec.get('block_size', ''),
                grid_size=k_rec.get('grid_size', ''),
                execution_metrics=exec_m,
                memory_metrics=mem_m,
                compute_metrics=comp_m,
                occupancy_metrics=occ_m,
                raw_metrics=flat_raw,
                nsight_rules=rules_list
            )
            normalized_list.append(kp)
            
        return normalized_list
