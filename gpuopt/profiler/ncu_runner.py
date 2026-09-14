#!/usr/bin/env python3
"""
GPUOpt NCU Runner

Wrapper around NVIDIA Nsight Compute CLI for profiling CUDA applications.
"""

import subprocess
import os
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import shlex


@dataclass
class ProfileResult:
    """Result of a profiling run."""
    success: bool
    output_file: Optional[str] = None
    error: Optional[str] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    return_code: int = 0


class NCURunner:
    """Runs NVIDIA Nsight Compute (ncu) to profile CUDA applications."""
    
    # Default metrics for memory analysis
    DEFAULT_MEMORY_METRICS = [
        # Global Memory
        'dram__throughput.avg.pct_of_peak_sustained_elapsed',
        'dram__bytes_read.sum',
        'dram__bytes_write.sum',
        'l1tex__t_sector_hit_rate.pct',
        'l1tex__t_sector_miss_rate.pct',
        'l2__slice_read_throughput.avg.pct_of_peak_sustained_elapsed',
        'l2__slice_write_throughput.avg.pct_of_peak_sustained_elapsed',
        'l2__cache_hit_rate.pct',
        
        # Memory Throughput
        'memory__throughput.avg.pct_of_peak_sustained_elapsed',
        'memory__bytes_read.sum',
        'memory__bytes_write.sum',
        
        # Shared Memory
        'shared__load_throughput.avg.pct_of_peak_sustained_elapsed',
        'shared__store_throughput.avg.pct_of_peak_sustained_elapsed',
        'shared__bank_conflicts.sum',
        
        # Compute
        'sm__throughput.avg.pct_of_peak_sustained_elapsed',
        'sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_elapsed',
        'sm__inst_executed_pipe_fp32.avg.pct_of_peak_sustained_elapsed',
        'sm__inst_executed_pipe_fp64.avg.pct_of_peak_sustained_elapsed',
        'sm__inst_executed_pipe_int.avg.pct_of_peak_sustained_elapsed',
        
        # Occupancy
        'achieved_occupancy',
        'theoretical_occupancy',
        
        # Execution
        'elapsed_cycles',
        'duration',
        'grid_size',
        'block_size',
        'registers_per_thread',
        'dynamic_shared_memory_per_block',
        'static_shared_memory_per_block',
    ]
    
    def __init__(self, ncu_path: str = 'ncu'):
        """
        Initialize NCU runner.
        
        Args:
            ncu_path: Path to ncu executable (default: 'ncu' from PATH)
        """
        self.ncu_path = ncu_path
        self._verify_ncu()
    
    def _verify_ncu(self) -> None:
        """Verify ncu is available."""
        try:
            result = subprocess.run([self.ncu_path, '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                raise RuntimeError(f"ncu not found or failed: {result.stderr}")
        except FileNotFoundError:
            raise RuntimeError(f"ncu not found in PATH. Please install NVIDIA Nsight Compute.")
        except subprocess.TimeoutExpired:
            raise RuntimeError("ncu --version timed out")
    
    def profile(self, 
                executable: str,
                output: str,
                metrics: Optional[str] = None,
                kernel: Optional[str] = None,
                replay_mode: str = 'kernel',
                target_processes: str = 'all',
                additional_args: Optional[List[str]] = None) -> ProfileResult:
        """
        Profile a CUDA executable using Nsight Compute.
        
        Args:
            executable: Path to CUDA executable
            output: Output CSV file path
            metrics: Comma-separated list of metrics (default: memory-focused metrics)
            kernel: Specific kernel to profile (default: all)
            replay_mode: 'kernel' or 'application'
            target_processes: 'all' or 'current'
            additional_args: Additional arguments to pass to ncu
            
        Returns:
            ProfileResult with success status and output info
        """
        # Build command
        cmd = [self.ncu_path]
        
        # Output format
        cmd.extend(['--csv', '--log-file', output.replace('.csv', '.log')])
        
        # Metrics
        if metrics:
            metric_list = [m.strip() for m in metrics.split(',')]
        else:
            metric_list = self.DEFAULT_MEMORY_METRICS
        cmd.extend(['--metrics', ','.join(metric_list)])
        
        # Kernel filter
        if kernel:
            cmd.extend(['--kernel-name', kernel])
        
        # Replay mode
        cmd.extend(['--replay-mode', replay_mode])
        
        # Target processes
        cmd.extend(['--target-processes', target_processes])
        
        # Additional args
        if additional_args:
            cmd.extend(additional_args)
        
        # Executable
        cmd.append(executable)
        
        print(f"Running: {' '.join(shlex.quote(c) for c in cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            # Check if CSV was generated
            csv_path = output
            if not os.path.exists(csv_path):
                # Try with .csv extension
                if not csv_path.endswith('.csv'):
                    csv_path = csv_path + '.csv'
            
            success = result.returncode == 0 and os.path.exists(csv_path)
            
            return ProfileResult(
                success=success,
                output_file=csv_path if success else None,
                error=result.stderr if not success else None,
                stdout=result.stdout,
                stderr=result.stderr,
                return_code=result.returncode
            )
            
        except subprocess.TimeoutExpired:
            return ProfileResult(
                success=False,
                error="Profiling timed out (300s)",
                return_code=-1
            )
        except Exception as e:
            return ProfileResult(
                success=False,
                error=str(e),
                return_code=-1
            )
    
    def get_available_metrics(self) -> List[str]:
        """Get list of available metrics from ncu."""
        try:
            result = subprocess.run([self.ncu_path, '--query-metrics'], 
                                  capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                # Parse metrics from output
                metrics = []
                for line in result.stdout.split('\n'):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        metrics.append(line.split()[0])
                return metrics
        except Exception:
            pass
        return []
    
    def get_kernel_names(self, executable: str) -> List[str]:
        """Get list of kernel names in an executable."""
        try:
            result = subprocess.run(
                [self.ncu_path, '--kernel-name', '.*', '--csv', executable],
                capture_output=True, text=True, timeout=60
            )
            if result.returncode == 0:
                # Parse kernel names from CSV
                import csv
                from io import StringIO
                reader = csv.DictReader(StringIO(result.stdout))
                kernels = set()
                for row in reader:
                    if 'Kernel Name' in row:
                        kernels.add(row['Kernel Name'])
                return sorted(kernels)
        except Exception:
            pass
        return []


if __name__ == '__main__':
    # Test
    runner = NCURunner()
    print("Available metrics:", len(runner.get_available_metrics()))