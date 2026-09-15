#!/usr/bin/env python3
"""
GPUOpt NCU Runner

Wrapper around NVIDIA Nsight Compute CLI for profiling CUDA applications.
"""

import subprocess
import os
import sys
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
    
    # Default metrics for memory and performance analysis
    DEFAULT_MEMORY_METRICS = [
        'dram__throughput.avg.pct_of_peak_sustained_elapsed',
        'dram__bytes_read.sum',
        'dram__bytes_write.sum',
        'l1tex__t_sector_hit_rate.pct',
        'l1tex__t_sector_miss_rate.pct',
        'l2__slice_read_throughput.avg.pct_of_peak_sustained_elapsed',
        'l2__slice_write_throughput.avg.pct_of_peak_sustained_elapsed',
        'l2__cache_hit_rate.pct',
        'memory__throughput.avg.pct_of_peak_sustained_elapsed',
        'memory__bytes_read.sum',
        'memory__bytes_write.sum',
        'shared__load_throughput.avg.pct_of_peak_sustained_elapsed',
        'shared__store_throughput.avg.pct_of_peak_sustained_elapsed',
        'shared__bank_conflicts.sum',
        'sm__throughput.avg.pct_of_peak_sustained_elapsed',
        'achieved_occupancy',
        'theoretical_occupancy',
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
        self.use_sudo = False
        self._verify_ncu()
    
    def _verify_ncu(self) -> None:
        """Verify ncu is available."""
        try:
            result = subprocess.run([self.ncu_path, '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                sudo_res = subprocess.run(['sudo', '-n', self.ncu_path, '--version'],
                                          capture_output=True, text=True, timeout=10)
                if sudo_res.returncode == 0:
                    self.use_sudo = True
                else:
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
                launch_count: int = 1,
                additional_args: Optional[List[str]] = None) -> ProfileResult:
        """
        Profile a CUDA executable using Nsight Compute.
        """
        if not output.endswith('.csv'):
            output = output + '.csv'
            
        def run_profile_command(cmd_args: List[str]) -> ProfileResult:
            base_cmd = ['sudo', '-n', self.ncu_path] if self.use_sudo else [self.ncu_path]
            full_cmd = base_cmd + cmd_args
            print(f"Running: {' '.join(shlex.quote(c) for c in full_cmd)}")
            try:
                proc = subprocess.run(full_cmd, capture_output=True, text=True, timeout=300)
                stdout, stderr = proc.stdout, proc.stderr
                
                # Check for permission error
                if "ERR_NVGPUCTRPERM" in stderr or "ERR_NVGPUCTRPERM" in stdout:
                    if not self.use_sudo:
                        print("Performance counter permission restricted. Escalating with sudo -n...")
                        self.use_sudo = True
                        return run_profile_command(cmd_args)
                
                # Check if CSV content was emitted
                if stdout and ('"Kernel Name"' in stdout or '"Metric Name"' in stdout or '"ID"' in stdout):
                    with open(output, 'w') as f:
                        f.write(stdout)
                    return ProfileResult(
                        success=True,
                        output_file=output,
                        stdout=stdout,
                        stderr=stderr,
                        return_code=0
                    )
                elif proc.returncode == 0 and os.path.exists(output) and os.path.getsize(output) > 0:
                    return ProfileResult(
                        success=True,
                        output_file=output,
                        stdout=stdout,
                        stderr=stderr,
                        return_code=0
                    )
                else:
                    return ProfileResult(
                        success=False,
                        error=stderr or stdout or "ncu produced no CSV output",
                        stdout=stdout,
                        stderr=stderr,
                        return_code=proc.returncode
                    )
            except subprocess.TimeoutExpired:
                return ProfileResult(success=False, error="Profiling timed out (300s)", return_code=-1)
            except Exception as e:
                return ProfileResult(success=False, error=str(e), return_code=-1)
        
        # Build command arguments
        cmd_args = ['--csv']
        
        if launch_count > 0:
            cmd_args.extend(['--launch-count', str(launch_count)])
            
        if metrics:
            metric_list = [m.strip() for m in metrics.split(',')]
            cmd_args.extend(['--metrics', ','.join(metric_list)])
        
        if kernel:
            cmd_args.extend(['--kernel-name', kernel])
            
        cmd_args.extend(['--replay-mode', replay_mode])
        cmd_args.extend(['--target-processes', target_processes])
        
        if additional_args:
            cmd_args.extend(additional_args)
            
        cmd_args.append(executable)
        
        return run_profile_command(cmd_args)