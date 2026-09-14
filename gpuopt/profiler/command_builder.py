#!/usr/bin/env python3
"""
GPUOpt Command Builder

Builds Nsight Compute and Nsight Systems command lines with proper arguments.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum


class ReplayMode(Enum):
    """Nsight Compute replay modes."""
    KERNEL = 'kernel'
    APPLICATION = 'application'


class TargetProcesses(Enum):
    """Nsight Compute target processes."""
    ALL = 'all'
    CURRENT = 'current'


class TraceType(Enum):
    """Nsight Systems trace types."""
    CUDA = 'cuda'
    NVTX = 'nvtx'
    OSRT = 'osrt'
    OPENMP = 'openmp'
    MPI = 'mpi'
    OPENCL = 'opencl'
    VULKAN = 'vulkan'
    GRAPHICS = 'graphics'


@dataclass
class NCUCommand:
    """Nsight Compute command configuration."""
    executable: str
    output: str
    metrics: List[str] = field(default_factory=list)
    kernel_name: Optional[str] = None
    replay_mode: ReplayMode = ReplayMode.KERNEL
    target_processes: TargetProcesses = TargetProcesses.ALL
    additional_args: List[str] = field(default_factory=list)
    csv_output: bool = True
    log_file: Optional[str] = None
    
    def build(self) -> List[str]:
        """Build the command line as a list of arguments."""
        cmd = ['ncu']
        
        if self.csv_output:
            cmd.append('--csv')
        
        if self.log_file:
            cmd.extend(['--log-file', self.log_file])
        elif self.output:
            cmd.extend(['--log-file', self.output.replace('.csv', '.log')])
        
        if self.metrics:
            cmd.extend(['--metrics', ','.join(self.metrics)])
        
        if self.kernel_name:
            cmd.extend(['--kernel-name', self.kernel_name])
        
        cmd.extend(['--replay-mode', self.replay_mode.value])
        cmd.extend(['--target-processes', self.target_processes.value])
        
        cmd.extend(self.additional_args)
        cmd.append(self.executable)
        
        return cmd
    
    def build_string(self) -> str:
        """Build the command line as a string."""
        import shlex
        return ' '.join(shlex.quote(c) for c in self.build())


@dataclass
class NSysCommand:
    """Nsight Systems command configuration."""
    executable: str
    output: str
    trace: List[TraceType] = field(default_factory=lambda: [TraceType.CUDA, TraceType.NVTX, TraceType.OSRT])
    duration: Optional[float] = None
    additional_args: List[str] = field(default_factory=list)
    force_overwrite: bool = True
    
    def build(self) -> List[str]:
        """Build the command line as a list of arguments."""
        cmd = ['nsys', 'profile']
        
        cmd.extend(['-o', self.output])
        
        if self.trace:
            cmd.extend(['-t', ','.join(t.value for t in self.trace)])
        
        if self.duration:
            cmd.extend(['--duration', str(self.duration)])
        
        if self.force_overwrite:
            cmd.append('--force-overwrite=true')
        
        cmd.extend(self.additional_args)
        cmd.append(self.executable)
        
        return cmd
    
    def build_string(self) -> str:
        """Build the command line as a string."""
        import shlex
        return ' '.join(shlex.quote(c) for c in self.build())


class CommandBuilder:
    """High-level command builder for profiling tools."""
    
    # Predefined metric sets
    METRIC_SETS = {
        'memory': [
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
        ],
        'shared_memory': [
            'shared__load_throughput.avg.pct_of_peak_sustained_elapsed',
            'shared__store_throughput.avg.pct_of_peak_sustained_elapsed',
            'shared__bank_conflicts.sum',
        ],
        'compute': [
            'sm__throughput.avg.pct_of_peak_sustained_elapsed',
            'sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_elapsed',
            'sm__inst_executed_pipe_fp32.avg.pct_of_peak_sustained_elapsed',
            'sm__inst_executed_pipe_fp64.avg.pct_of_peak_sustained_elapsed',
            'sm__inst_executed_pipe_int.avg.pct_of_peak_sustained_elapsed',
        ],
        'occupancy': [
            'achieved_occupancy',
            'theoretical_occupancy',
        ],
        'execution': [
            'elapsed_cycles',
            'duration',
            'grid_size',
            'block_size',
            'registers_per_thread',
            'dynamic_shared_memory_per_block',
            'static_shared_memory_per_block',
        ],
        'full': [
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
            'sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_elapsed',
            'sm__inst_executed_pipe_fp32.avg.pct_of_peak_sustained_elapsed',
            'sm__inst_executed_pipe_fp64.avg.pct_of_peak_sustained_elapsed',
            'sm__inst_executed_pipe_int.avg.pct_of_peak_sustained_elapsed',
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
    }
    
    @classmethod
    def build_ncu(cls, 
                  executable: str,
                  output: str,
                  metric_set: str = 'full',
                  custom_metrics: Optional[List[str]] = None,
                  kernel_name: Optional[str] = None,
                  replay_mode: ReplayMode = ReplayMode.KERNEL,
                  target_processes: TargetProcesses = TargetProcesses.ALL,
                  additional_args: Optional[List[str]] = None) -> NCUCommand:
        """Build an NCU command with predefined metric sets."""
        if custom_metrics:
            metrics = custom_metrics
        else:
            metrics = cls.METRIC_SETS.get(metric_set, cls.METRIC_SETS['full'])
        
        return NCUCommand(
            executable=executable,
            output=output,
            metrics=metrics,
            kernel_name=kernel_name,
            replay_mode=replay_mode,
            target_processes=target_processes,
            additional_args=additional_args or []
        )
    
    @classmethod
    def build_nsys(cls,
                   executable: str,
                   output: str,
                   trace: Optional[List[TraceType]] = None,
                   duration: Optional[float] = None,
                   additional_args: Optional[List[str]] = None) -> NSysCommand:
        """Build an NSys command."""
        return NSysCommand(
            executable=executable,
            output=output,
            trace=trace or [TraceType.CUDA, TraceType.NVTX, TraceType.OSRT],
            duration=duration,
            additional_args=additional_args or []
        )
    
    @classmethod
    def get_metric_set(cls, name: str) -> List[str]:
        """Get a predefined metric set by name."""
        return cls.METRIC_SETS.get(name, [])
    
    @classmethod
    def list_metric_sets(cls) -> List[str]:
        """List available metric set names."""
        return list(cls.METRIC_SETS.keys())


if __name__ == '__main__':
    # Test
    builder = CommandBuilder()
    
    # Test NCU command
    ncu_cmd = builder.build_ncu('./matmul', 'matmul_profile.csv', metric_set='memory')
    print("NCU Command:", ncu_cmd.build_string())
    
    # Test NSys command
    nsys_cmd = builder.build_nsys('./matmul', 'matmul_profile', duration=10.0)
    print("NSys Command:", nsys_cmd.build_string())
    
    print("\nAvailable metric sets:", builder.list_metric_sets())