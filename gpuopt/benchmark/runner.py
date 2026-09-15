#!/usr/bin/env python3
"""
GPUOpt Benchmark Runner

Executes CUDA executables with warmup runs and statistical repeated measurements.
"""

import subprocess
import time
import numpy as np
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any


@dataclass
class BenchmarkResult:
    """Result of a benchmark execution."""
    success: bool
    executable: str
    warmup_runs: int
    measured_runs: int
    mean_ms: float
    median_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    raw_times_ms: List[float]
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BenchmarkRunner:
    """Runs benchmarks on CUDA executables with reproducible statistical aggregation."""
    
    def run(self, executable: str, warmup_runs: int = 10, measured_runs: int = 100, args: Optional[List[str]] = None) -> BenchmarkResult:
        """
        Execute benchmark runs for a given executable.
        """
        cmd = [executable]
        if args:
            cmd.extend(args)
            
        times = []
        
        try:
            # Warmup runs
            for _ in range(warmup_runs):
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                if proc.returncode != 0:
                    return BenchmarkResult(
                        success=False, executable=executable, warmup_runs=warmup_runs, measured_runs=measured_runs,
                        mean_ms=0, median_ms=0, std_ms=0, min_ms=0, max_ms=0, raw_times_ms=[], error=proc.stderr
                    )
                    
            # Measured runs
            for _ in range(measured_runs):
                t0 = time.perf_counter()
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                t1 = time.perf_counter()
                if proc.returncode != 0:
                    return BenchmarkResult(
                        success=False, executable=executable, warmup_runs=warmup_runs, measured_runs=measured_runs,
                        mean_ms=0, median_ms=0, std_ms=0, min_ms=0, max_ms=0, raw_times_ms=[], error=proc.stderr
                    )
                times.append((t1 - t0) * 1000.0) # Convert to ms
                
            arr = np.array(times)
            
            return BenchmarkResult(
                success=True,
                executable=executable,
                warmup_runs=warmup_runs,
                measured_runs=measured_runs,
                mean_ms=float(np.mean(arr)),
                median_ms=float(np.median(arr)),
                std_ms=float(np.std(arr)),
                min_ms=float(np.min(arr)),
                max_ms=float(np.max(arr)),
                raw_times_ms=times
            )
        except Exception as e:
            return BenchmarkResult(
                success=False, executable=executable, warmup_runs=warmup_runs, measured_runs=measured_runs,
                mean_ms=0, median_ms=0, std_ms=0, min_ms=0, max_ms=0, raw_times_ms=[], error=str(e)
            )
