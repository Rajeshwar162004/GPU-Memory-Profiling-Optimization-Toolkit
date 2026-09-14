#!/usr/bin/env python3
"""
GPUOpt NSys Runner

Wrapper around NVIDIA Nsight Systems CLI for application-level profiling.
"""

import subprocess
import os
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


class NSysRunner:
    """Runs NVIDIA Nsight Systems (nsys) to profile CUDA applications at application level."""
    
    def __init__(self, nsys_path: str = 'nsys'):
        """
        Initialize NSys runner.
        
        Args:
            nsys_path: Path to nsys executable (default: 'nsys' from PATH)
        """
        self.nsys_path = nsys_path
        self._verify_nsys()
    
    def _verify_nsys(self) -> None:
        """Verify nsys is available."""
        try:
            result = subprocess.run([self.nsys_path, '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                raise RuntimeError(f"nsys not found or failed: {result.stderr}")
        except FileNotFoundError:
            raise RuntimeError(f"nsys not found in PATH. Please install NVIDIA Nsight Systems.")
        except subprocess.TimeoutExpired:
            raise RuntimeError("nsys --version timed out")
    
    def profile(self, 
                executable: str,
                output: str,
                duration: Optional[float] = None,
                trace: Optional[List[str]] = None,
                additional_args: Optional[List[str]] = None) -> ProfileResult:
        """
        Profile a CUDA executable using Nsight Systems.
        
        Args:
            executable: Path to CUDA executable
            output: Output file path (without extension)
            duration: Maximum profiling duration in seconds
            trace: List of trace types (e.g., ['cuda', 'nvtx', 'osrt'])
            additional_args: Additional arguments to pass to nsys
            
        Returns:
            ProfileResult with success status and output info
        """
        # Build command
        cmd = [self.nsys_path, 'profile']
        
        # Output
        cmd.extend(['-o', output])
        
        # Trace options
        if trace:
            cmd.extend(['-t', ','.join(trace)])
        else:
            cmd.extend(['-t', 'cuda,nvtx,osrt'])
        
        # Duration
        if duration:
            cmd.extend(['--duration', str(duration)])
        
        # Force overwrite
        cmd.append('--force-overwrite=true')
        
        # Additional args
        if additional_args:
            cmd.extend(additional_args)
        
        # Executable
        cmd.append(executable)
        
        print(f"Running: {' '.join(shlex.quote(c) for c in cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            # Check if output was generated
            qdrep_path = output + '.qdrep'
            sqlite_path = output + '.sqlite'
            
            success = result.returncode == 0 and (os.path.exists(qdrep_path) or os.path.exists(sqlite_path))
            
            return ProfileResult(
                success=success,
                output_file=qdrep_path if os.path.exists(qdrep_path) else sqlite_path,
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
    
    def export_sqlite(self, qdrep_file: str, output: Optional[str] = None) -> ProfileResult:
        """Export qdrep file to SQLite for analysis."""
        if not output:
            output = qdrep_file.replace('.qdrep', '.sqlite')
        
        cmd = [self.nsys_path, 'export', '--type', 'sqlite', '-o', output, qdrep_file]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            success = result.returncode == 0 and os.path.exists(output)
            return ProfileResult(
                success=success,
                output_file=output if success else None,
                error=result.stderr if not success else None,
                stdout=result.stdout,
                stderr=result.stderr,
                return_code=result.returncode
            )
        except Exception as e:
            return ProfileResult(success=False, error=str(e), return_code=-1)
    
    def export_csv(self, qdrep_file: str, output: Optional[str] = None, 
                   type: str = 'gpukernsum') -> ProfileResult:
        """Export qdrep file to CSV."""
        if not output:
            output = qdrep_file.replace('.qdrep', f'_{type}.csv')
        
        cmd = [self.nsys_path, 'export', '--type', type, '-o', output, qdrep_file]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            success = result.returncode == 0 and os.path.exists(output)
            return ProfileResult(
                success=success,
                output_file=output if success else None,
                error=result.stderr if not success else None,
                stdout=result.stdout,
                stderr=result.stderr,
                return_code=result.returncode
            )
        except Exception as e:
            return ProfileResult(success=False, error=str(e), return_code=-1)


if __name__ == '__main__':
    # Test
    runner = NSysRunner()
    print("NSys runner initialized")