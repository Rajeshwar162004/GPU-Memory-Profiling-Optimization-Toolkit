#!/usr/bin/env python3
"""
GPUOpt - Automated CUDA GPU Memory Profiling, Bottleneck Detection & Optimization Toolkit

A tool that profiles CUDA applications using NVIDIA Nsight Compute, analyzes memory behavior,
detects bottlenecks, provides human-readable explanations and optimization recommendations,
and validates improvements through before/after benchmarking.
"""

import sys
import os

# Add the gpuopt directory (parent of cli) and current working dir to sys.path
cli_dir = os.path.dirname(os.path.abspath(__file__))
gpuopt_dir = os.path.dirname(cli_dir)
if gpuopt_dir not in sys.path:
    sys.path.insert(0, gpuopt_dir)

from cli.commands import GPUOptCLI


def main():
    """Main entry point for GPUOpt CLI."""
    cli = GPUOptCLI()
    sys.exit(cli.run(sys.argv[1:]))


if __name__ == "__main__":
    main()