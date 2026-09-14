#!/usr/bin/env python3
"""
GPUOpt CLI Arguments

Argument definitions and validation for GPUOpt CLI.
"""

import argparse
from typing import List, Optional
from dataclasses import dataclass, field


@dataclass
class ProfileArgs:
    """Arguments for profile command."""
    executable: str
    output: Optional[str] = None
    metrics: Optional[str] = None
    kernel: Optional[str] = None
    replay_mode: str = 'kernel'
    target_processes: str = 'all'


@dataclass
class AnalyzeArgs:
    """Arguments for analyze command."""
    profile: str
    kernel: Optional[str] = None
    output: Optional[str] = None
    format: str = 'text'


@dataclass
class BenchmarkArgs:
    """Arguments for benchmark command."""
    executable: str
    warmup: int = 10
    runs: int = 100
    output: Optional[str] = None


@dataclass
class CompareArgs:
    """Arguments for compare command."""
    before: str
    after: str
    output: Optional[str] = None
    format: str = 'text'


@dataclass
class ReportArgs:
    """Arguments for report command."""
    input: str
    output: Optional[str] = None
    template: Optional[str] = None


@dataclass
class ListKernelsArgs:
    """Arguments for list-kernels command."""
    profile: str


def create_profile_parser(subparsers) -> argparse.ArgumentParser:
    """Create profile command parser."""
    parser = subparsers.add_parser('profile', help='Profile a CUDA executable using Nsight Compute')
    parser.add_argument('executable', help='Path to CUDA executable')
    parser.add_argument('-o', '--output', help='Output profile file (default: <executable>_profile.csv)')
    parser.add_argument('--metrics', help='Comma-separated list of metrics to collect')
    parser.add_argument('--kernel', help='Specific kernel to profile (default: all)')
    parser.add_argument('--replay-mode', choices=['kernel', 'application'], default='kernel',
                       help='Replay mode for Nsight Compute')
    parser.add_argument('--target-processes', choices=['all', 'current'], default='all',
                       help='Target processes to profile')
    return parser


def create_analyze_parser(subparsers) -> argparse.ArgumentParser:
    """Create analyze command parser."""
    parser = subparsers.add_parser('analyze', help='Analyze a profile CSV file')
    parser.add_argument('profile', help='Path to profile CSV file')
    parser.add_argument('--kernel', help='Specific kernel to analyze')
    parser.add_argument('--output', '-o', help='Output analysis file (JSON)')
    parser.add_argument('--format', choices=['text', 'json'], default='text',
                       help='Output format')
    return parser


def create_benchmark_parser(subparsers) -> argparse.ArgumentParser:
    """Create benchmark command parser."""
    parser = subparsers.add_parser('benchmark', help='Run benchmark on CUDA executable')
    parser.add_argument('executable', help='Path to CUDA executable')
    parser.add_argument('--warmup', type=int, default=10, help='Number of warmup runs')
    parser.add_argument('--runs', type=int, default=100, help='Number of measured runs')
    parser.add_argument('--output', '-o', help='Output benchmark file (JSON)')
    return parser


def create_compare_parser(subparsers) -> argparse.ArgumentParser:
    """Create compare command parser."""
    parser = subparsers.add_parser('compare', help='Compare two profiles or benchmarks')
    parser.add_argument('before', help='Before profile/benchmark file')
    parser.add_argument('after', help='After profile/benchmark file')
    parser.add_argument('--output', '-o', help='Output comparison file (JSON)')
    parser.add_argument('--format', choices=['text', 'json'], default='text',
                       help='Output format')
    return parser


def create_report_parser(subparsers) -> argparse.ArgumentParser:
    """Create report command parser."""
    parser = subparsers.add_parser('report', help='Generate HTML report from analysis')
    parser.add_argument('input', help='Input analysis or profile file')
    parser.add_argument('--output', '-o', help='Output HTML file')
    parser.add_argument('--template', help='Custom template file')
    return parser


def create_list_kernels_parser(subparsers) -> argparse.ArgumentParser:
    """Create list-kernels command parser."""
    parser = subparsers.add_parser('list-kernels', help='List kernels in a profile')
    parser.add_argument('profile', help='Path to profile CSV file')
    return parser


def validate_profile_args(args: ProfileArgs) -> List[str]:
    """Validate profile arguments. Returns list of errors."""
    errors = []
    if not args.executable:
        errors.append("Executable path is required")
    return errors


def validate_analyze_args(args: AnalyzeArgs) -> List[str]:
    """Validate analyze arguments. Returns list of errors."""
    errors = []
    if not args.profile:
        errors.append("Profile file path is required")
    return errors


def validate_benchmark_args(args: BenchmarkArgs) -> List[str]:
    """Validate benchmark arguments. Returns list of errors."""
    errors = []
    if not args.executable:
        errors.append("Executable path is required")
    if args.warmup < 0:
        errors.append("Warmup runs must be non-negative")
    if args.runs <= 0:
        errors.append("Measured runs must be positive")
    return errors


def validate_compare_args(args: CompareArgs) -> List[str]:
    """Validate compare arguments. Returns list of errors."""
    errors = []
    if not args.before:
        errors.append("Before file path is required")
    if not args.after:
        errors.append("After file path is required")
    return errors


def validate_report_args(args: ReportArgs) -> List[str]:
    """Validate report arguments. Returns list of errors."""
    errors = []
    if not args.input:
        errors.append("Input file path is required")
    return errors