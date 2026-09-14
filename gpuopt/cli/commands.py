#!/usr/bin/env python3
"""
GPUOpt CLI Commands

Implements the command-line interface for GPUOpt.
"""

import argparse
import sys
import os
import subprocess
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

# Import internal modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from profiler.ncu_runner import NCURunner
from profiler.profile_manager import ProfileManager
from parser.csv_parser import CSVParser
from parser.metric_normalizer import MetricNormalizer
from analyzer.analyzer import Analyzer
from analyzer.bottleneck_classifier import BottleneckClassifier
from recommendations.engine import RecommendationEngine
from benchmark.runner import BenchmarkRunner
from benchmark.comparison import ComparisonEngine
from reports.generator import ReportGenerator


class GPUOptCLI:
    """Main CLI class for GPUOpt."""
    
    def __init__(self):
        self.parser = self._create_parser()
        self.profile_manager = ProfileManager()
        self.ncu_runner = NCURunner()
        self.csv_parser = CSVParser()
        self.metric_normalizer = MetricNormalizer()
        self.analyzer = Analyzer()
        self.bottleneck_classifier = BottleneckClassifier()
        self.recommendation_engine = RecommendationEngine()
        self.benchmark_runner = BenchmarkRunner()
        self.comparison_engine = ComparisonEngine()
        self.report_generator = ReportGenerator()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create the argument parser with all subcommands."""
        parser = argparse.ArgumentParser(
            prog='gpuopt',
            description='GPUOpt - Automated CUDA GPU Memory Profiling, Bottleneck Detection & Optimization Toolkit',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  gpuopt profile ./matmul                    # Profile a CUDA executable
  gpuopt analyze profile.csv                 # Analyze a profile CSV file
  gpuopt analyze profile.csv --kernel matmul # Analyze specific kernel
  gpuopt benchmark ./before ./after          # Benchmark before/after
  gpuopt compare before.json after.json      # Compare two profiles
  gpuopt report profile.json                 # Generate HTML report
            """
        )
        
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Profile command
        profile_parser = subparsers.add_parser('profile', help='Profile a CUDA executable using Nsight Compute')
        profile_parser.add_argument('executable', help='Path to CUDA executable')
        profile_parser.add_argument('-o', '--output', help='Output profile file (default: <executable>_profile.csv)')
        profile_parser.add_argument('--metrics', help='Comma-separated list of metrics to collect')
        profile_parser.add_argument('--kernel', help='Specific kernel to profile (default: all)')
        profile_parser.add_argument('--replay-mode', choices=['kernel', 'application'], default='kernel',
                                   help='Replay mode for Nsight Compute')
        profile_parser.add_argument('--target-processes', choices=['all', 'current'], default='all',
                                   help='Target processes to profile')
        
        # Analyze command
        analyze_parser = subparsers.add_parser('analyze', help='Analyze a profile CSV file')
        analyze_parser.add_argument('profile', help='Path to profile CSV file')
        analyze_parser.add_argument('--kernel', help='Specific kernel to analyze')
        analyze_parser.add_argument('--output', '-o', help='Output analysis file (JSON)')
        analyze_parser.add_argument('--format', choices=['text', 'json'], default='text',
                                   help='Output format')
        
        # Benchmark command
        benchmark_parser = subparsers.add_parser('benchmark', help='Run benchmark on CUDA executable')
        benchmark_parser.add_argument('executable', help='Path to CUDA executable')
        benchmark_parser.add_argument('--warmup', type=int, default=10, help='Number of warmup runs')
        benchmark_parser.add_argument('--runs', type=int, default=100, help='Number of measured runs')
        benchmark_parser.add_argument('--output', '-o', help='Output benchmark file (JSON)')
        
        # Compare command
        compare_parser = subparsers.add_parser('compare', help='Compare two profiles or benchmarks')
        compare_parser.add_argument('before', help='Before profile/benchmark file')
        compare_parser.add_argument('after', help='After profile/benchmark file')
        compare_parser.add_argument('--output', '-o', help='Output comparison file (JSON)')
        compare_parser.add_argument('--format', choices=['text', 'json'], default='text',
                                   help='Output format')
        
        # Report command
        report_parser = subparsers.add_parser('report', help='Generate HTML report from analysis')
        report_parser.add_argument('input', help='Input analysis or profile file')
        report_parser.add_argument('--output', '-o', help='Output HTML file')
        report_parser.add_argument('--template', help='Custom template file')
        
        # List kernels command
        list_parser = subparsers.add_parser('list-kernels', help='List kernels in a profile')
        list_parser.add_argument('profile', help='Path to profile CSV file')
        
        return parser
    
    def run(self, args: List[str]) -> int:
        """Run the CLI with given arguments."""
        parsed = self.parser.parse_args(args)
        
        if not parsed.command:
            self.parser.print_help()
            return 1
        
        try:
            if parsed.command == 'profile':
                return self._cmd_profile(parsed)
            elif parsed.command == 'analyze':
                return self._cmd_analyze(parsed)
            elif parsed.command == 'benchmark':
                return self._cmd_benchmark(parsed)
            elif parsed.command == 'compare':
                return self._cmd_compare(parsed)
            elif parsed.command == 'report':
                return self._cmd_report(parsed)
            elif parsed.command == 'list-kernels':
                return self._cmd_list_kernels(parsed)
            else:
                print(f"Unknown command: {parsed.command}")
                return 1
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            if os.environ.get('GPUOPT_DEBUG'):
                import traceback
                traceback.print_exc()
            return 1
    
    def _cmd_profile(self, args) -> int:
        """Profile a CUDA executable."""
        executable = args.executable
        if not os.path.exists(executable):
            print(f"Error: Executable not found: {executable}")
            return 1
        
        output = args.output
        if not output:
            base = os.path.splitext(os.path.basename(executable))[0]
            output = f"{base}_profile.csv"
        
        print(f"Profiling {executable}...")
        print(f"Output: {output}")
        
        # Run Nsight Compute
        result = self.ncu_runner.profile(
            executable=executable,
            output=output,
            metrics=args.metrics,
            kernel=args.kernel,
            replay_mode=args.replay_mode,
            target_processes=args.target_processes
        )
        
        if result.success:
            print(f"Profile saved to {output}")
            return 0
        else:
            print(f"Profiling failed: {result.error}")
            return 1
    
    def _cmd_analyze(self, args) -> int:
        """Analyze a profile CSV file."""
        profile_path = args.profile
        if not os.path.exists(profile_path):
            print(f"Error: Profile file not found: {profile_path}")
            return 1
        
        print(f"Analyzing {profile_path}...")
        
        # Parse CSV
        raw_data = self.csv_parser.parse(profile_path)
        
        # Normalize metrics
        normalized = self.metric_normalizer.normalize(raw_data)
        
        # Filter by kernel if specified
        if args.kernel:
            normalized = [k for k in normalized if k.name == args.kernel]
            if not normalized:
                print(f"Error: Kernel '{args.kernel}' not found in profile")
                return 1
        
        # Analyze each kernel
        results = []
        for kernel_data in normalized:
            print(f"\nAnalyzing kernel: {kernel_data.name}")
            
            # Run analysis
            analysis = self.analyzer.analyze(kernel_data)
            
            # Classify bottleneck
            classification = self.bottleneck_classifier.classify(kernel_data, analysis)
            
            # Generate recommendations
            recommendations = self.recommendation_engine.generate(analysis, classification)
            
            result = {
                'kernel': kernel_data.name,
                'device': kernel_data.device_info,
                'execution': kernel_data.execution_metrics,
                'memory': kernel_data.memory_metrics,
                'analysis': analysis,
                'classification': classification,
                'recommendations': recommendations
            }
            results.append(result)
            
            # Print summary
            self._print_analysis_summary(result)
        
        # Save output if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nAnalysis saved to {args.output}")
        
        return 0
    
    def _cmd_benchmark(self, args) -> int:
        """Run benchmark on CUDA executable."""
        executable = args.executable
        if not os.path.exists(executable):
            print(f"Error: Executable not found: {executable}")
            return 1
        
        print(f"Benchmarking {executable}...")
        print(f"Warmup runs: {args.warmup}, Measured runs: {args.runs}")
        
        result = self.benchmark_runner.run(
            executable=executable,
            warmup_runs=args.warmup,
            measured_runs=args.runs
        )
        
        if result.success:
            print(f"\nBenchmark Results:")
            print(f"  Median: {result.median_ms:.3f} ms")
            print(f"  Mean:   {result.mean_ms:.3f} ms")
            print(f"  Std:    {result.std_ms:.3f} ms")
            print(f"  Min:    {result.min_ms:.3f} ms")
            print(f"  Max:    {result.max_ms:.3f} ms")
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(result.to_dict(), f, indent=2, default=str)
                print(f"\nBenchmark saved to {args.output}")
            return 0
        else:
            print(f"Benchmark failed: {result.error}")
            return 1
    
    def _cmd_compare(self, args) -> int:
        """Compare two profiles or benchmarks."""
        before_path = args.before
        after_path = args.after
        
        if not os.path.exists(before_path):
            print(f"Error: Before file not found: {before_path}")
            return 1
        if not os.path.exists(after_path):
            print(f"Error: After file not found: {after_path}")
            return 1
        
        print(f"Comparing {before_path} vs {after_path}...")
        
        # Load both files
        with open(before_path) as f:
            before = json.load(f)
        with open(after_path) as f:
            after = json.load(f)
        
        # Compare
        comparison = self.comparison_engine.compare(before, after)
        
        # Print results
        self._print_comparison(comparison)
        
        # Save output if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(comparison, f, indent=2, default=str)
            print(f"\nComparison saved to {args.output}")
        
        return 0
    
    def _cmd_report(self, args) -> int:
        """Generate HTML report."""
        input_path = args.input
        if not os.path.exists(input_path):
            print(f"Error: Input file not found: {input_path}")
            return 1
        
        output = args.output
        if not output:
            base = os.path.splitext(os.path.basename(input_path))[0]
            output = f"{base}_report.html"
        
        print(f"Generating report from {input_path}...")
        
        with open(input_path) as f:
            data = json.load(f)
        
        html = self.report_generator.generate(data, template=args.template)
        
        with open(output, 'w') as f:
            f.write(html)
        
        print(f"Report saved to {output}")
        return 0
    
    def _cmd_list_kernels(self, args) -> int:
        """List kernels in a profile."""
        profile_path = args.profile
        if not os.path.exists(profile_path):
            print(f"Error: Profile file not found: {profile_path}")
            return 1
        
        raw_data = self.csv_parser.parse(profile_path)
        kernels = self.csv_parser.get_kernels(raw_data)
        
        print(f"Kernels in {profile_path}:")
        for i, kernel in enumerate(kernels, 1):
            print(f"  {i}. {kernel}")
        
        return 0
    
    def _print_analysis_summary(self, result: Dict[str, Any]) -> None:
        """Print analysis summary to console."""
        print(f"\n{'='*60}")
        print(f"GPUOpt Analysis: {result['kernel']}")
        print(f"{'='*60}")
        
        # Device info
        dev = result.get('device', {})
        print(f"\nDevice: {dev.get('name', 'Unknown')}")
        print(f"Compute Capability: {dev.get('compute_capability', 'Unknown')}")
        
        # Execution
        exec_metrics = result.get('execution', {})
        print(f"\nExecution Time: {exec_metrics.get('elapsed_cycles', 'N/A')} cycles")
        
        # Classification
        cls = result.get('classification', {})
        print(f"\nClassification: {cls.get('category', 'Unknown')}")
        print(f"Confidence: {cls.get('confidence', 'N/A')}")
        
        # Analysis issues
        analysis = result.get('analysis', {})
        issues = analysis.get('issues', [])
        if issues:
            print(f"\nDetected Issues ({len(issues)}):")
            for issue in issues:
                severity = issue.get('severity', 'INFO')
                symbol = {'INFO': 'ℹ', 'WARNING': '⚠', 'HIGH': '❌', 'CRITICAL': '🔥'}.get(severity, '•')
                print(f"  {symbol} [{severity}] {issue.get('problem', 'Unknown')}")
                print(f"     Evidence: {issue.get('evidence', 'N/A')}")
                print(f"     Explanation: {issue.get('explanation', 'N/A')}")
        
        # Recommendations
        recs = result.get('recommendations', [])
        if recs:
            print(f"\nRecommendations ({len(recs)}):")
            for i, rec in enumerate(recs, 1):
                print(f"  {i}. {rec.get('title', 'Unknown')}")
                print(f"     {rec.get('description', 'N/A')}")
                if rec.get('example'):
                    print(f"     Example: {rec['example']}")
    
    def _print_comparison(self, comparison: Dict[str, Any]) -> None:
        """Print comparison results to console."""
        print(f"\n{'='*60}")
        print(f"GPUOpt Comparison")
        print(f"{'='*60}")
        
        speedup = comparison.get('speedup', 0)
        improvement = comparison.get('improvement_pct', 0)
        
        print(f"\nSpeedup: {speedup:.2f}x")
        print(f"Execution Time Improvement: {improvement:.1f}%")
        
        metrics = comparison.get('metrics', {})
        if metrics:
            print(f"\nMetric Comparison:")
            print(f"{'Metric':<30} {'Before':>12} {'After':>12} {'Change':>10}")
            print(f"{'-'*64}")
            for name, vals in metrics.items():
                before = vals.get('before', 0)
                after = vals.get('after', 0)
                change = vals.get('change_pct', 0)
                print(f"{name:<30} {before:>12.3f} {after:>12.3f} {change:>+9.1f}%")


if __name__ == '__main__':
    cli = GPUOptCLI()
    sys.exit(cli.run(sys.argv[1:]))