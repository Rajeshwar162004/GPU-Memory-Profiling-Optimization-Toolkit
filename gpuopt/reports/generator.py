#!/usr/bin/env python3
"""
GPUOpt HTML & CLI Report Generator
"""

import os
import json
from typing import Dict, Any, List


class ReportGenerator:
    """Generates standalone HTML performance reports."""
    
    def __init__(self, template_path: str = None):
        if not template_path:
            template_path = os.path.join(os.path.dirname(__file__), 'templates', 'report.html')
        self.template_path = template_path

    def generate(self, data: Any, template: str = None) -> str:
        """
        Generate HTML content from analysis or comparison data dictionary.
        """
        path = template or self.template_path
        with open(path, 'r') as f:
            html = f.read()
            
        record = data[0] if isinstance(data, list) and len(data) > 0 else data
        if not isinstance(record, dict):
            record = {}
            
        dev = record.get('device', {})
        gpu_name = dev.get('name', 'NVIDIA GPU')
        gpu_cc = dev.get('compute_capability', '8.9')
        kname = record.get('kernel', 'CUDA Kernel')
        
        cls = record.get('classification', {})
        classification = cls.get('category', 'MEMORY-LIMITED / MEMORY-INEFFICIENT')
        
        issues = record.get('analysis', {}).get('issues', [])
        issues_html = ""
        for issue in issues:
            sev = issue.get('severity', 'WARNING')
            issues_html += f"""
            <div class="issue-item {sev}">
              <strong>[{sev}] {issue.get('problem')}</strong><br>
              <span class="stat-label">Evidence:</span> {issue.get('evidence')}<br>
              <span class="stat-label">Explanation:</span> {issue.get('explanation')}
            </div>
            """
            
        recs = record.get('recommendations', [])
        recs_html = ""
        for rec in recs:
            example_code = f"<pre>{rec.get('example')}</pre>" if rec.get('example') else ""
            recs_html += f"""
            <div class="issue-item">
              <strong>{rec.get('title')}</strong><br>
              {rec.get('description')}<br>
              {example_code}
            </div>
            """
            
        html = html.replace('id="gpu-name">NVIDIA GPU', f'id="gpu-name">{gpu_name}')
        html = html.replace('id="gpu-cc">8.9', f'id="gpu-cc">{gpu_cc}')
        html = html.replace('id="kernel-name">transpose_naive', f'id="kernel-name">{kname}')
        html = html.replace('id="classification">MEMORY-LIMITED / MEMORY-INEFFICIENT', f'id="classification">{classification}')
        
        if issues_html:
            html = html.replace('<!-- Issues dynamically inserted -->', issues_html)
        if recs_html:
            html = html.replace('<!-- Recommendations dynamically inserted -->', recs_html)
            
        return html
