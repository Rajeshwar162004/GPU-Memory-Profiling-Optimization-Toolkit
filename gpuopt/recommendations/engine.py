#!/usr/bin/env python3
"""
GPUOpt Recommendation Engine

Generates human-readable optimization recommendations for detected GPU memory issues.
"""

import os
import yaml
from typing import List, Dict, Any


class RecommendationEngine:
    """Generates specific, actionable CUDA optimization recommendations."""
    
    def __init__(self, kb_path: str = None):
        if not kb_path:
            kb_path = os.path.join(os.path.dirname(__file__), 'knowledge_base.yaml')
        self.kb = self._load_kb(kb_path)
        
    def _load_kb(self, path: str) -> Dict[str, Any]:
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    return yaml.safe_load(f) or {}
            except Exception:
                pass
        return {}
        
    def generate(self, analysis_result: Dict[str, Any], classification: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate list of recommendation objects based on detected issues.
        """
        recommendations = []
        issues = analysis_result.get('issues', [])
        
        seen_ids = set()
        for issue in issues:
            issue_id = issue.get('id')
            if issue_id and issue_id not in seen_ids:
                seen_ids.add(issue_id)
                kb_entry = self.kb.get(issue_id, {})
                
                rec = {
                    "issue_id": issue_id,
                    "problem": issue.get('problem'),
                    "severity": issue.get('severity'),
                    "title": kb_entry.get('title', f"Optimize {issue.get('problem')}"),
                    "description": kb_entry.get('description', issue.get('explanation')),
                    "example": kb_entry.get('example', ''),
                    "reason": kb_entry.get('reason', '')
                }
                recommendations.append(rec)
                
        return recommendations
