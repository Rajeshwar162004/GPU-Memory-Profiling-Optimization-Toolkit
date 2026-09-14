#!/usr/bin/env python3
"""
GPUOpt Profile Manager

Manages profiling sessions, stores and retrieves profile data.
"""

import os
import json
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import uuid


@dataclass
class ProfileSession:
    """Represents a profiling session."""
    id: str
    name: str
    executable: str
    timestamp: str
    profile_file: str
    log_file: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ProfileManager:
    """Manages profiling sessions and profile data storage."""
    
    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialize profile manager.
        
        Args:
            base_dir: Base directory for storing profiles (default: ~/.gpuopt/profiles)
        """
        if base_dir is None:
            base_dir = os.path.expanduser('~/.gpuopt/profiles')
        
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.sessions_file = self.base_dir / 'sessions.json'
        self._load_sessions()
    
    def _load_sessions(self) -> None:
        """Load sessions from disk."""
        self.sessions: Dict[str, ProfileSession] = {}
        if self.sessions_file.exists():
            try:
                with open(self.sessions_file) as f:
                    data = json.load(f)
                for session_data in data:
                    session = ProfileSession(**session_data)
                    self.sessions[session.id] = session
            except Exception:
                pass
    
    def _save_sessions(self) -> None:
        """Save sessions to disk."""
        data = [asdict(s) for s in self.sessions.values()]
        with open(self.sessions_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def create_session(self, 
                       name: str,
                       executable: str,
                       profile_file: str,
                       log_file: Optional[str] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> ProfileSession:
        """Create a new profiling session."""
        session = ProfileSession(
            id=str(uuid.uuid4())[:8],
            name=name,
            executable=executable,
            timestamp=datetime.now().isoformat(),
            profile_file=profile_file,
            log_file=log_file,
            metadata=metadata or {}
        )
        self.sessions[session.id] = session
        self._save_sessions()
        return session
    
    def get_session(self, session_id: str) -> Optional[ProfileSession]:
        """Get a session by ID."""
        return self.sessions.get(session_id)
    
    def list_sessions(self) -> List[ProfileSession]:
        """List all sessions, sorted by timestamp (newest first)."""
        return sorted(self.sessions.values(), key=lambda s: s.timestamp, reverse=True)
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session and its files."""
        session = self.sessions.get(session_id)
        if not session:
            return False
        
        # Delete profile file
        try:
            if os.path.exists(session.profile_file):
                os.remove(session.profile_file)
            if session.log_file and os.path.exists(session.log_file):
                os.remove(session.log_file)
        except Exception:
            pass
        
        del self.sessions[session_id]
        self._save_sessions()
        return True
    
    def export_session(self, session_id: str, output_dir: str) -> bool:
        """Export session files to a directory."""
        session = self.sessions.get(session_id)
        if not session:
            return False
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            shutil.copy2(session.profile_file, output_path / f"{session.name}_profile.csv")
            if session.log_file and os.path.exists(session.log_file):
                shutil.copy2(session.log_file, output_path / f"{session.name}_profile.log")
            
            # Save metadata
            with open(output_path / f"{session.name}_metadata.json", 'w') as f:
                json.dump(asdict(session), f, indent=2, default=str)
            
            return True
        except Exception:
            return False
    
    def get_profile_path(self, session_id: str) -> Optional[str]:
        """Get the profile file path for a session."""
        session = self.sessions.get(session_id)
        return session.profile_file if session else None
    
    def cleanup_old_sessions(self, keep: int = 10) -> int:
        """Remove old sessions, keeping only the most recent N."""
        sessions = self.list_sessions()
        removed = 0
        for session in sessions[keep:]:
            if self.delete_session(session.id):
                removed += 1
        return removed


if __name__ == '__main__':
    # Test
    manager = ProfileManager('/tmp/gpuopt_test')
    session = manager.create_session(
        name='test_matmul',
        executable='./matmul',
        profile_file='/tmp/matmul_profile.csv'
    )
    print(f"Created session: {session.id}")
    
    sessions = manager.list_sessions()
    print(f"Total sessions: {len(sessions)}")