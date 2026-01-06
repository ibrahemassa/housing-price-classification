#!/usr/bin/env python3
"""
Script to run the MLOps Monitoring Dashboard.
Usage: python run_dashboard.py
"""

import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    dashboard_path = Path(__file__).parent / "src" / "monitoring" / "dashboard.py"

    if not dashboard_path.exists():
        print(f"Error: Dashboard file not found at {dashboard_path}")
        sys.exit(1)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(dashboard_path),
            "--server.port=8501",
            "--server.address=0.0.0.0",
        ]
    )




