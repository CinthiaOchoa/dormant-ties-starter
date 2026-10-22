"""Run the full Phase 2 pipeline in order.

Author: Cinthia Ochoa Torre
"""
import runpy
from pathlib import Path

STEPS = ['audit_data.py', 'crosswalk.py', 'build_dataset.py', 'train_evaluate.py', 'case_study.py']

if __name__ == '__main__':
    here = Path(__file__).resolve().parent
    for step in STEPS:
        print(f'\n=== {step} ===')
        runpy.run_path(str(here / step), run_name='__main__')
