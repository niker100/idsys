#!/usr/bin/env python3
"""
Checkpointing module for resumable computation in identification systems analysis.

This module provides functionality to:
- Save intermediate results periodically to CSV files
- Resume computation from the last checkpoint
- Track progress and prevent data loss
"""

import pandas as pd
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime


class AnalysisCheckpoint:
    """
    A checkpoint manager for long-running analysis computations.
    
    Features:
    - Periodic saving of results to CSV
    - Resume capability from last checkpoint
    - Progress tracking with metadata
    - Automatic backup creation
    """
    
    def __init__(self, 
                 output_dir: str,
                 analysis_name: str,
                 save_interval: int = 10,
                 backup_count: int = 3):
        """
        Initialize the checkpoint manager.
        
        Args:
            output_dir: Directory to save checkpoint files
            analysis_name: Name of the analysis (used for file naming)
            save_interval: Save every N iterations/parameter combinations
            backup_count: Number of backup files to keep
        """
        self.output_dir = Path(output_dir)
        self.analysis_name = analysis_name
        self.save_interval = save_interval
        self.backup_count = backup_count
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.csv_file = self.output_dir / f"{analysis_name}_results.csv"
        self.metadata_file = self.output_dir / f"{analysis_name}_metadata.json"
        self.progress_file = self.output_dir / f"{analysis_name}_progress.json"
        
        # Initialize data structures
        self.results_df = pd.DataFrame()
        self.metadata = {}
        self.progress = {
            "completed_params": [],
            "current_iteration": 0,
            "total_iterations": 0,
            "start_time": None,
            "last_save_time": None,
            "analysis_complete": False
        }
        
        # Setup logging
        self._setup_logging()
        
        # Load existing checkpoint if available
        self._load_checkpoint()
    
    def _setup_logging(self):
        """Setup logging for checkpoint operations."""
        log_file = self.output_dir / f"{self.analysis_name}_checkpoint.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(f"checkpoint_{self.analysis_name}")
    
    def _load_checkpoint(self):
        """Load existing checkpoint data if available."""
        try:
            # Load CSV results
            if self.csv_file.exists():
                self.results_df = pd.read_csv(self.csv_file)
                self.logger.info(f"Loaded {len(self.results_df)} existing results from {self.csv_file}")
            
            # Load metadata
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
                self.logger.info(f"Loaded metadata from {self.metadata_file}")
            
            # Load progress
            if self.progress_file.exists():
                with open(self.progress_file, 'r') as f:
                    saved_progress = json.load(f)
                    self.progress.update(saved_progress)
                self.logger.info(f"Loaded progress: {len(self.progress['completed_params'])} completed parameter sets")
                
        except Exception as e:
            self.logger.warning(f"Could not load checkpoint: {e}")
            self.logger.info("Starting fresh analysis")
    
    def initialize_analysis(self, 
                          parameter_sets: List[Dict[str, Any]], 
                          metadata: Dict[str, Any] = None):
        """
        Initialize a new analysis or resume from checkpoint.
        
        Args:
            parameter_sets: List of parameter dictionaries to test
            metadata: Additional metadata about the analysis
            
        Returns:
            List of remaining parameter sets to process
        """
        if metadata:
            self.metadata.update(metadata)
        
        self.progress["total_iterations"] = len(parameter_sets)
        
        if self.progress["start_time"] is None:
            self.progress["start_time"] = datetime.now().isoformat()
            self.logger.info(f"Starting new analysis: {self.analysis_name}")
        else:
            self.logger.info(f"Resuming analysis: {self.analysis_name}")
        
        # Filter out already completed parameter sets
        # Convert parameters to hashable format for comparison
        def make_hashable(obj):
            """Convert lists and other unhashable types to hashable tuples."""
            if isinstance(obj, list):
                return tuple(obj)
            elif isinstance(obj, dict):
                return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
            else:
                return obj
        
        completed_params = set(make_hashable(p) for p in self.progress["completed_params"])
        remaining_params = [p for p in parameter_sets 
                          if make_hashable(p) not in completed_params]
        
        self.logger.info(f"Total parameters: {len(parameter_sets)}, "
                        f"Completed: {len(self.progress['completed_params'])}, "
                        f"Remaining: {len(remaining_params)}")
        
        return remaining_params
    
    def add_result(self, params: Dict[str, Any], results: Dict[str, Any]):
        """
        Add a result for a specific parameter set.
        
        Args:
            params: Parameter dictionary used for this result
            results: Results dictionary containing metrics
        """
        # Combine parameters and results into a single row
        row_data = {**params, **results}
        
        # Add timestamp
        row_data['timestamp'] = datetime.now().isoformat()
        
        # Add to dataframe
        self.results_df = pd.concat([self.results_df, pd.DataFrame([row_data])], 
                                  ignore_index=True)
        
        # Update progress
        self.progress["completed_params"].append(params)
        self.progress["current_iteration"] += 1
        
        # Check if we should save
        if (self.progress["current_iteration"] % self.save_interval == 0 or 
            self.progress["current_iteration"] == self.progress["total_iterations"]):
            self.save_checkpoint()
    
    def save_checkpoint(self):
        """Save current state to files."""
        try:
            # Create backups
            self._create_backups()
            
            # Save CSV results
            self.results_df.to_csv(self.csv_file, index=False)
            
            # Save metadata
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            
            # Save progress
            self.progress["last_save_time"] = datetime.now().isoformat()
            with open(self.progress_file, 'w') as f:
                json.dump(self.progress, f, indent=2)
            
            self.logger.info(f"Checkpoint saved: {self.progress['current_iteration']}/{self.progress['total_iterations']} iterations completed")
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
    
    def _create_backups(self):
        """Create backup copies of existing files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for file_path in [self.csv_file, self.metadata_file, self.progress_file]:
            if file_path.exists():
                backup_dir = self.output_dir / "backups"
                backup_dir.mkdir(exist_ok=True)
                
                backup_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
                backup_path = backup_dir / backup_name
                
                # Copy file to backup
                backup_path.write_bytes(file_path.read_bytes())
        
        # Clean up old backups
        self._cleanup_old_backups()
    
    def _cleanup_old_backups(self):
        """Remove old backup files, keeping only the most recent ones."""
        backup_dir = self.output_dir / "backups"
        if not backup_dir.exists():
            return
        
        # Group backups by base name
        backup_groups = {}
        for backup_file in backup_dir.glob("*"):
            base_name = "_".join(backup_file.stem.split("_")[:-2])  # Remove timestamp
            if base_name not in backup_groups:
                backup_groups[base_name] = []
            backup_groups[base_name].append(backup_file)
        
        # Keep only the most recent backups for each group
        for base_name, files in backup_groups.items():
            files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            for old_file in files[self.backup_count:]:
                old_file.unlink()
    
    def finalize_analysis(self):
        """Mark analysis as complete and perform final save."""
        self.progress["analysis_complete"] = True
        self.progress["end_time"] = datetime.now().isoformat()
        self.save_checkpoint()
        
        self.logger.info(f"Analysis '{self.analysis_name}' completed successfully!")
        self.logger.info(f"Final results saved to: {self.csv_file}")
        
        # Print summary statistics
        self._print_summary()
    
    def _print_summary(self):
        """Print a summary of the analysis results."""
        if len(self.results_df) == 0:
            return
        
        print("\n" + "="*60)
        print(f"ANALYSIS SUMMARY: {self.analysis_name}")
        print("="*60)
        print(f"Total parameter combinations tested: {len(self.results_df)}")
        print(f"Results file: {self.csv_file}")
        print(f"Columns in results: {list(self.results_df.columns)}")
        
        if 'timestamp' in self.results_df.columns:
            start_time = pd.to_datetime(self.results_df['timestamp']).min()
            end_time = pd.to_datetime(self.results_df['timestamp']).max()
            duration = end_time - start_time
            print(f"Analysis duration: {duration}")
        
        print("="*60)
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """Get the current results as a pandas DataFrame."""
        return self.results_df.copy()
    
    def is_parameter_completed(self, params: Dict[str, Any]) -> bool:
        """Check if a parameter set has already been completed."""
        def make_hashable(obj):
            """Convert lists and other unhashable types to hashable tuples."""
            if isinstance(obj, list):
                return tuple(obj)
            elif isinstance(obj, dict):
                return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
            else:
                return obj
        
        param_hashable = make_hashable(params)
        completed_hashable = set(make_hashable(p) for p in self.progress["completed_params"])
        return param_hashable in completed_hashable
    
    def get_completion_percentage(self) -> float:
        """Get the percentage of analysis completed."""
        if self.progress["total_iterations"] == 0:
            return 0.0
        return (self.progress["current_iteration"] / self.progress["total_iterations"]) * 100


# Convenience function for quick integration
def create_checkpoint_manager(output_dir: str, 
                            analysis_name: str, 
                            save_interval: int = 10) -> AnalysisCheckpoint:
    """
    Create a checkpoint manager with sensible defaults.
    
    Args:
        output_dir: Directory to save checkpoint files
        analysis_name: Name of the analysis
        save_interval: Save every N iterations
    
    Returns:
        Configured AnalysisCheckpoint instance
    """
    return AnalysisCheckpoint(output_dir, analysis_name, save_interval)
