#!/usr/bin/env python3
"""
GitLab Cleanup Migration Script
================================
Safely moves unused/standalone scripts to depreciated/ folder.

This script:
- Creates timestamped backup of current state
- Moves test/utility files not used by main pipeline
- Generates migration log
- Provides rollback capability

Usage:
    python migrate_to_depreciated.py [--dry-run] [--rollback]

Options:
    --dry-run    Show what would be moved without actually moving
    --rollback   Restore files from most recent migration

Author: Thermal Post-Processing Team
Date: January 12, 2026
"""

import os
import shutil
import json
from pathlib import Path
from datetime import datetime
import argparse

# Files to move to depreciated/
FILES_TO_MOVE = [
    # Standalone test scripts
    "test_flir_loader.py",
    "test_gradient_boosting.py", 
    "test_regression_plot.py",
    "test_temporal_spatial_features.py",
    "test_validation.py",
    "validation_output.txt",
    
    # Utility/research scripts not used in main pipeline
    "calibration_convergence_analysis.py",
    "check_delta_t.py",
    "check_deltas.py",
    "check_pcbs.py",
    "config_file_finder.py",
    "roi_pixel_mapper.py",
    "visualize_ml_model_comparison.py",
    "viz_calibration_validation.py",
]

# Files that need special handling (used by ml_model/)
SPECIAL_CASE_FILES = [
    "cnn_data_preprocessor.py",
    "flir_frame_loader.py",
]

def get_workspace_root():
    """Get the workspace root directory"""
    return Path(__file__).parent.resolve()

def create_migration_backup(workspace_root):
    """Create timestamped backup before migration"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"pre_gitlab_migration_{timestamp}"
    backup_dir = workspace_root / "backup" / backup_name
    
    print(f"\n📦 Creating backup: {backup_name}")
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy files that will be moved
    files_backed_up = []
    for filename in FILES_TO_MOVE + SPECIAL_CASE_FILES:
        src = workspace_root / filename
        if src.exists():
            dst = backup_dir / filename
            shutil.copy2(src, dst)
            files_backed_up.append(filename)
            print(f"  ✓ Backed up: {filename}")
    
    # Save migration metadata
    metadata = {
        "timestamp": timestamp,
        "backup_dir": str(backup_dir),
        "files_moved": FILES_TO_MOVE,
        "special_case_files": SPECIAL_CASE_FILES,
        "files_backed_up": files_backed_up,
        "migration_date": datetime.now().isoformat()
    }
    
    metadata_file = backup_dir / "migration_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  ✓ Backup complete: {len(files_backed_up)} files")
    return backup_dir, metadata

def move_files_to_depreciated(workspace_root, dry_run=False):
    """Move files to depreciated/ folder"""
    depreciated_dir = workspace_root / "depreciated" / "gitlab_cleanup_20260112"
    
    if not dry_run:
        depreciated_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🗂️  Moving files to: {depreciated_dir.name}/")
    print("=" * 70)
    
    moved_files = []
    missing_files = []
    
    for filename in FILES_TO_MOVE:
        src = workspace_root / filename
        dst = depreciated_dir / filename
        
        if src.exists():
            if dry_run:
                print(f"  [DRY-RUN] Would move: {filename}")
            else:
                shutil.move(str(src), str(dst))
                print(f"  ✓ Moved: {filename}")
            moved_files.append(filename)
        else:
            print(f"  ⚠ Not found: {filename}")
            missing_files.append(filename)
    
    return moved_files, missing_files, depreciated_dir

def handle_special_cases(workspace_root, dry_run=False):
    """Handle files that need investigation (ml_model dependencies)"""
    print(f"\n⚠️  SPECIAL CASE FILES (Manual Review Needed):")
    print("=" * 70)
    
    ml_model_dir = workspace_root / "ml_model"
    ml_utils_dir = ml_model_dir / "utils"
    
    for filename in SPECIAL_CASE_FILES:
        src = workspace_root / filename
        
        if src.exists():
            print(f"\n  📄 {filename}")
            print(f"     Used by: ml_model/cnn_thermal_modeling/")
            print(f"     Recommendation: Move to ml_model/utils/")
            
            if ml_model_dir.exists():
                if dry_run:
                    print(f"     [DRY-RUN] Would suggest: ml_model/utils/{filename}")
                else:
                    print(f"     Action: MANUAL - Review and move if needed")
            else:
                print(f"     Action: MANUAL - ml_model/ not found, keep or move to depreciated/")

def create_migration_report(workspace_root, moved_files, missing_files, backup_dir):
    """Create detailed migration report"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = workspace_root / f"MIGRATION_REPORT_{timestamp}.txt"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("GITLAB CLEANUP MIGRATION REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Migration Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Backup Location: {backup_dir}\n\n")
        
        f.write(f"Files Successfully Moved ({len(moved_files)}):\n")
        f.write("-" * 70 + "\n")
        for filename in sorted(moved_files):
            f.write(f"  ✓ {filename}\n")
        
        if missing_files:
            f.write(f"\nFiles Not Found ({len(missing_files)}):\n")
            f.write("-" * 70 + "\n")
            for filename in sorted(missing_files):
                f.write(f"  ⚠ {filename}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("SPECIAL CASE FILES (Requires Manual Review):\n")
        f.write("=" * 70 + "\n")
        for filename in SPECIAL_CASE_FILES:
            f.write(f"  📄 {filename}\n")
            f.write(f"     → Consider moving to ml_model/utils/\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("ROLLBACK INSTRUCTIONS:\n")
        f.write("=" * 70 + "\n")
        f.write(f"To restore files:\n")
        f.write(f"  python migrate_to_depreciated.py --rollback\n\n")
        f.write(f"Or manually copy from: {backup_dir}\n")
    
    print(f"\n📄 Migration report saved: {report_file.name}")
    return report_file

def rollback_migration(workspace_root):
    """Restore files from most recent migration backup"""
    backup_root = workspace_root / "backup"
    
    # Find most recent migration backup
    migration_backups = sorted(backup_root.glob("pre_gitlab_migration_*"))
    
    if not migration_backups:
        print("❌ No migration backups found!")
        return False
    
    latest_backup = migration_backups[-1]
    print(f"\n🔄 Rolling back from: {latest_backup.name}")
    
    # Load metadata
    metadata_file = latest_backup / "migration_metadata.json"
    if not metadata_file.exists():
        print("❌ Migration metadata not found!")
        return False
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Restore files
    restored = []
    for filename in metadata['files_backed_up']:
        src = latest_backup / filename
        dst = workspace_root / filename
        
        if src.exists():
            if dst.exists():
                print(f"  ⚠ File exists, skipping: {filename}")
            else:
                shutil.copy2(src, dst)
                print(f"  ✓ Restored: {filename}")
                restored.append(filename)
        else:
            print(f"  ⚠ Backup file not found: {filename}")
    
    print(f"\n✅ Rollback complete: {len(restored)} files restored")
    return True

def main():
    parser = argparse.ArgumentParser(description='GitLab cleanup migration tool')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be moved without actually moving')
    parser.add_argument('--rollback', action='store_true',
                       help='Restore files from most recent migration backup')
    args = parser.parse_args()
    
    workspace_root = get_workspace_root()
    
    print("\n" + "=" * 70)
    print("  GITLAB CLEANUP MIGRATION")
    print("=" * 70)
    print(f"Workspace: {workspace_root}")
    
    if args.rollback:
        rollback_migration(workspace_root)
        return
    
    if args.dry_run:
        print("\n🔍 DRY-RUN MODE - No files will be moved")
    
    # Create backup
    if not args.dry_run:
        backup_dir, metadata = create_migration_backup(workspace_root)
    else:
        backup_dir = workspace_root / "backup" / "dry_run"
        print("\n📦 [DRY-RUN] Backup would be created")
    
    # Move files
    moved_files, missing_files, depreciated_dir = move_files_to_depreciated(
        workspace_root, dry_run=args.dry_run
    )
    
    # Handle special cases
    handle_special_cases(workspace_root, dry_run=args.dry_run)
    
    # Create report
    if not args.dry_run:
        report_file = create_migration_report(
            workspace_root, moved_files, missing_files, backup_dir
        )
    
    # Summary
    print("\n" + "=" * 70)
    print("  MIGRATION SUMMARY")
    print("=" * 70)
    print(f"Files moved: {len(moved_files)}")
    print(f"Files not found: {len(missing_files)}")
    print(f"Special case files: {len(SPECIAL_CASE_FILES)} (manual review needed)")
    
    if not args.dry_run:
        print(f"\n✅ Migration complete!")
        print(f"   Moved to: depreciated/{depreciated_dir.name}/")
        print(f"   Backup: {backup_dir.name}/")
    else:
        print(f"\n🔍 Dry-run complete - no changes made")
        print(f"   Run without --dry-run to execute migration")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
