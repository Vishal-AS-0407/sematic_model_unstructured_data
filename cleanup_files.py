"""
Cleanup Script - Archive Unused Files

Moves deprecated and unused files to OLD_ARCHIVED_FILES folder.
Keeps only the essential files for the current pipeline.
"""
import os
import shutil
from datetime import datetime

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ARCHIVE_DIR = os.path.join(PROJECT_DIR, "OLD_ARCHIVED_FILES", f"archived_{datetime.now().strftime('%Y%m%d_%H%M')}")

# Files/folders to KEEP (essential for pipeline)
KEEP_FILES = {
    # Core scripts
    'run_complete_pipeline.py',
    'run_phase5.py',
    'run_phase6_evaluation.py',
    'generate_showcase.py',
    'compare_retrieval.py',
    'config.py',
    'utils.py',
    
    # Essential modules
    'modules',
    
    # Data and output
    'data',
    'showcase',
    
    # Config files
    '.env',
    '.env.example',
    '.gitignore',
    'requirements.txt',
    'requirements_exported.txt',
    
    # Main documentation
    'README.md',
    'PROJECT_STRUCTURE.md',
    
    # Keep archive folder itself
    'OLD_ARCHIVED_FILES',
    
    # Hidden/system
    '.claude',
    '__pycache__',
    'tests',
}

# Files to ARCHIVE (deprecated/unused)
ARCHIVE_FILES = {
    # Old demos
    'demo_best.py',
    'demo_best_enhanced.py',
    'demo_hierarchical.py',
    
    # Old visualizations
    'visualize_hierarchy.py',
    'visualizations',  # old folder
    
    # Old documentation
    'BEST_ARCHITECTURE.md',
    'COMPLETE_DETAILED_PIPELINE_FLOW.md',
    'DEMO_RESULTS.md',
    'DETAILED_PIPELINE_DOCUMENTATION.md',
    'ENHANCED_DEMO_RESULTS.md',
    'FINAL_SUMMARY.md',
    'PIPELINE_RESULTS_SUMMARY.md',
    
    # Old requirements
    'requirements_improved.txt',
    
    # Utility scripts
    'download_more_pdfs.py',
    'test_retrieval.py',
    'run_best_pipeline.py',
}


def cleanup():
    """Move unused files to archive."""
    print("\n" + "="*60)
    print(" 🧹 CLEANUP: Archive Unused Files")
    print("="*60)
    
    # Create archive directory
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    print(f"\n📁 Archive folder: {ARCHIVE_DIR}")
    
    archived = []
    kept = []
    
    for item in os.listdir(PROJECT_DIR):
        item_path = os.path.join(PROJECT_DIR, item)
        
        # Skip if should keep
        if item in KEEP_FILES:
            kept.append(item)
            continue
        
        # Skip if already in archive
        if 'OLD_ARCHIVED_FILES' in item_path:
            continue
        
        # Archive if in archive list
        if item in ARCHIVE_FILES:
            target = os.path.join(ARCHIVE_DIR, item)
            try:
                if os.path.isdir(item_path):
                    shutil.move(item_path, target)
                else:
                    shutil.move(item_path, target)
                archived.append(item)
                print(f"   📦 Archived: {item}")
            except Exception as e:
                print(f"   ⚠️ Failed to archive {item}: {e}")
    
    print("\n" + "-"*60)
    print(" 📊 CLEANUP SUMMARY")
    print("-"*60)
    print(f"\n   ✅ Kept: {len(kept)} items")
    print(f"   📦 Archived: {len(archived)} items")
    
    print("\n   Kept files:")
    for f in sorted(kept)[:10]:
        print(f"      • {f}")
    if len(kept) > 10:
        print(f"      ... and {len(kept) - 10} more")
    
    print("\n   Archived files:")
    for f in archived:
        print(f"      • {f}")
    
    print("\n" + "="*60)
    print(" ✅ CLEANUP COMPLETE!")
    print("="*60)
    
    return {'archived': archived, 'kept': kept}


if __name__ == "__main__":
    cleanup()
