import os
import subprocess
import sys
from datetime import datetime

def run_command(command, cwd=None):
    """Execute shell command, return (success, output)"""
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            cwd=cwd,
            capture_output=True, 
            text=True,
            check=True
        )
        return True, result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return False, e.stderr.strip()

def get_file_status(repo_path):
    """Get lists of modified, deleted, and untracked files"""
    # Get modified/deleted tracked files
    success, output = run_command("git diff --name-only --diff-filter=ACDMRTUXB", repo_path)
    tracked_changes = output.splitlines() if success and output else []
    
    # Get untracked files
    success, output = run_command("git ls-files --others --exclude-standard", repo_path)
    untracked = output.splitlines() if success and output else []
    
    return tracked_changes, untracked

def generate_commit_message(file_path, change_type):
    """Generate semantic commit message based on file type and change"""
    ext = os.path.splitext(file_path)[1].lower()
    filename = os.path.basename(file_path)
    
    # Language/framework specific messages
    lang_map = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.jsx': 'react',
        '.tsx': 'react',
        '.html': 'html',
        '.css': 'css',
        '.scss': 'scss',
        '.json': 'json',
        '.yml': 'yaml',
        '.yaml': 'yaml',
        '.md': 'documentation',
        '.txt': 'text',
        '.env': 'environment',
        '.dockerfile': 'docker',
        '.sql': 'database',
        '.xml': 'xml',
        '.java': 'java',
        '.cpp': 'c++',
        '.c': 'c',
        '.cs': 'c#',
        '.go': 'go',
        '.rs': 'rust',
        '.rb': 'ruby',
        '.php': 'php',
        '.swift': 'swift',
        '.kt': 'kotlin',
        '.scala': 'scala',
        '.sh': 'shell script',
        '.bat': 'batch script',
        '.ps1': 'powershell',
        '.cfg': 'config',
        '.ini': 'config',
        '.toml': 'config',
    }
    
    tech = lang_map.get(ext, 'file')
    action = "Update" if change_type == "modified" else "Delete" if change_type == "deleted" else "Add"
    
    return f"{action} {tech} {filename}"

def main():
    repo_path = input("Enter repository path (or press Enter for current directory): ").strip()
    if not repo_path:
        repo_path = os.getcwd()
    
    # Verify git repo
    if not os.path.exists(os.path.join(repo_path, '.git')):
        print(f"❌ Error: Not a git repository: {repo_path}")
        sys.exit(1)
    
    # Switch to main branch
    print("🔄 Switching to daddydash branch...")
    success, output = run_command("git checkout daddydash", repo_path)
    if not success:
        # Try creating main branch
        success, _ = run_command("git checkout -b daddydash", repo_path)
        if not success:
            print("❌ Failed to access daddydash branch")
            sys.exit(1)
    
    # Get file statuses
    tracked_changes, untracked = get_file_status(repo_path)
    all_files = tracked_changes + untracked
    
    if not all_files:
        print("✅ No changes to commit")
        return
    
    print(f"\nFound {len(tracked_changes)} tracked changes and {len(untracked)} new files")
    confirm = input(f"Commit {len(all_files)} files individually? (y/N): ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("Operation cancelled.")
        sys.exit(0)
    
    try:
        # Process tracked changes (modified/deleted)
        for file in tracked_changes:
            # Determine change type
            success, status = run_command(f"git status --porcelain -- {file}", repo_path)
            if not success:
                continue
                
            change_type = "modified"
            if status.startswith("D"):
                change_type = "deleted"
            elif status.startswith("A"):
                change_type = "added"
            
            # Stage individual file
            success, _ = run_command(f"git add {file}", repo_path)
            if not success:
                print(f"⚠️  Failed to stage {file}")
                continue
            
            # Generate and commit message
            msg = generate_commit_message(file, change_type)
            success, _ = run_command(f'git commit -m "{msg}"', repo_path)
            if success:
                print(f"✅ Committed: {msg}")
            else:
                print(f"⚠️  Commit failed for {file}")
        
        # Handle untracked files (new files)
        if untracked:
            # Stage all untracked files at once (more efficient)
            success, _ = run_command("git add .", repo_path)
            if success:
                msg = f"Add {len(untracked)} new files"
                if len(untracked) == 1:
                    msg = generate_commit_message(untracked[0], "added")
                
                success, _ = run_command(f'git commit -m "{msg}"', repo_path)
                if success:
                    print(f"✅ Committed: {msg}")
        
        # Final push
        print("\n🚀 Pushing to GitHub...")
        success, output = run_command("git push origin hubtel_payment", repo_path)
        if not success:
            print(f"❌ Push failed: {output}")
            print("\n💡 Tips:")
            print("- Verify remote: git remote -v")
            print("- Check authentication")
            sys.exit(1)
        
        print("\n✅ Successfully pushed all changes to GitHub hubtel_payment branch!")
        
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()