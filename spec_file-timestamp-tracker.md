# Specification: File Timestamp Tracker

## Executive Summary

A Node.js script that maintains an immutable historical record of file creation/modification timestamps in a project. The script tracks development progress and provides a development timeline that survives file modifications - old entries remain unchanged even if files are modified later.

## Requirements

### Functional Requirements

1. **Scan Project Files**
   - Recursively scan all files in the project directory (unlimited depth)
   - Track all file types (.py, .js, .md, .json, .txt, etc.)
   - Exclude hidden files (starting with `.`)
   - Exclude specific folders: `.claude/`, `.memo/`, `.octie/`, `.pytest_cache/`
   - Exclude specific files: `.gitignore`, `CLAUDE.md`

2. **Track Timestamps**
   - Record the first time a file is discovered (creation time in the tracking system)
   - Never update existing entries - historical data is immutable
   - Only add new files that weren't previously tracked

3. **Output Format**
   - Generate `file_timestamp.md` in the project root
   - Organize by folder path (sections with headers)
   - Within each folder, sort by timestamp (oldest first)
   - Use human-readable timestamp format: `Feb 26, 2026 3:30 PM`

### Non-Functional Requirements

1. **Simplicity**: No command-line options - just run `node track-files.js`
2. **Persistence**: Survives script re-runs and file modifications
3. **Performance**: Efficient scanning for large projects
4. **Idempotency**: Running multiple times produces same result (no duplicates)

### Markdown Structure

```markdown
# File Timestamp Tracker

Last updated: Feb 26, 2026 11:30 PM

---

## phase1_basics/

| File | First Tracked |
|------|---------------|
| activations.py | Feb 25, 2026 2:15 PM |
| mlp.py | Feb 25, 2026 3:45 PM |

## phase2_architectures/

| File | First Tracked |
|------|---------------|
| cnn_layers.py | Feb 25, 2026 8:00 PM |
| attention.py | Feb 26, 2026 10:30 AM |
```

## Edge Cases

| Scenario | Behavior |
|----------|----------|
| File deleted | Keep entry in markdown (historical record) |
| File renamed | Treat as new file (new entry) |
| New file added | Add to markdown with current timestamp |
| Script run twice | No changes (idempotent) |
| Empty folder | Skip (don't create empty section) |

## Success Criteria

- [ ] Script runs without errors
- [ ] Generates `file_timestamp.md` in correct format
- [ ] Existing entries are preserved across runs
- [ ] New files are correctly added
- [ ] Excluded folders/files are not tracked
- [ ] Timestamps are human-readable

## Technical Notes

- Use Node.js built-in `fs` module for file operations
- Use `path` module for cross-platform path handling
- Use `fs.statSync().mtime` for modification time
- Parse existing markdown to extract tracked files
- Compare file paths to determine new entries
