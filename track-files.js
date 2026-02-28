#!/usr/bin/env node

/**
 * File Timestamp Tracker
 *
 * Tracks file modification timestamps in a project and maintains an immutable
 * historical record. Old entries are preserved even if files are modified or deleted.
 *
 * Usage: node track-files.js
 *
 * Output: file_timestamp.md in project root
 */

const fs = require('fs');
const path = require('path');

// Configuration
const OUTPUT_FILE = 'file_timestamp.md';
const EXCLUDED_FOLDERS = ['.claude', '.memo', '.octie', '.pytest_cache', '.git', 'node_modules', '__pycache__', '.venv', 'venv'];
const EXCLUDED_FILES = ['.gitignore', 'CLAUDE.md', 'file_timestamp.md', 'track-files.js'];

/**
 * Format date to human-readable string
 * @param {Date} date
 * @returns {string} Format: "Feb 26, 2026 3:30 PM"
 */
function formatDate(date) {
    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    const month = months[date.getMonth()];
    const day = date.getDate();
    const year = date.getFullYear();
    let hours = date.getHours();
    const minutes = date.getMinutes().toString().padStart(2, '0');
    const ampm = hours >= 12 ? 'PM' : 'AM';
    hours = hours % 12 || 12;

    return `${month} ${day}, ${year} ${hours}:${minutes} ${ampm}`;
}

/**
 * Check if a file or folder should be excluded
 * @param {string} name - File or folder name
 * @param {string} relativePath - Relative path from project root
 * @returns {boolean}
 */
function shouldExclude(name, relativePath) {
    // Exclude hidden files/folders (starting with .)
    if (name.startsWith('.') && !EXCLUDED_FOLDERS.includes(name) && !EXCLUDED_FILES.includes(name)) {
        // Allow specific hidden files if needed, but by default exclude
        return true;
    }

    // Check excluded folders
    if (EXCLUDED_FOLDERS.includes(name)) {
        return true;
    }

    // Check excluded files
    if (EXCLUDED_FILES.includes(name)) {
        return true;
    }

    // Check if any parent folder is excluded
    for (const folder of EXCLUDED_FOLDERS) {
        if (relativePath.includes(folder + path.sep) || relativePath.startsWith(folder + path.sep)) {
            return true;
        }
    }

    return false;
}

/**
 * Recursively scan directory for all files
 * @param {string} dir - Directory to scan
 * @param {string} baseDir - Base directory (project root)
 * @param {Map<string, {mtime: Date}>} files - Map to store found files
 */
function scanDirectory(dir, baseDir, files) {
    const entries = fs.readdirSync(dir, { withFileTypes: true });

    for (const entry of entries) {
        const fullPath = path.join(dir, entry.name);
        const relativePath = path.relative(baseDir, fullPath);

        if (shouldExclude(entry.name, relativePath)) {
            continue;
        }

        if (entry.isDirectory()) {
            scanDirectory(fullPath, baseDir, files);
        } else if (entry.isFile()) {
            try {
                const stat = fs.statSync(fullPath);
                files.set(relativePath.replace(/\\/g, '/'), {
                    mtime: stat.mtime
                });
            } catch (err) {
                // Skip files that can't be read
                console.warn(`Warning: Could not stat ${fullPath}: ${err.message}`);
            }
        }
    }
}

/**
 * Parse existing markdown file to extract tracked files
 * @param {string} filePath - Path to markdown file
 * @returns {Map<string, string>} Map of file path -> timestamp string
 */
function parseExistingMarkdown(filePath) {
    const tracked = new Map();

    if (!fs.existsSync(filePath)) {
        return tracked;
    }

    const content = fs.readFileSync(filePath, 'utf-8');
    const lines = content.split('\n');

    let currentFolder = '';

    for (const line of lines) {
        // Check for folder header: ## folder/path/
        const folderMatch = line.match(/^## (.+)\/$/);
        if (folderMatch) {
            currentFolder = folderMatch[1];
            // Convert (root) to empty string for path matching
            if (currentFolder === '(root)') {
                currentFolder = '';
            }
            continue;
        }

        // Check for table row: | filename | timestamp |
        const rowMatch = line.match(/^\| ([^|]+) \| ([^|]+) \|$/);
        if (rowMatch && currentFolder !== undefined) {
            const fileName = rowMatch[1].trim();
            const timestamp = rowMatch[2].trim();
            // Skip header row
            if (fileName === 'File' && timestamp === 'First Tracked') {
                continue;
            }
            // Skip separator row
            if (fileName.startsWith('-')) {
                continue;
            }
            const relativePath = currentFolder ? `${currentFolder}/${fileName}` : fileName;
            tracked.set(relativePath, timestamp);
        }
    }

    return tracked;
}

/**
 * Group files by folder
 * @param {Map<string, {mtime: Date}>} files
 * @returns {Map<string, Array<{name: string, mtime: Date}>>}
 */
function groupByFolder(files) {
    const grouped = new Map();

    for (const [relativePath, info] of files) {
        const parts = relativePath.split('/');
        const fileName = parts.pop();
        const folder = parts.join('/');

        if (!grouped.has(folder)) {
            grouped.set(folder, []);
        }

        grouped.get(folder).push({
            name: fileName,
            mtime: info.mtime,
            path: relativePath
        });
    }

    // Sort files within each folder by mtime (oldest first)
    for (const [folder, filelist] of grouped) {
        filelist.sort((a, b) => a.mtime - b.mtime);
    }

    // Sort folders alphabetically
    const sortedGrouped = new Map(
        [...grouped.entries()].sort((a, b) => {
            // Put root folder (empty string) first
            if (a[0] === '') return -1;
            if (b[0] === '') return 1;
            return a[0].localeCompare(b[0]);
        })
    );

    return sortedGrouped;
}

/**
 * Generate markdown content
 * @param {Map<string, Array<{name: string, mtime: Date, path: string}>>} groupedFiles
 * @param {Map<string, string>} existingTracked - Already tracked files with their timestamps
 * @returns {string}
 */
function generateMarkdown(groupedFiles, existingTracked) {
    let content = `# File Timestamp Tracker

Tracks file creation/modification timestamps. Historical entries are immutable -
even if files are modified or deleted, their original tracking time is preserved.

Last updated: ${formatDate(new Date())}

---

`;

    // Section 1: By Folder
    content += `## By Folder\n\n`;

    for (const [folder, filelist] of groupedFiles) {
        const folderDisplay = folder || '(root)';

        content += `### ${folderDisplay}/\n\n`;
        content += `| File | First Tracked |\n`;
        content += `|------|---------------|\n`;

        for (const file of filelist) {
            const timestamp = existingTracked.has(file.path)
                ? existingTracked.get(file.path)
                : formatDate(file.mtime);

            content += `| ${file.name} | ${timestamp} |\n`;
        }

        content += '\n';
    }

    // Section 2: Chronological (all files sorted by time)
    content += `---\n\n`;
    content += `## Chronological Timeline\n\n`;
    content += `All files sorted by first tracked time (oldest first):\n\n`;
    content += `| File | First Tracked |\n`;
    content += `|------|---------------|\n`;

    // Collect all files with their timestamps
    const allFiles = [];
    for (const [folder, filelist] of groupedFiles) {
        for (const file of filelist) {
            const timestamp = existingTracked.has(file.path)
                ? existingTracked.get(file.path)
                : formatDate(file.mtime);
            allFiles.push({
                path: file.path,
                timestamp: timestamp,
                mtime: file.mtime
            });
        }
    }

    // Sort by mtime (oldest first)
    allFiles.sort((a, b) => a.mtime - b.mtime);

    for (const file of allFiles) {
        content += `| ${file.path} | ${file.timestamp} |\n`;
    }

    return content;
}

/**
 * Main function
 */
function main() {
    const projectRoot = process.cwd();
    const outputPath = path.join(projectRoot, OUTPUT_FILE);

    console.log('File Timestamp Tracker');
    console.log('======================');
    console.log(`Scanning: ${projectRoot}`);

    // Scan current files
    const currentFiles = new Map();
    scanDirectory(projectRoot, projectRoot, currentFiles);
    console.log(`Found ${currentFiles.size} files`);

    // Parse existing tracked files
    const existingTracked = parseExistingMarkdown(outputPath);
    console.log(`Previously tracked: ${existingTracked.size} files`);

    // Find new files
    let newCount = 0;
    const newFiles = [];
    for (const [filePath] of currentFiles) {
        if (!existingTracked.has(filePath)) {
            newCount++;
            newFiles.push(filePath);
        }
    }
    console.log(`New files to add: ${newCount}`);
    if (newFiles.length > 0) {
        console.log('New files:', newFiles);
    }

    if (newCount === 0 && existingTracked.size > 0) {
        console.log('No new files to track. Markdown file unchanged.');
        return;
    }

    // Group by folder
    const groupedFiles = groupByFolder(currentFiles);

    // Generate markdown
    const markdown = generateMarkdown(groupedFiles, existingTracked);

    // Write output
    fs.writeFileSync(outputPath, markdown, 'utf-8');
    console.log(`Updated: ${outputPath}`);
    console.log('Done!');
}

// Run
main();
