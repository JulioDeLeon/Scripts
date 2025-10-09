% gf(1) 1.0.0
% Julio de Leon (jangelod@gmail.com)
% 2024

# NAME
gf - friendly file search tool with flexible filtering

# SYNOPSIS
**gf** -s *PATTERN* [*OPTIONS*]

# DESCRIPTION
Welcome to **gf**, a friendly and powerful file search tool designed to make finding content in your files both easy and efficient. Whether you're searching through code, logs, or any text files, **gf** provides intuitive filtering options to help you find exactly what you're looking for.

**gf** recursively searches for text patterns within files starting from your current directory. What makes it special is its flexible filtering system - you can easily target specific file types or ignore unwanted ones, making your searches faster and more focused. Perfect for developers, system administrators, and anyone who needs to search through files regularly.

## Search Options
**-s**, **--search** *PATTERN*
: The text pattern to search for within files (required). Supports regular expressions for powerful pattern matching.

## File Filtering Options
**-t**, **--target** *PATTERN*
: Target specific file patterns to include in the search. Use comma-separated patterns like '*.pl,*.pm' to search only Perl files, or '*.js,*.ts' for JavaScript/TypeScript files.

**-i**, **--ignore** *PATTERN*
: File patterns to exclude from the search. Use comma-separated patterns like '*.log,*.tmp' to ignore log and temporary files, keeping your results clean and relevant.

## Output Control Options
**-c**, **--context** *NUMBER*
: Show NUMBER of lines before and after each match for better context. Helpful when you need to understand the surrounding code or content.

**-m**, **--maxline** *NUMBER*
: Limit the number of lines read from each file. Useful for speeding up searches in large files or when you only care about content near the beginning of files.

## Utility Options
**-d**, **--debug**
: Enable detailed debug output to see what files are being processed and how patterns are being matched.

**-h**, **--help**
: Display comprehensive help information with examples and usage guidance.

**-V**, **--version**
: Show version information and author details.

# EXAMPLES

## Basic Searches
**gf -s 'function'**
: Find all occurrences of 'function' in files within the current directory and subdirectories.

**gf -s 'TODO|FIXME'**
: Search for either 'TODO' or 'FIXME' comments using regular expression alternation.

**gf -s 'myRegex[ab]+'**
: Use advanced regex patterns to find complex text patterns.

## File Type Filtering
**gf -s 'error' -t '*.pl,*.pm'**
: Search for 'error' only in Perl files (.pl and .pm extensions).

**gf -s 'function' -t '*.js,*.ts'**
: Find 'function' declarations in JavaScript and TypeScript files only.

**gf -s 'config' -i '*.log,*.tmp'**
: Search for 'config' while ignoring log files and temporary files.

**gf -s 'import' -i '.git/*,node_modules/*'**
: Find 'import' statements while excluding version control and dependency directories.

## Context and Output Control
**gf -s 'sub main' -c 3**
: Find 'sub main' and show 3 lines of context before and after each match for better understanding.

**gf -s 'class' -c 5 -t '*.py'**
: Search for Python class definitions with 5 lines of context around each match.

**gf -s 'import' -m 100**
: Look for 'import' statements but only read the first 100 lines of each file (useful for large files).

## Advanced Combined Usage
**gf -s 'error' -t '*.pl' -c 2 -i '*.bak'**
: Complex search: find 'error' in Perl files, show 2 lines of context, but ignore backup files.

**gf -s 'TODO.*urgent' -t '*.c,*.h' -c 1**
: Find urgent TODO comments in C source and header files with minimal context.

**gf -s 'password|secret|key' -i '*.log,*.cache' -d**
: Security audit: find sensitive terms while ignoring logs and cache files, with debug output.

## Real-World Use Cases
**gf -s 'deprecated' -t '*.pl,*.pm' -c 2**
: Find deprecated function usage in a Perl codebase with context to understand impact.

**gf -s 'SELECT.*FROM' -t '*.sql,*.pl' -i '*.bak,*.old'**
: Find SQL SELECT statements in database-related files, ignoring backup files.

**gf -s 'console\.log|print|echo' -t '*.js,*.php,*.sh' -c 1**
: Find debug output statements across multiple scripting languages.

# CONFIGURATION

**gf** supports configuration files to set default ignore and target patterns, making it easy to customize behavior for your specific workflow without typing the same options repeatedly.

## Configuration File Locations

Configuration files are checked in the following order:

**~/.gfconf**
: User-specific configuration file in your home directory. This is the recommended location for personal settings.

**/etc/gfconf**
: System-wide configuration file. Settings here apply to all users on the system.

## Configuration Syntax

Configuration files use a simple line-based format with three main directives:

**ignore** *PATTERN*
: Add a file pattern to ignore during searches. Multiple ignore directives can be used.

**target** *PATTERN*
: Add a file pattern to target during searches. Multiple target directives can be used.

**source** *PATH*
: Include another configuration file. Useful for sharing common settings across multiple config files.

### Pattern Matching

- Patterns support standard shell wildcards (* and ?)
- Patterns are case-insensitive by default
- Use full paths or relative patterns as needed
- Multiple patterns can be specified with multiple directive lines

### Comments and Formatting

- Lines starting with # are treated as comments
- Empty lines are ignored
- Leading and trailing whitespace is automatically trimmed

## Configuration Examples

### Basic User Configuration (~/.gfconf)
```
# Common development file ignores
ignore *.log
ignore *.tmp
ignore *.bak
ignore *.old

# Version control and build artifacts
ignore .git/*
ignore .svn/*
ignore node_modules/*
ignore target/*
ignore build/*

# Target common source files by default
target *.pl
target *.pm
target *.js
target *.py
target *.c
target *.h
```

### Project-Specific Configuration
```
# Web development project settings
ignore *.min.js
ignore *.min.css
ignore dist/*
ignore coverage/*

target *.html
target *.css
target *.js
target *.php

# Include team-wide settings
source /shared/config/gf-team.conf
```

### Advanced Configuration with Source Directive
```
# Main config file
source ~/.gfconf-base

# Project-specific additions
ignore vendor/*
ignore cache/*
target *.twig
target *.yml
```

## How Configuration Interacts with Command Line

- Configuration file patterns are loaded first
- Command-line options (-t, -i) are added to configuration patterns
- Command-line options do not override configuration; they supplement it
- If both target patterns exist (config + command line), files must match at least one pattern from either source
- If ignore patterns exist, files matching any ignore pattern are excluded

## Troubleshooting Configuration

**Permission Issues**
: If a configuration file exists but cannot be read, **gf** will display a warning and continue without that configuration.

**Invalid Syntax**
: Unknown configuration directives will generate warnings but won't stop execution.

**Missing Source Files**
: If a source directive references a non-existent file, a warning is displayed and that directive is skipped.

**Debug Configuration**
: Use the -d (--debug) flag to see which configuration files are being loaded and what patterns are being applied.

# EXIT STATUS
**0**
: Successful completion. The search completed without errors, regardless of whether matches were found.

**1**
: Error occurred. This includes invalid command line arguments, file permission issues, missing search patterns, or other runtime errors.

# FILES
**~/.gfconf**
: User-specific configuration file. This is the primary location for personal gf settings and is checked first.

**/etc/gfconf**
: System-wide configuration file. Settings here apply to all users and are checked if no user-specific config exists.

**Additional configuration files**
: Any files referenced by 'source' directives in the main configuration files.

# SEE ALSO
**grep**(1), **find**(1), **ag**(1), **rg**(1), **ack**(1)

**grep**(1) - The classic Unix text search utility. **gf** provides more user-friendly output and better file filtering options.

**find**(1) - File system search utility. Use with **gf** to pre-filter directories before searching content.

**ag**(1) - The Silver Searcher, a fast code search tool. **gf** offers more flexible configuration options.

**rg**(1) - ripgrep, an extremely fast search tool. **gf** provides a more approachable interface for casual users.

**ack**(1) - A grep-like tool optimized for programmers. **gf** offers similar functionality with simpler configuration.

# ENVIRONMENT
**HOME**
: Used to locate the user-specific configuration file at ~/.gfconf.

# BUGS
Bug reports and feature requests can be submitted to the author. When reporting bugs, please include:

- The exact command line used
- The error message or unexpected behavior
- Your operating system and Perl version
- Contents of any relevant configuration files
- A minimal example that reproduces the issue

For the most helpful bug reports, try running your command with the --debug flag to provide additional context about what **gf** is doing internally.

# AUTHOR
Written by Julio de Leon (jangelod@gmail.com).

# COPYRIGHT
This is free software; you are free to change and redistribute it. There is NO WARRANTY, to the extent permitted by law.
