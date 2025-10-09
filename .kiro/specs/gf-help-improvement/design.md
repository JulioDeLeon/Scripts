# Design Document

## Overview

This design outlines the improvements to the 'gf' Perl script's help system and documentation. The goal is to transform the current minimal error handling and technical documentation into a user-friendly, comprehensive help system that guides users effectively.

The design focuses on three main areas:
1. Enhanced command-line help and error messaging within the script
2. Improved man page content and structure
3. Better integration between the script's help output and the man page

## Architecture

### Help System Components

```
gf script
├── Command Line Argument Processing (enhanced)
│   ├── Help flag handling (--help, -h)
│   ├── Version flag handling (--version)
│   └── Improved error messages
├── Help Content Functions
│   ├── show_usage() - Brief usage summary
│   ├── show_help() - Comprehensive help
│   ├── show_version() - Version information
│   └── show_examples() - Common use cases
└── Error Handling (enhanced)
    ├── Friendly error messages
    ├── Context-aware suggestions
    └── Graceful degradation
```

### Documentation Structure

```
Documentation
├── Inline Help (in script)
│   ├── Usage summary
│   ├── Option descriptions
│   ├── Examples
│   └── Configuration help
└── Man Page (gf.1.md)
    ├── Enhanced introduction
    ├── Comprehensive examples
    ├── Configuration section
    └── Related tools section
```

## Components and Interfaces

### 1. Enhanced Argument Processing

**Current State:** Basic GetOptions with minimal error handling
**New Design:** Comprehensive argument processing with help integration

```perl
# New help-related options
GetOptions(
    "help|h" => \$show_help,
    "version|V" => \$show_version,
    # ... existing options
) or show_usage_and_exit();
```

**Interface:**
- `show_usage()` - Displays brief usage when errors occur
- `show_help()` - Displays comprehensive help for --help flag
- `show_version()` - Displays version information
- `show_usage_and_exit()` - Shows usage and exits with error code

### 2. Help Content Functions

**show_usage():**
- Brief one-line description
- Basic syntax
- Pointer to --help for more information

**show_help():**
- Tool description and purpose
- Organized option groups:
  - Search options (-s, --search)
  - Filtering options (-t, --target, -i, --ignore)
  - Output control (-c, --context, -m, --maxline)
  - Utility options (-d, --debug, -h, --help, -V, --version)
- Practical examples section
- Configuration file explanation
- Exit codes

**show_examples():**
- Common use cases with explanations
- Progressive complexity (simple to advanced)
- Real-world scenarios

### 3. Error Message Enhancement

**Current:** Generic "die" statements
**New:** Context-aware, helpful error messages

Error categories:
- Missing search term
- Invalid arguments
- File permission issues
- Configuration file problems

Each error includes:
- Clear description of the problem
- Suggestion for resolution
- Reference to help if appropriate

### 4. Man Page Improvements

**Structure:**
- NAME: Friendly one-line description
- SYNOPSIS: Clear usage patterns
- DESCRIPTION: Welcoming introduction explaining the tool's purpose
- OPTIONS: Organized by category with clear explanations
- CONFIGURATION: Dedicated section for config file usage
- EXAMPLES: Comprehensive examples with explanations
- EXIT STATUS: Clear exit code meanings
- FILES: Configuration file locations
- SEE ALSO: Related tools (grep, find, ag, rg)
- BUGS: More professional bug reporting information

## Data Models

### Help Content Structure

```perl
# Help content organized in data structures for maintainability
my %help_content = (
    description => "Friendly file search tool with flexible filtering",
    usage => "gf -s PATTERN [OPTIONS]",
    option_groups => {
        search => {
            title => "Search Options",
            options => { ... }
        },
        filtering => {
            title => "File Filtering",
            options => { ... }
        },
        # ...
    },
    examples => [ ... ],
    config_help => { ... }
);
```

### Version Information

```perl
my $VERSION = "1.0.0";
my $VERSION_DATE = "2024";
my $AUTHOR = "Julio de Leon";
```

## Error Handling

### Error Message Design

**Principle:** Every error should help the user understand what went wrong and what to do next.

**Error Types:**

1. **Missing Search Term**
   - Current: "Expected search term"
   - New: "Error: No search pattern provided.\nUsage: gf -s 'pattern' [options]\nTry 'gf --help' for more information."

2. **Invalid Arguments**
   - Current: "Error: Could not process command line arguments"
   - New: Specific error about which argument was invalid, with suggestion

3. **Permission Errors**
   - Current: Basic warning
   - New: Clear explanation with context about which files/directories

4. **Configuration Errors**
   - Current: None
   - New: Specific errors about config file syntax or location issues

### Graceful Degradation

- If help content fails to load, fall back to basic usage
- If man page generation fails, script help still works
- Configuration errors don't prevent basic functionality

## Testing Strategy

### Manual Testing Scenarios

1. **Help System Testing**
   - Run `gf` with no arguments
   - Run `gf --help` and `gf -h`
   - Run `gf --version`
   - Test with invalid arguments

2. **Error Message Testing**
   - Test all error conditions
   - Verify error messages are helpful
   - Check that suggestions work

3. **Documentation Testing**
   - Verify man page renders correctly
   - Test all examples in documentation
   - Check cross-references between help and man page

4. **Integration Testing**
   - Test help system with existing functionality
   - Verify no regression in core search features
   - Test configuration file integration with help

### Content Validation

- All examples in help output should be tested
- Man page examples should match script help examples
- Version information should be consistent across all outputs
- Configuration file examples should be valid

## Implementation Notes

### Backward Compatibility

- All existing command-line options remain unchanged
- Existing functionality is preserved
- Configuration file format remains the same
- Exit codes remain consistent

### Maintainability

- Help content separated into functions for easy updates
- Version information centralized
- Examples organized for easy addition/modification
- Man page source (markdown) maintained alongside script

### Performance Considerations

- Help functions only called when needed
- No performance impact on normal search operations
- Quick exit for help/version requests