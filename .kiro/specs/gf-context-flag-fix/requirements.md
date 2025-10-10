# Requirements Document

## Introduction

The gf command-line tool has a bug in its `-c` (context) flag implementation. According to the documentation, the `-c` flag should display a specified number of lines both before and after each match to provide context. However, the current implementation only displays lines before the match, missing the lines that should appear after each match. This significantly reduces the usefulness of the context feature for understanding the surrounding code or content.

## Requirements

### Requirement 1

**User Story:** As a developer using gf to search through code files, I want the `-c` flag to show lines both before and after each match, so that I can understand the complete context around my search results.

#### Acceptance Criteria

1. WHEN I use `gf -s "pattern" -c 2` THEN the system SHALL display 2 lines before each match AND 2 lines after each match
2. WHEN I use `gf -s "pattern" -c 5` THEN the system SHALL display 5 lines before each match AND 5 lines after each match
3. WHEN I use `gf -s "pattern" -c 0` THEN the system SHALL display only the match line with no context lines
4. WHEN context lines extend beyond the end of a file THEN the system SHALL display all available lines without error

### Requirement 2

**User Story:** As a user searching through log files, I want the context display to be properly formatted and clearly distinguish between before-context, match, and after-context lines, so that I can easily understand the sequence of events.

#### Acceptance Criteria

1. WHEN displaying context lines THEN the system SHALL maintain the existing line number format `[line_num]` for all lines
2. WHEN displaying matches with context THEN the system SHALL preserve the existing highlighting for matched text
3. WHEN displaying context THEN the system SHALL maintain proper spacing and formatting consistency with the current output format
4. WHEN multiple matches occur close together THEN the system SHALL handle overlapping context appropriately without duplicating lines

### Requirement 3

**User Story:** As a user with performance concerns, I want the context feature to maintain good performance even when processing large files, so that my searches complete in reasonable time.

#### Acceptance Criteria

1. WHEN processing large files with context THEN the system SHALL use efficient buffering to avoid loading entire files into memory
2. WHEN reading context lines THEN the system SHALL minimize file I/O operations through appropriate buffering strategies
3. WHEN the maxline limit is set THEN the system SHALL respect the limit and not read beyond it even when collecting after-context
4. WHEN processing files THEN the system SHALL maintain the existing performance characteristics of the search operation

### Requirement 4

**User Story:** As a developer maintaining the gf codebase, I want the context fix to integrate cleanly with existing functionality, so that no existing features are broken.

#### Acceptance Criteria

1. WHEN the context fix is implemented THEN all existing command-line options SHALL continue to work as before
2. WHEN no context is specified (default behavior) THEN the system SHALL behave exactly as it currently does
3. WHEN context is combined with other flags like `-t`, `-i`, `-m` THEN all combinations SHALL work correctly
4. WHEN processing binary files, permission issues, or other edge cases THEN the system SHALL handle them the same way as before