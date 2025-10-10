# Design Document

## Overview

The current implementation of the `-c` context flag in the gf tool only displays lines before matches due to its streaming, rolling-buffer approach. To fix this, we need to modify the file processing logic to collect and display lines that come after each match while maintaining the performance benefits of streaming processing.

The core challenge is that the current streaming approach processes one line at a time and immediately displays context when a match is found. To show after-context, we need to either:
1. Look ahead in the stream to collect future lines, or  
2. Defer the display of matches until we've collected sufficient after-context

We'll implement a look-ahead buffering approach that maintains performance while providing complete context.

## Architecture

### Current Architecture Issues
- `_process_file_streaming()` processes lines sequentially and immediately displays matches
- `_display_match_with_context()` only has access to previous lines in the rolling buffer
- No mechanism exists to collect lines that come after a match

### Proposed Architecture Changes
- Modify `_process_file_streaming()` to use a look-ahead buffer system
- Implement deferred match display that waits until after-context is collected
- Update `_display_match_with_context()` to handle both before and after context
- Maintain backward compatibility with existing behavior when context = 0

## Components and Interfaces

### Modified Components

#### 1. `_process_file_streaming()` Function
**Current Behavior:**
- Reads lines sequentially
- Immediately displays matches with before-context only
- Uses rolling buffer for before-context

**New Behavior:**
- Maintains a pending matches queue
- Collects after-context for pending matches
- Displays complete matches (before + match + after) when sufficient context is available
- Handles end-of-file scenarios for incomplete after-context

#### 2. `_display_match_with_context()` Function  
**Current Behavior:**
- Displays before-context from rolling buffer
- Displays match line
- No after-context handling

**New Behavior:**
- Displays before-context from rolling buffer
- Displays match line with highlighting
- Displays after-context from provided after-context array
- Maintains existing formatting and spacing

### New Data Structures

#### Pending Match Structure
```perl
{
  line_num => $match_line_num,
  content => $match_line_content,
  before_context => [@before_context_lines],
  after_context => [],  # Will be populated as we read ahead
  after_needed => $context_count
}
```

#### Context Line Structure (existing, enhanced)
```perl
{
  line_num => $line_number,
  content => $line_content
}
```

## Data Models

### File Processing State
- **Rolling Buffer**: Maintains before-context lines (existing)
- **Pending Matches Queue**: Array of matches waiting for complete after-context
- **Current Line**: Currently processed line information
- **Look-ahead Counter**: Tracks how many lines ahead we need to read

### Context Display Model
- **Before Context**: Array of context lines before the match
- **Match Line**: The line containing the search pattern
- **After Context**: Array of context lines after the match
- **Line Numbers**: Preserved for all context and match lines

## Error Handling

### End of File Scenarios
- **Incomplete After-Context**: When a match occurs near the end of file and insufficient after-context lines are available, display all available lines
- **Empty Files**: Handle gracefully without errors
- **Single Line Files**: Display match without context when context is requested but unavailable

### Memory Management
- **Bounded Queues**: Limit pending matches queue size to prevent memory issues with files containing many matches
- **Buffer Limits**: Maintain existing buffer size limits for rolling context buffer
- **Cleanup**: Ensure proper cleanup of pending matches when processing completes

### Performance Considerations
- **Maxline Limit**: Respect existing maxline limits and don't read beyond specified limits even for after-context
- **Large Files**: Maintain streaming approach to avoid loading entire files into memory
- **I/O Efficiency**: Minimize additional file reads through efficient buffering

## Testing Strategy

### Unit Tests
- Test context display with various context values (0, 1, 3, 5)
- Test end-of-file scenarios with incomplete after-context
- Test overlapping matches with context
- Test performance with large files and many matches

### Integration Tests  
- Test context flag with other command-line options (-t, -i, -m)
- Test context display formatting and line numbering
- Test binary file handling with context
- Test permission errors with context

### Edge Case Tests
- Empty files with context requested
- Single-line files with context
- Files where every line matches (stress test)
- Very large context values
- Context with maxline limits

### Regression Tests
- Ensure existing behavior unchanged when context = 0
- Verify all existing command-line combinations still work
- Confirm performance characteristics maintained
- Test backward compatibility with existing scripts

## Implementation Approach

### Phase 1: Core Logic Changes
1. Modify `_process_file_streaming()` to implement pending matches queue
2. Add look-ahead logic to collect after-context lines
3. Implement deferred match display system

### Phase 2: Context Display Enhancement
1. Update `_display_match_with_context()` to handle after-context
2. Ensure proper formatting and spacing maintained
3. Handle overlapping context scenarios

### Phase 3: Edge Case Handling
1. Implement end-of-file handling for incomplete after-context
2. Add memory management for pending matches queue
3. Ensure maxline limits are respected

### Phase 4: Testing and Validation
1. Create comprehensive test suite
2. Performance testing with large files
3. Regression testing for existing functionality
4. Integration testing with all command-line options

## Backward Compatibility

- When context = 0 (default), behavior remains exactly the same
- All existing command-line options continue to work unchanged
- Output format and highlighting preserved
- Performance characteristics maintained for non-context usage
- Existing scripts and automation continue to work without modification