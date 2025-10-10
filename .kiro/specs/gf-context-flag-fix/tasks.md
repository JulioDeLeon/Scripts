# Implementation Plan

- [x] 1. Create test infrastructure for context flag testing
  - Set up test files with known content for context testing
  - Create helper functions to validate context output format
  - Implement test utilities for checking line numbers and content
  - _Requirements: 1.1, 1.2, 1.3, 2.1, 2.2, 2.3_

- [x] 2. Implement pending matches queue system
  - [x] 2.1 Add pending matches data structure to _process_file_streaming
    - Define pending match structure with line_num, content, before_context, after_context, after_needed fields
    - Initialize pending matches array in streaming function
    - _Requirements: 1.1, 1.2, 3.1_

  - [x] 2.2 Modify match detection logic to queue matches instead of immediate display
    - Change match detection to create pending match entries
    - Store before-context from rolling buffer in pending match
    - Set after_needed counter based on context parameter
    - _Requirements: 1.1, 1.2, 2.3_

  - [x] 2.3 Write unit tests for pending matches queue
    - Test pending match creation and structure
    - Test queue management with multiple matches
    - _Requirements: 1.1, 1.2_

- [x] 3. Implement after-context collection logic
  - [x] 3.1 Add after-context collection in main streaming loop
    - Modify line processing to check for pending matches needing after-context
    - Add current line to after_context arrays of relevant pending matches
    - Decrement after_needed counters as lines are collected
    - _Requirements: 1.1, 1.2, 3.1_

  - [x] 3.2 Implement match display triggering when after-context is complete
    - Check if pending matches have collected sufficient after-context
    - Trigger display of complete matches (before + match + after)
    - Remove completed matches from pending queue
    - _Requirements: 1.1, 1.2, 2.3_

  - [x] 3.3 Write unit tests for after-context collection
    - Test after-context collection with various context values
    - Test match completion detection and display triggering
    - _Requirements: 1.1, 1.2_

- [x] 4. Update _display_match_with_context function for after-context
  - [x] 4.1 Modify function signature to accept after-context parameter
    - Add after_context_ref parameter to function signature
    - Update function documentation and parameter descriptions
    - _Requirements: 1.1, 2.1, 2.2_

  - [x] 4.2 Implement after-context display logic
    - Add loop to display after-context lines with proper line numbers
    - Maintain existing formatting and spacing for after-context
    - Ensure proper spacing between matches when context is displayed
    - _Requirements: 1.1, 2.1, 2.2, 2.3_

  - [x] 4.3 Write unit tests for enhanced context display function
    - Test before and after context display with various context values
    - Test formatting and line number display
    - _Requirements: 1.1, 2.1, 2.2_

- [x] 5. Handle end-of-file scenarios for incomplete after-context
  - [x] 5.1 Implement end-of-file detection in streaming loop
    - Detect when file processing is complete
    - Identify pending matches with incomplete after-context
    - _Requirements: 1.4, 3.3_

  - [x] 5.2 Display pending matches with available after-context at EOF
    - Process remaining pending matches when file ends
    - Display matches with whatever after-context is available
    - Ensure proper cleanup of pending matches queue
    - _Requirements: 1.4, 3.3_

  - [x] 5.3 Write unit tests for end-of-file scenarios
    - Test matches near end of file with insufficient after-context
    - Test empty files and single-line files with context
    - _Requirements: 1.4_

- [ ] 6. Implement overlapping context handling
  - [x] 6.1 Add logic to detect overlapping context between consecutive matches
    - Check if after-context of one match overlaps with before-context of next match
    - Implement deduplication logic to avoid displaying same lines multiple times
    - _Requirements: 2.4_

  - [x] 6.2 Optimize context display for overlapping matches
    - Merge overlapping context sections efficiently
    - Maintain proper line numbering and formatting
    - Ensure match highlighting is preserved in overlapping sections
    - _Requirements: 2.4, 2.1, 2.2_

  - [x] 6.3 Write unit tests for overlapping context scenarios
    - Test consecutive matches with overlapping context
    - Test various overlap scenarios and context values
    - _Requirements: 2.4_

- [x] 7. Add memory management and performance optimizations
  - [x] 7.1 Implement bounded pending matches queue
    - Add maximum queue size limit to prevent memory issues
    - Implement queue overflow handling for files with many matches
    - _Requirements: 3.1, 3.2_

  - [x] 7.2 Ensure maxline limits are respected with after-context
    - Modify after-context collection to respect maxline parameter
    - Ensure we don't read beyond maxline limit even for context
    - Handle cases where maxline limit prevents complete after-context collection
    - _Requirements: 3.3, 1.4_

  - [x] 7.3 Write performance tests for memory management
    - Test large files with many matches and context
    - Test memory usage with bounded queues
    - _Requirements: 3.1, 3.2_

- [x] 8. Update function call sites and maintain backward compatibility
  - [x] 8.1 Update all calls to _display_match_with_context with new signature
    - Find all call sites of the function
    - Update calls to pass after-context parameter (empty array for existing calls)
    - _Requirements: 4.1, 4.2_

  - [x] 8.2 Ensure zero context behavior remains unchanged
    - Verify that context=0 (default) produces identical output to current implementation
    - Test that no performance regression occurs for non-context usage
    - _Requirements: 4.2, 4.3_

  - [x] 8.3 Write regression tests for backward compatibility
    - Test all existing command-line option combinations
    - Test default behavior (no context) remains unchanged
    - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [ ] 9. Create comprehensive integration tests
  - [x] 9.1 Test context flag with other command-line options
    - Test context with target patterns (-t)
    - Test context with ignore patterns (-i)
    - Test context with maxline limits (-m)
    - Test context with debug output (-d)
    - _Requirements: 4.4_

  - [x] 9.2 Test context display formatting and edge cases
    - Test various context values (1, 3, 5, 10)
    - Test context with binary file detection
    - Test context with permission errors
    - Test context with empty and single-line files
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.1, 2.2_

  - [x] 9.3 Write performance benchmarks for context feature
    - Benchmark context vs non-context performance
    - Test performance with large files and high context values
    - _Requirements: 3.1, 3.2_