# Design Document

## Overview

This design document outlines the architectural changes needed to optimize the gf file search tool's performance. The optimization focuses on four key areas: memory efficiency through streaming file processing, pattern matching optimization through compiled regex caching, directory traversal improvements by eliminating expensive system calls, and redundant operation elimination through better ordering and caching.

The design maintains complete backward compatibility while dramatically improving performance for large files and directory structures.

## Architecture

### Current Architecture Issues

The current implementation suffers from several architectural problems:
- **Memory-intensive approach**: Loads entire files into `@fileContext` arrays
- **Inefficient pattern matching**: Uses Perl's `/p` flag with expensive special variables
- **Expensive directory operations**: Relies on `chdir()` calls and repeated path construction
- **Poor operation ordering**: Performs expensive operations before cheaper validation checks

### Optimized Architecture

The new architecture implements a streaming, cache-optimized approach:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Startup       │    │   Directory      │    │   File          │
│   - Compile     │───▶│   Traversal      │───▶│   Processing    │
│     patterns    │    │   - Path-based   │    │   - Streaming   │
│   - Cache       │    │   - No chdir()   │    │   - Early exit  │
│     config      │    │   - Fast skip    │    │   - Buffered    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Components and Interfaces

### 1. Pattern Compilation Cache

**Purpose**: Compile and cache all regex patterns at startup to avoid repeated compilation.

**Interface**:
```perl
# Global compiled pattern cache
my $search_pattern;           # Compiled search regex
my @ignore_patterns;          # Array of compiled ignore regexes  
my @target_patterns;          # Array of compiled target regexes

sub compile_patterns {
    # Compile search pattern once
    $search_pattern = qr/$term/;
    
    # Compile ignore patterns
    @ignore_patterns = map { qr/$_/i } keys %ignores;
    
    # Compile target patterns  
    @target_patterns = map { qr/$_/i } keys %targets;
}
```

### 2. Streaming File Processor

**Purpose**: Process files line-by-line without loading entire contents into memory.

**Interface**:
```perl
sub process_file_streaming {
    my ($filepath) = @_;
    my $line_num = 0;
    my @context_buffer;  # Rolling buffer for context lines
    my $found_match = 0;
    
    # Process file line by line
    while (my $line = <$fh>) {
        $line_num++;
        
        # Check for match using compiled pattern
        if ($line =~ /$search_pattern/) {
            # Handle match with context from buffer
            display_match_with_context($filepath, $line_num, $line, \@context_buffer);
            $found_match = 1;
        }
        
        # Maintain rolling context buffer
        manage_context_buffer(\@context_buffer, $line);
        
        # Early exit for maxline limit
        last if $maxline && $line_num >= $maxline;
    }
}
```

### 3. Path-Based Directory Traversal

**Purpose**: Eliminate expensive `chdir()` operations by working with absolute paths.

**Interface**:
```perl
sub traverse_directory_optimized {
    my ($dir_path) = @_;
    
    opendir(my $dh, $dir_path) or return;
    
    while (my $entry = readdir($dh)) {
        next if $entry =~ /^\.\.?$/;  # Skip . and ..
        
        my $full_path = File::Spec->catfile($dir_path, $entry);
        
        # Fast skip check before expensive operations
        next if should_skip_fast($full_path);
        
        # Process based on file type
        if (-d $full_path) {
            traverse_directory_optimized($full_path);
        } elsif (-f $full_path) {
            process_file_optimized($full_path);
        }
    }
    
    closedir($dh);
}
```

### 4. Optimized Pattern Matching and Output

**Purpose**: Replace recursive `printStr()` with efficient single-pass highlighting and implement buffered output.

**Interface**:
```perl
sub highlight_matches_efficient {
    my ($line) = @_;
    
    # Use split with compiled pattern to avoid /p flag overhead
    my @parts = split(/($search_pattern)/, $line);
    
    my $output = '';
    for (my $i = 0; $i < @parts; $i++) {
        if ($i % 2 == 1) {  # Odd indices are matches
            $output .= color("bold red") . $parts[$i] . color("reset");
        } else {
            $output .= $parts[$i];
        }
    }
    
    return $output;
}

# Buffered output for performance
my @output_buffer;
sub buffer_output {
    my ($text) = @_;
    push @output_buffer, $text;
    
    # Flush buffer when it reaches optimal size
    if (@output_buffer >= 50) {
        print join('', @output_buffer);
        @output_buffer = ();
    }
}
```

### 5. Early Validation Pipeline

**Purpose**: Perform cheap checks before expensive operations to fail fast.

**Interface**:
```perl
# Cache for file stat information to avoid repeated system calls
my %stat_cache;

sub should_skip_fast {
    my ($filepath) = @_;
    
    # 1. Quick extension check (cheapest)
    return 1 if quick_extension_check($filepath);
    
    # 2. Pattern matching with compiled regexes
    return 1 if matches_ignore_patterns($filepath);
    return 1 if !matches_target_patterns($filepath);
    
    # 3. Cached file system checks (more expensive)
    my $stat_info = get_cached_stat($filepath);
    return 1 if !$stat_info;           # File doesn't exist
    return 1 if !-r _;                 # Not readable (using cached stat)
    return 1 if -B _;                  # Binary file (using cached stat)
    
    return 0;  # File should be processed
}

sub get_cached_stat {
    my ($filepath) = @_;
    return $stat_cache{$filepath} //= [stat($filepath)];
}
```

## Data Models

### Context Buffer Management

Instead of loading entire files, maintain a rolling buffer for context lines:

```perl
# Context buffer structure
my @context_buffer = (
    { line_num => $num, content => $line },
    # ... up to $context lines
);

sub manage_context_buffer {
    my ($buffer_ref, $new_line) = @_;
    
    push @$buffer_ref, { 
        line_num => $current_line_num, 
        content => $new_line 
    };
    
    # Keep buffer size limited to context requirement
    shift @$buffer_ref if @$buffer_ref > $context;
}
```

### File Processing State

Track processing state without loading full file contents:

```perl
my %file_state = (
    current_line => 0,
    matches_found => 0,
    last_match_line => 0,
    context_needed => 0,
    binary_detected => 0,
    permission_checked => 0
);
```

### Configuration Cache

Cache compiled configuration to avoid repeated parsing:

```perl
my %config_cache = (
    ignore_patterns => [],
    target_patterns => [],
    max_line_limit => 0,
    context_lines => 0,
    compiled => 0
);

sub load_and_cache_config {
    return if $config_cache{compiled};
    
    # Load configuration once at startup
    parse_config_files();
    compile_patterns();
    $config_cache{compiled} = 1;
}

## Error Handling

### Graceful Degradation

- **File access errors**: Continue processing other files, log warnings
- **Pattern compilation errors**: Validate patterns at startup, fail early with clear messages
- **Memory constraints**: Streaming approach naturally handles large files
- **Permission issues**: Check permissions before file operations, provide helpful error messages

### Error Recovery Strategies

```perl
sub safe_file_processing {
    my ($filepath) = @_;
    
    eval {
        process_file_streaming($filepath);
    };
    
    if ($@) {
        warn "Error processing $filepath: $@";
        # Continue with next file
    }
}
```

## Testing Strategy

### Performance Testing

1. **Memory Usage Tests**
   - Compare memory consumption before/after optimization
   - Test with files of varying sizes (1MB, 10MB, 100MB+)
   - Monitor peak memory usage during processing

2. **Speed Benchmarks**
   - Measure search time across different directory sizes
   - Compare pattern matching performance
   - Test directory traversal speed improvements

3. **Regression Testing**
   - Verify identical output format for all existing test cases
   - Test all command-line options maintain same behavior
   - Validate configuration file processing remains unchanged

### Test Implementation Approach

```perl
# Performance comparison framework
sub benchmark_search {
    my ($test_dir, $pattern) = @_;
    
    my $start_time = time();
    my $start_memory = get_memory_usage();
    
    # Run optimized search
    run_search($test_dir, $pattern);
    
    my $end_time = time();
    my $end_memory = get_memory_usage();
    
    return {
        duration => $end_time - $start_time,
        memory_used => $end_memory - $start_memory
    };
}
```

### Compatibility Testing

- **Output Format Validation**: Ensure search results, line numbers, and context display remain identical
- **Configuration File Support**: Test all .gfconf directives work as before  
- **Command-Line Interface**: Verify all options produce expected behavior
- **Error Message Consistency**: Maintain same error messages and exit codes