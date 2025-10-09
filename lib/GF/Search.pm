package GF::Search;

use strict;
use warnings;
use Exporter qw(import);
use Term::ANSIColor;
use File::stat;
use Cwd;
use GF::PatternCache qw(get_search_pattern get_ignore_patterns get_target_patterns);

our @EXPORT_OK = qw(
  process_args print_targets should_skip
  handle_dir check_file print_str main_search is_binary_file
  buffer_output flush_output_buffer
  _traverse_directory_optimized _construct_path_optimized _should_skip_fast _quick_extension_check
  _validate_file_fast _get_cached_stat _check_file_permissions _is_binary_cached
  _clear_validation_cache _get_cache_stats _optimize_cache_size
);

# Buffered output system to minimize I/O operations
my @output_buffer = ();
my $buffer_size_limit = 50;  # Flush buffer when it reaches this size

# File metadata cache to avoid repeated system calls
my %stat_cache = ();
my %permission_cache = ();
my %binary_cache = ();

# Cache statistics for performance monitoring
my %cache_stats = (
  stat_hits => 0,
  stat_misses => 0,
  permission_hits => 0,
  permission_misses => 0,
  binary_hits => 0,
  binary_misses => 0
);

=head1 NAME

GF::Search - Core search functionality for the gf file search tool

=head1 SYNOPSIS

    use GF::Search qw(process_args main_search);
    
    my ($term, $ignores_ref, $targets_ref, $context, $maxline) = process_args();
    main_search($term, $ignores_ref, $targets_ref, $context, $maxline, $debug);

=head1 DESCRIPTION

This module contains the core search logic for the gf tool, including argument
processing, file filtering, directory traversal, and pattern matching.

=cut

=head2 buffer_output

Adds text to the output buffer to minimize I/O operations.
Automatically flushes when buffer reaches optimal size.

Arguments:
  - $text: Text to add to buffer

=cut

sub buffer_output {
  my ($text) = @_;
  
  push @output_buffer, $text;
  
  # Flush buffer when it reaches optimal size for performance
  if (@output_buffer >= $buffer_size_limit) {
    flush_output_buffer();
  }
}

=head2 flush_output_buffer

Flushes the output buffer to STDOUT and clears the buffer.
Should be called at the end of processing to ensure all output is displayed.

=cut

sub flush_output_buffer {
  if (@output_buffer) {
    print join('', @output_buffer);
    STDOUT->flush() if STDOUT->can('flush');  # Ensure output is flushed
    @output_buffer = ();
  }
}

sub print_targets {
  my ($ignores_ref, $targets_ref, $debug) = @_;
  if ($debug) {
    print Term::ANSIColor::color("magenta");
    print "ign: " . join(" ", keys %$ignores_ref) . "\n";
    print "tar: " . join(" ", keys %$targets_ref) . "\n";
    print Term::ANSIColor::color("reset");
  }
}

=head2 process_args

Processes command line arguments and returns parsed values.

Returns:
  - $term: Search pattern
  - $ignores_ref: Hash reference of ignore patterns
  - $targets_ref: Hash reference of target patterns
  - $context: Number of context lines
  - $maxline: Maximum lines to read per file
  - $debug: Debug flag

=cut

sub process_args {
  require Getopt::Long;
  require GF::Help;
  
  my $retTerm = undef;
  my %retIgnores;
  my %retTargets;
  my @ignoresArr;
  my @targetsArr;
  my $show_help = 0;
  my $show_version = 0;
  my $context = 0;
  my $maxline = 0;
  my $debug = 0;
  
  Getopt::Long::GetOptions(
    "search=s" => \$retTerm,
    "target=s" => \@targetsArr,
    "ignore=s" => \@ignoresArr,
    "context=i" => \$context,
    "maxline=i" => \$maxline,
    "debug" => \$debug,
    "help|h" => \$show_help,
    "version|V" => \$show_version
  ) or GF::Help::show_usage_and_exit();
  
  # Handle help and version flags - these take precedence over other processing
  if ($show_help) {
    GF::Help::show_help();
    exit 0;
  }
  
  if ($show_version) {
    GF::Help::show_version();
    exit 0;
  }
  
  if ($debug) {
    print Term::ANSIColor::color("magenta");
    print "retTerm: " . (defined $retTerm ? $retTerm : "undefined") . "\n";
    print "ignoreArr: " . join(" ", @ignoresArr) . "\n";
    print "targetArr: " . join(" ", @targetsArr) . "\n";
    print Term::ANSIColor::color("reset");
  }
  
  # Check if search term was provided
  if (! defined $retTerm) {
    print STDERR "Error: No search pattern provided.\n";
    print STDERR "\n";
    GF::Help::show_usage();
    exit 1;
  }

  # Allow commas in targets and ignores 
  @ignoresArr = split(/,/, join(',', @ignoresArr));
  @targetsArr = split(/,/, join(',', @targetsArr));

  foreach my $ign (@ignoresArr) {
    $retIgnores{$ign} = 1;
  }

  foreach my $tar (@targetsArr) {
    $retTargets{$tar} = 1;
  }
  
  return ($retTerm, \%retIgnores, \%retTargets, $context, $maxline, $debug);
}

=head2 print_str

Prints a string with search term highlighted in red using efficient single-pass processing.
Replaces recursive highlighting with split-based pattern matching to avoid Perl /p flag overhead.

Arguments:
  - $str: String to print
  - $term: Search pattern to highlight (kept for compatibility)

=cut

sub print_str {
  my ($str, $term) = @_;
  
  # Use compiled search pattern from cache for better performance
  my $search_pattern = get_search_pattern();
  
  # Use split with compiled pattern to avoid /p flag overhead and capture variables
  # This processes all matches in a single pass instead of recursive calls
  my @parts = split(/($search_pattern)/, $str);
  
  # Build output string with efficient colorization that doesn't impact performance
  my $output = '';
  for (my $i = 0; $i < @parts; $i++) {
    if ($i % 2 == 1) {  # Odd indices contain the matches
      $output .= Term::ANSIColor::color("bold red") . $parts[$i] . Term::ANSIColor::color("reset");
    } else {
      $output .= $parts[$i];
    }
  }
  
  # Use buffered output to minimize I/O operations
  buffer_output($output);
}

=head2 is_binary_file

Detects if a file is binary before opening it for processing to avoid unnecessary operations.

Arguments:
  - $filepath: Path to the file to check

Returns:
  - 1 if file is binary, 0 if text file

=cut

sub is_binary_file {
  my ($filepath) = @_;
  
  # Return 0 for non-existent files
  return 0 unless -e $filepath;
  
  # Return 0 for empty files (they're not binary)
  return 0 if -z $filepath;
  
  # Use Perl's built-in binary file test
  return (-B $filepath) ? 1 : 0;
}

=head2 check_file

Searches for the pattern within a file and prints matches with context using streaming processing.
Includes early binary file detection to skip files before reading content.

Arguments:
  - $fn: File path
  - $term: Search pattern
  - $context: Number of context lines
  - $maxline: Maximum lines to read
  - $debug: Debug flag

=cut

sub check_file {
  my ($fn, $term, $context, $maxline, $debug) = @_;
  
  if ($debug) {
    print Term::ANSIColor::color("magenta");
    print "in checkFile (streaming)\n";
    print Term::ANSIColor::color("reset");
  }
  
  # Check if file exists first and provide helpful error message
  if (!-e $fn) {
    print STDERR "Warning: Could not open file '$fn': No such file or directory\n";
    return;
  }
  
  # Early binary detection using cached results to skip files before reading content
  if (_is_binary_cached($fn, $debug)) {
    if ($debug) {
      print Term::ANSIColor::color("magenta");
      print "skipping binary file: $fn\n";
      print Term::ANSIColor::color("reset");
    }
    return;
  }
  
  my $fh;
  if (!open($fh, "<", $fn)) {
    print STDERR "Warning: Could not open file '$fn': $!\n";
    return;
  }
  
  # Use streaming file processing
  _process_file_streaming($fh, $fn, $term, $context, $maxline, $debug);
  
  close $fh;
}

=head2 _process_file_streaming

Internal function that processes a file using streaming approach with rolling context buffer.

Arguments:
  - $fh: File handle
  - $fn: File path (for display)
  - $term: Search pattern
  - $context: Number of context lines
  - $maxline: Maximum lines to read
  - $debug: Debug flag

=cut

sub _process_file_streaming {
  my ($fh, $fn, $term, $context, $maxline, $debug) = @_;
  
  my $line_num = 0;
  my $found_match = 0;
  my @context_buffer = ();  # Rolling buffer for context lines
  my $search_pattern = get_search_pattern();
  
  while (my $line = <$fh>) {
    chomp($line);
    $line_num++;
    
    # Check for match using compiled pattern
    if ($line =~ /$search_pattern/) {
      # Print filename header on first match using buffered output
      if (!$found_match) {
        $found_match = 1;
        buffer_output("$fn\n");
      }
      
      # Display match with context
      _display_match_with_context(\@context_buffer, $line, $line_num, $term, $context);
    }
    
    # Maintain rolling context buffer
    _manage_context_buffer(\@context_buffer, $line, $line_num, $context);
    
    # Early exit for maxline limit to avoid unnecessary processing
    last if ($maxline && $line_num >= $maxline);
  }
  
  if ($found_match) {
    buffer_output("\n");
    # Flush buffer to ensure output is displayed
    flush_output_buffer();
  }
}

=head2 _manage_context_buffer

Manages the rolling context buffer to maintain search context without loading entire files.

Arguments:
  - $buffer_ref: Reference to context buffer array
  - $line: Current line content
  - $line_num: Current line number
  - $context: Number of context lines to maintain

=cut

sub _manage_context_buffer {
  my ($buffer_ref, $line, $line_num, $context) = @_;
  
  # Add current line to buffer
  push @$buffer_ref, {
    line_num => $line_num,
    content => $line
  };
  
  # Keep buffer size limited to context requirement
  # We need context lines before + current line + context lines after
  my $max_buffer_size = ($context * 2) + 1;
  
  # Remove oldest entries if buffer exceeds maximum size
  while (@$buffer_ref > $max_buffer_size) {
    shift @$buffer_ref;
  }
}

=head2 _display_match_with_context

Displays a match with appropriate context lines from the rolling buffer.

Arguments:
  - $buffer_ref: Reference to context buffer array
  - $match_line: The line containing the match
  - $match_line_num: Line number of the match
  - $term: Search pattern for highlighting
  - $context: Number of context lines to display

=cut

sub _display_match_with_context {
  my ($buffer_ref, $match_line, $match_line_num, $term, $context) = @_;
  
  if ($context) {
    # Display context before match from buffer without re-reading file sections
    my $start_context_line = $match_line_num - $context;
    
    foreach my $entry (@$buffer_ref) {
      if ($entry->{line_num} >= $start_context_line && $entry->{line_num} < $match_line_num) {
        buffer_output("[$entry->{line_num}]\t$entry->{content}\n");
      }
    }
  } else {
    # Trim whitespace if no context requested
    $match_line =~ s/^\s+|\s+$//g;
  }
  
  # Display the match line with highlighting using buffered output
  buffer_output("[$match_line_num]\t");
  print_str($match_line, $term);
  buffer_output("\n");
  
  if ($context) {
    # Add spacing for context display
    buffer_output("\n");
  }
}

=head2 should_skip

Determines if a file should be skipped based on ignore and target patterns.
Uses compiled patterns and optimized evaluation order for fast failure.

Arguments:
  - $entry: File path to check
  - $ignores_ref: Hash reference of ignore patterns (kept for compatibility)
  - $targets_ref: Hash reference of target patterns (kept for compatibility)
  - $debug: Debug flag

Returns:
  - 1 if file should be skipped, 0 otherwise

=cut

sub should_skip {
  my ($entry, $ignores_ref, $targets_ref, $debug) = @_;
  
  if ($debug) {
    print Term::ANSIColor::color("magenta");
    print "should skip $entry?\n";
    print Term::ANSIColor::color("reset");
  }
  
  # Use compiled patterns from cache for better performance
  my $ignore_patterns = get_ignore_patterns();
  my $target_patterns = get_target_patterns();
  
  # Optimized pattern evaluation order for fast failure:
  # 1. Check ignore patterns first (most likely to match and fail fast)
  if (_matches_ignore_patterns_optimized($entry, $ignore_patterns, $debug)) {
    if ($debug) {
      print Term::ANSIColor::color("magenta");
      print "\tmatched ignore pattern, skipping\n";
      print Term::ANSIColor::color("reset");
    }
    return 1;
  }
  
  # 2. Check target patterns only if we have any defined
  if (@$target_patterns && !_matches_target_patterns_optimized($entry, $target_patterns, $debug)) {
    if ($debug) {
      print Term::ANSIColor::color("magenta");
      print "\tno target pattern matched, skipping\n";
      print Term::ANSIColor::color("reset");
    }
    return 1;
  }
  
  if ($debug) {
    print Term::ANSIColor::color("magenta");
    print "\tnot skipping file\n";
    print Term::ANSIColor::color("reset");
  }
  return 0;
}

=head2 _matches_ignore_patterns_optimized

Optimized ignore pattern matching with early exit for performance.
Evaluates patterns in order optimized for fast failure.

Arguments:
  - $entry: File path to check
  - $ignore_patterns: Array reference of compiled ignore patterns
  - $debug: Debug flag

Returns:
  - 1 if entry matches any ignore pattern, 0 otherwise

=cut

sub _matches_ignore_patterns_optimized {
  my ($entry, $ignore_patterns, $debug) = @_;
  
  # Early exit if no ignore patterns defined
  return 0 unless @$ignore_patterns;
  
  # Check each compiled pattern - exit immediately on first match
  foreach my $pattern (@$ignore_patterns) {
    if ($debug) {
      print Term::ANSIColor::color("magenta");
      print "\tchecking ignore pattern against $entry\n";
      print Term::ANSIColor::color("reset");
    }
    
    if ($entry =~ /$pattern/) {
      return 1;  # Fast failure - found a match
    }
  }
  
  return 0;  # No ignore patterns matched
}

=head2 _matches_target_patterns_optimized

Optimized target pattern matching with early exit for performance.
Evaluates patterns in order optimized for fast success.

Arguments:
  - $entry: File path to check
  - $target_patterns: Array reference of compiled target patterns
  - $debug: Debug flag

Returns:
  - 1 if entry matches any target pattern, 0 otherwise

=cut

sub _matches_target_patterns_optimized {
  my ($entry, $target_patterns, $debug) = @_;
  
  # If no target patterns defined, everything matches
  return 1 unless @$target_patterns;
  
  # Check each compiled pattern - exit immediately on first match
  foreach my $pattern (@$target_patterns) {
    if ($debug) {
      print Term::ANSIColor::color("magenta");
      print "\tchecking target pattern against $entry\n";
      print Term::ANSIColor::color("reset");
    }
    
    if ($entry =~ /$pattern/) {
      return 1;  # Fast success - found a match
    }
  }
  
  return 0;  # No target patterns matched
}



=head2 handle_dir

Recursively processes directories using optimized path-based traversal.
Replaces directory change operations with absolute path handling and implements
path construction optimization to avoid repeated string operations.

Arguments:
  - $dn: Directory path
  - $term: Search pattern
  - $ignores_ref: Hash reference of ignore patterns
  - $targets_ref: Hash reference of target patterns
  - $seen_ref: Hash reference of seen files
  - $context: Number of context lines
  - $maxline: Maximum lines to read
  - $debug: Debug flag

=cut

sub handle_dir {
  my ($dn, $term, $ignores_ref, $targets_ref, $seen_ref, $context, $maxline, $debug) = @_;
  
  # Use optimized path-based directory traversal
  _traverse_directory_optimized($dn, $term, $ignores_ref, $targets_ref, $seen_ref, $context, $maxline, $debug);
}

=head2 _traverse_directory_optimized

Internal optimized directory traversal that avoids chdir() operations and works with absolute paths.
Implements path construction optimization to avoid repeated string operations.

Arguments:
  - $dir_path: Directory path to traverse
  - $term: Search pattern
  - $ignores_ref: Hash reference of ignore patterns
  - $targets_ref: Hash reference of target patterns
  - $seen_ref: Hash reference of seen files
  - $context: Number of context lines
  - $maxline: Maximum lines to read
  - $debug: Debug flag

=cut

sub _traverse_directory_optimized {
  my ($dir_path, $term, $ignores_ref, $targets_ref, $seen_ref, $context, $maxline, $debug) = @_;
  
  my $dh;
  if (!opendir($dh, $dir_path)) {
    print STDERR "Error: Could not open directory '$dir_path': $!\n";
    print STDERR "This directory will be skipped. Check permissions or path validity.\n";
    return;
  }
  
  # Process directory entries without changing directories
  while (my $entry = readdir($dh)) {
    # Skip . and .. entries
    next if $entry =~ /^\.\.?$/;
    
    # Build absolute path once and reuse throughout traversal
    my $full_path = _construct_path_optimized($dir_path, $entry);
    
    if ($debug) {
      print Term::ANSIColor::color("magenta");
      print "processing path -> $full_path\n";
      print Term::ANSIColor::color("reset");
    }
    
    # Skip if already seen (avoid loops)
    next if exists $seen_ref->{$full_path};
    
    # Fast skip check before expensive operations
    next if _should_skip_fast($full_path, $ignores_ref, $targets_ref, $debug);
    
    # Check permissions using cached validation
    if (!_check_file_permissions($full_path, $debug)) {
      print Term::ANSIColor::color("bold yellow");
      print "Warning: Permission denied accessing '$full_path'\n";
      print "Tip: You may need to run with different permissions or check file ownership.\n";
      print Term::ANSIColor::color("reset");
      next;
    }
    
    # Handle symbolic links
    if (-l $full_path) {
      my $link_target = readlink($full_path);
      next if !defined $link_target || $link_target eq ".";
      
      # Convert relative links to absolute paths
      if ($link_target !~ /^\//) {
        $link_target = _construct_path_optimized($dir_path, $link_target);
      }
      
      # Track both the link and its target to avoid loops
      $seen_ref->{$full_path} = 1;
      $seen_ref->{$link_target} = 1;
      next; # Skip processing symbolic links directly
    }
    
    # Mark as seen
    $seen_ref->{$full_path} = 1;
    
    # Process based on file type using absolute paths
    if (-d $full_path) {
      # Recursively traverse subdirectory
      _traverse_directory_optimized($full_path, $term, $ignores_ref, $targets_ref, $seen_ref, $context, $maxline, $debug);
    } elsif (-f $full_path) {
      # Process file
      check_file($full_path, $term, $context, $maxline, $debug);
    }
  }
  
  closedir($dh);
}

=head2 _construct_path_optimized

Optimized path construction that builds absolute paths efficiently.
Avoids repeated string operations by constructing paths once and reusing them.

Arguments:
  - $dir_path: Directory path
  - $entry: File/directory entry name

Returns:
  - Optimized absolute path

=cut

sub _construct_path_optimized {
  my ($dir_path, $entry) = @_;
  
  # Use File::Spec for portable path construction
  require File::Spec;
  my $full_path = File::Spec->catfile($dir_path, $entry);
  
  # Normalize path to avoid double slashes and other issues
  $full_path =~ s/\/+/\//g;  # Replace multiple slashes with single slash
  
  return $full_path;
}

=head2 _should_skip_fast

Enhanced fast skip check that performs cheap validation before expensive operations.
Uses optimal pattern evaluation ordering and cached file metadata for performance.

Arguments:
  - $filepath: Full file path to check
  - $ignores_ref: Hash reference of ignore patterns
  - $targets_ref: Hash reference of target patterns
  - $debug: Debug flag

Returns:
  - 1 if file should be skipped, 0 otherwise

=cut

sub _should_skip_fast {
  my ($filepath, $ignores_ref, $targets_ref, $debug) = @_;
  
  # For directories, only check ignore patterns (not target patterns)
  # Directories should be traversed unless explicitly ignored
  if (-d $filepath) {
    my $ignore_patterns = get_ignore_patterns();
    return _matches_ignore_patterns_optimized($filepath, $ignore_patterns, $debug);
  }
  
  # For files, use enhanced fast validation pipeline with optimal ordering:
  
  # 1. Pattern matching first (cheapest - compiled regex operations)
  my $ignore_patterns = get_ignore_patterns();
  my $target_patterns = get_target_patterns();
  
  # Check ignore patterns first (most likely to match and cause skip)
  if (_matches_ignore_patterns_optimized($filepath, $ignore_patterns, $debug)) {
    if ($debug) {
      print Term::ANSIColor::color("magenta");
      print "skipping file due to ignore pattern: $filepath\n";
      print Term::ANSIColor::color("reset");
    }
    return 1;
  }
  
  # Check target patterns only if defined
  if (@$target_patterns && !_matches_target_patterns_optimized($filepath, $target_patterns, $debug)) {
    if ($debug) {
      print Term::ANSIColor::color("magenta");
      print "skipping file - no target pattern match: $filepath\n";
      print Term::ANSIColor::color("reset");
    }
    return 1;
  }
  
  # 2. Enhanced file validation with caching (more expensive but optimized)
  if (_validate_file_fast($filepath, $debug)) {
    if ($debug) {
      print Term::ANSIColor::color("magenta");
      print "skipping file due to validation failure: $filepath\n";
      print Term::ANSIColor::color("reset");
    }
    return 1;
  }
  
  return 0;  # File should be processed
}

=head2 _quick_extension_check

Performs quick file extension checks to skip obviously unwanted files.
This is the cheapest check and should be performed first.

Arguments:
  - $filepath: File path to check

Returns:
  - 1 if file should be skipped based on extension, 0 otherwise

=cut

sub _quick_extension_check {
  my ($filepath) = @_;
  
  # Skip common binary file extensions that we definitely don't want to search
  # This is a fast string operation that avoids more expensive pattern matching
  if ($filepath =~ /\.(exe|dll|so|dylib|bin|obj|o|a|lib|jar|war|ear|class|pyc|pyo|pyd)$/i) {
    return 1;
  }
  
  # Skip image files
  if ($filepath =~ /\.(jpg|jpeg|png|gif|bmp|tiff|ico|svg|webp)$/i) {
    return 1;
  }
  
  # Skip video/audio files
  if ($filepath =~ /\.(mp4|avi|mov|wmv|flv|mp3|wav|flac|ogg|m4a)$/i) {
    return 1;
  }
  
  # Skip archive files
  if ($filepath =~ /\.(zip|rar|7z|tar|gz|bz2|xz|lzma)$/i) {
    return 1;
  }
  
  return 0;  # Extension looks fine for text searching
}

=head2 _validate_file_fast

Enhanced fast-fail validation system that performs file extension checking,
permission validation, and binary detection in optimal order before expensive operations.

Arguments:
  - $filepath: Full file path to validate
  - $debug: Debug flag

Returns:
  - 1 if file should be skipped, 0 if file is valid for processing

=cut

sub _validate_file_fast {
  my ($filepath, $debug) = @_;
  
  if ($debug) {
    print Term::ANSIColor::color("magenta");
    print "fast validation for: $filepath\n";
    print Term::ANSIColor::color("reset");
  }
  
  # 1. Quick extension check (cheapest - string comparison only)
  if (_quick_extension_check($filepath)) {
    if ($debug) {
      print Term::ANSIColor::color("magenta");
      print "  failed extension check\n";
      print Term::ANSIColor::color("reset");
    }
    return 1;
  }
  
  # 2. File existence and basic stat check (cached)
  my $stat_info = _get_cached_stat($filepath);
  if (!$stat_info) {
    if ($debug) {
      print Term::ANSIColor::color("magenta");
      print "  file does not exist\n";
      print Term::ANSIColor::color("reset");
    }
    return 1;
  }
  
  # 3. Permission validation (cached)
  if (!_check_file_permissions($filepath, $debug)) {
    if ($debug) {
      print Term::ANSIColor::color("magenta");
      print "  failed permission check\n";
      print Term::ANSIColor::color("reset");
    }
    return 1;
  }
  
  # 4. Binary file detection (cached)
  if (_is_binary_cached($filepath, $debug)) {
    if ($debug) {
      print Term::ANSIColor::color("magenta");
      print "  detected as binary file\n";
      print Term::ANSIColor::color("reset");
    }
    return 1;
  }
  
  if ($debug) {
    print Term::ANSIColor::color("magenta");
    print "  passed all validation checks\n";
    print Term::ANSIColor::color("reset");
  }
  
  return 0;  # File is valid for processing
}

=head2 _get_cached_stat

Implements stat information caching to avoid repeated system calls.
Caches file metadata for performance optimization.

Arguments:
  - $filepath: File path to stat

Returns:
  - Array reference of stat information, or undef if file doesn't exist

=cut

sub _get_cached_stat {
  my ($filepath) = @_;
  
  # Return cached result if available
  if (exists $stat_cache{$filepath}) {
    $cache_stats{stat_hits}++;
    return $stat_cache{$filepath};
  }
  
  # Cache miss - perform stat and cache the result as a plain array reference
  $cache_stats{stat_misses}++;
  my @stat_info = CORE::stat($filepath);
  
  if (@stat_info) {
    # Ensure we store a plain array reference, not a File::stat object
    my @plain_stat = @stat_info;
    $stat_cache{$filepath} = \@plain_stat;
    return \@plain_stat;
  } else {
    # Cache negative result to avoid repeated failed stat calls
    $stat_cache{$filepath} = undef;
    return undef;
  }
}

=head2 _check_file_permissions

Cached permission validation to check file access before attempting operations.
Caches permission information to optimize file validation pipeline.

Arguments:
  - $filepath: File path to check
  - $debug: Debug flag

Returns:
  - 1 if file is readable, 0 otherwise

=cut

sub _check_file_permissions {
  my ($filepath, $debug) = @_;
  
  # Return cached result if available
  if (exists $permission_cache{$filepath}) {
    $cache_stats{permission_hits}++;
    return $permission_cache{$filepath};
  }
  
  # Cache miss - check if file is readable
  $cache_stats{permission_misses}++;
  my $is_readable = (-r $filepath) ? 1 : 0;
  
  # Cache the result
  $permission_cache{$filepath} = $is_readable;
  
  if ($debug && !$is_readable) {
    print Term::ANSIColor::color("magenta");
    print "  permission denied for: $filepath\n";
    print Term::ANSIColor::color("reset");
  }
  
  return $is_readable;
}

=head2 _is_binary_cached

Cached binary file detection that uses file metadata to determine if file is binary.
Caches binary detection results to avoid repeated expensive checks.

Arguments:
  - $filepath: File path to check
  - $debug: Debug flag

Returns:
  - 1 if file is binary, 0 if text file

=cut

sub _is_binary_cached {
  my ($filepath, $debug) = @_;
  
  # Return cached result if available
  if (exists $binary_cache{$filepath}) {
    $cache_stats{binary_hits}++;
    return $binary_cache{$filepath};
  }
  
  # Cache miss - perform binary detection
  $cache_stats{binary_misses}++;
  
  # Get cached stat info (avoids additional system call)
  my $stat_info = _get_cached_stat($filepath);
  if (!$stat_info) {
    $binary_cache{$filepath} = 1;  # Treat non-existent files as binary
    if ($debug) {
      print Term::ANSIColor::color("magenta");
      print "  no stat info, treating as binary: $filepath\n";
      print Term::ANSIColor::color("reset");
    }
    return 1;
  }
  

  
  # Check if file is empty (empty files are treated as text for searching)
  if (defined $stat_info->[7] && $stat_info->[7] == 0) {  # Size is 0
    $binary_cache{$filepath} = 0;

    return 0;
  }
  
  # Use Perl's built-in binary file test
  # For non-empty files, -B should give us a reliable binary detection
  my $is_binary = (-B $filepath) ? 1 : 0;
  
  # Cache the result
  $binary_cache{$filepath} = $is_binary;
  
  if ($debug && $is_binary) {
    print Term::ANSIColor::color("magenta");
    print "  detected binary file: $filepath\n";
    print Term::ANSIColor::color("reset");
  }
  
  return $is_binary;
}

=head2 _clear_validation_cache

Clears all validation caches. Useful for testing or when file system state may have changed.

=cut

sub _clear_validation_cache {
  %stat_cache = ();
  %permission_cache = ();
  %binary_cache = ();
  
  # Reset cache statistics
  %cache_stats = (
    stat_hits => 0,
    stat_misses => 0,
    permission_hits => 0,
    permission_misses => 0,
    binary_hits => 0,
    binary_misses => 0
  );
}

=head2 _get_cache_stats

Returns cache statistics for performance monitoring and optimization.

Returns:
  - Hash reference containing cache hit/miss statistics

=cut

sub _get_cache_stats {
  return \%cache_stats;
}

=head2 _optimize_cache_size

Optimizes cache size by removing least recently used entries when cache grows too large.
Implements cache size management to prevent excessive memory usage.

Arguments:
  - $max_entries: Maximum number of entries to keep in each cache (default: 1000)

=cut

sub _optimize_cache_size {
  my ($max_entries) = @_;
  $max_entries ||= 1000;  # Default maximum cache size
  
  # Optimize stat cache
  if (keys %stat_cache > $max_entries) {
    my @keys = keys %stat_cache;
    my $to_remove = @keys - $max_entries;
    
    # Remove oldest entries (simple FIFO approach)
    for my $i (0 .. $to_remove - 1) {
      delete $stat_cache{$keys[$i]};
    }
  }
  
  # Optimize permission cache
  if (keys %permission_cache > $max_entries) {
    my @keys = keys %permission_cache;
    my $to_remove = @keys - $max_entries;
    
    for my $i (0 .. $to_remove - 1) {
      delete $permission_cache{$keys[$i]};
    }
  }
  
  # Optimize binary cache
  if (keys %binary_cache > $max_entries) {
    my @keys = keys %binary_cache;
    my $to_remove = @keys - $max_entries;
    
    for my $i (0 .. $to_remove - 1) {
      delete $binary_cache{$keys[$i]};
    }
  }
}

=head2 main_search

Main search function that orchestrates the entire search process using optimized path-based traversal.

Arguments:
  - $term: Search pattern
  - $ignores_ref: Hash reference of ignore patterns
  - $targets_ref: Hash reference of target patterns
  - $context: Number of context lines
  - $maxline: Maximum lines to read
  - $debug: Debug flag

=cut

sub main_search {
  my ($term, $ignores_ref, $targets_ref, $context, $maxline, $debug) = @_;
  my %seen;
  
  # Get absolute path of current working directory once and reuse
  require Cwd;
  my $start_dir = Cwd::abs_path(Cwd::getcwd());
  
  if ($debug) {
    print Term::ANSIColor::color("magenta");
    print "starting search from absolute path: $start_dir\n";
    print Term::ANSIColor::color("reset");
  }
  
  handle_dir($start_dir, $term, $ignores_ref, $targets_ref, \%seen, $context, $maxline, $debug);
  
  # Optimize cache size to prevent excessive memory usage
  _optimize_cache_size();
  
  # Print cache statistics in debug mode
  if ($debug) {
    my $stats = _get_cache_stats();
    print Term::ANSIColor::color("magenta");
    print "Cache Statistics:\n";
    print "  Stat cache: $stats->{stat_hits} hits, $stats->{stat_misses} misses\n";
    print "  Permission cache: $stats->{permission_hits} hits, $stats->{permission_misses} misses\n";
    print "  Binary cache: $stats->{binary_hits} hits, $stats->{binary_misses} misses\n";
    print Term::ANSIColor::color("reset");
  }
  
  # Ensure all buffered output is flushed at the end of search
  flush_output_buffer();
}

1;

__END__

=head1 AUTHOR

Julio de Leon

=head1 COPYRIGHT AND LICENSE

This software is copyright (c) 2024 by Julio de Leon.

=cut