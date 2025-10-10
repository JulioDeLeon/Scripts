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
  my @pending_matches = (); # Queue for matches waiting for after-context
  my @recently_displayed = (); # Recently displayed matches for overlap detection
  my $search_pattern = get_search_pattern();
  
  # Memory management: bounded pending matches queue
  my $max_pending_matches = 100;  # Maximum number of pending matches to prevent memory issues
  
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
      
      # Create pending match entry instead of immediate display
      my @before_context = ();
      if ($context) {
        # Copy before-context from rolling buffer
        my $start_context_line = $line_num - $context;
        foreach my $entry (@context_buffer) {
          if ($entry->{line_num} >= $start_context_line && $entry->{line_num} < $line_num) {
            push @before_context, {
              line_num => $entry->{line_num},
              content => $entry->{content}
            };
          }
        }
      }
      
      # Create pending match structure
      my $pending_match = {
        line_num => $line_num,
        content => $line,
        before_context => \@before_context,
        after_context => [],
        after_needed => $context
      };
      
      # If no context needed, display immediately
      if ($context == 0) {
        _display_match_with_context(
          \@context_buffer,
          $pending_match->{content},
          $pending_match->{line_num},
          $term,
          $context,
          $pending_match->{after_context},
          undef
        );
      } else {
        # Handle queue overflow to prevent memory issues
        if (@pending_matches >= $max_pending_matches) {
          if ($debug) {
            print Term::ANSIColor::color("yellow");
            print "Warning: Pending matches queue overflow at line $line_num. Forcing display of oldest matches.\n";
            print Term::ANSIColor::color("reset");
          }
          
          # Force display of oldest matches to make room
          _force_display_oldest_matches(\@pending_matches, \@context_buffer, $term, $context, \@recently_displayed, $debug);
        }
        
        push @pending_matches, $pending_match;
      }
    }
    
    # Add current line to after-context of pending matches that need it
    # Respect maxline limits - don't collect after-context beyond maxline
    if (@pending_matches && $context > 0) {
      foreach my $pending_match (@pending_matches) {
        # Only collect after-context for matches that still need it
        if ($pending_match->{after_needed} > 0 && $line_num > $pending_match->{line_num}) {
          # Check if we're within maxline limit for after-context collection
          if (!$maxline || $line_num <= $maxline) {
            push @{$pending_match->{after_context}}, {
              line_num => $line_num,
              content => $line
            };
            $pending_match->{after_needed}--;
          } else {
            # Beyond maxline limit - mark after-context as complete with available lines
            if ($debug) {
              print Term::ANSIColor::color("yellow");
              print "Maxline limit ($maxline) reached - completing after-context for match at line $pending_match->{line_num} with available lines\n";
              print Term::ANSIColor::color("reset");
            }
            $pending_match->{after_needed} = 0;  # Mark as complete
          }
        }
      }
    }
    
    # Check for completed matches and display them with overlapping context handling
    my @remaining_matches = ();
    foreach my $pending_match (@pending_matches) {
      if ($pending_match->{after_needed} <= 0) {
        # Match has collected sufficient after-context
        # Check if we should delay display to handle potential overlaps
        if (_should_delay_match_display($pending_match, \@pending_matches, $context, $line_num, $debug)) {
          # Keep this match in pending queue for now
          push @remaining_matches, $pending_match;
        } else {
          # Safe to display this match now
          my $display_match = _prepare_match_for_display($pending_match, \@pending_matches, \@recently_displayed, $context, $debug);
          
          # Display the match with optimized context
          _display_match_with_context(
            \@context_buffer,
            $display_match->{content},
            $display_match->{line_num},
            $term,
            $context,
            $display_match->{after_context},
            $display_match->{optimized_before_context}
          );
          
          # Add to recently displayed matches for future overlap detection
          push @recently_displayed, $display_match;
          
          # Keep only recent matches to avoid memory growth
          if (@recently_displayed > 10) {
            shift @recently_displayed;
          }
        }
      } else {
        # Match still needs more after-context
        push @remaining_matches, $pending_match;
      }
    }
    @pending_matches = @remaining_matches;
    
    # Maintain rolling context buffer
    _manage_context_buffer(\@context_buffer, $line, $line_num, $context);
    
    # Early exit for maxline limit to avoid unnecessary processing
    # When maxline is reached, complete any pending matches with available after-context
    if ($maxline && $line_num >= $maxline) {
      if (@pending_matches && $debug) {
        print Term::ANSIColor::color("yellow");
        print "Maxline limit ($maxline) reached with " . scalar(@pending_matches) . " pending matches - completing with available after-context\n";
        print Term::ANSIColor::color("reset");
      }
      
      # Mark all pending matches as complete (they'll be processed in EOF handling)
      foreach my $pending_match (@pending_matches) {
        $pending_match->{after_needed} = 0;
      }
      
      last;
    }
  }
  
  # Handle end-of-file scenarios for incomplete after-context
  if (@pending_matches) {
    if ($debug) {
      print Term::ANSIColor::color("magenta");
      print "EOF detected with " . scalar(@pending_matches) . " pending matches\n";
      print Term::ANSIColor::color("reset");
    }
    
    # Sort pending matches by line number to process them in order
    my @sorted_pending = sort { $a->{line_num} <=> $b->{line_num} } @pending_matches;
    
    foreach my $pending_match (@sorted_pending) {
      if ($debug) {
        my $incomplete_context = $pending_match->{after_needed} > 0 ? "incomplete" : "complete";
        my $available_after = scalar(@{$pending_match->{after_context}});
        my $maxline_note = $pending_match->{maxline_limited} ? " (limited by maxline)" : "";
        print Term::ANSIColor::color("magenta");
        print "Processing pending match at line $pending_match->{line_num}: $incomplete_context after-context ($available_after lines available)$maxline_note\n";
        print Term::ANSIColor::color("reset");
      }
      
      # Prepare match for display with overlapping context handling at EOF
      my $display_match = _prepare_match_for_display($pending_match, \@sorted_pending, \@recently_displayed, $context, $debug);
      
      # Display matches with whatever after-context is available at EOF
      _display_match_with_context(
        \@context_buffer,
        $display_match->{content},
        $display_match->{line_num},
        $term,
        $context,
        $display_match->{after_context},
        $display_match->{optimized_before_context}
      );
      
      # Add to recently displayed matches for future overlap detection
      push @recently_displayed, $display_match;
      
      # Keep only recent matches to avoid memory growth
      if (@recently_displayed > 10) {
        shift @recently_displayed;
      }
    }
    
    # Clean up pending matches queue after processing all remaining matches
    @pending_matches = ();
    
    if ($debug) {
      print Term::ANSIColor::color("magenta");
      print "Pending matches queue cleaned up at EOF\n";
      print Term::ANSIColor::color("reset");
    }
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

=head2 _should_delay_match_display

Determines if a completed match should be delayed to handle potential overlapping context
with matches that might appear soon.

Arguments:
  - $completed_match: Match that has collected sufficient after-context
  - $pending_matches_ref: Reference to array of all pending matches
  - $context: Number of context lines
  - $current_line_num: Current line number being processed
  - $debug: Debug flag

Returns:
  - 1 if match should be delayed, 0 if safe to display now

=cut

sub _should_delay_match_display {
  my ($completed_match, $pending_matches_ref, $context, $current_line_num, $debug) = @_;
  
  # If no context, no need to delay
  return 0 if $context == 0;
  
  my $match_after_end = $completed_match->{line_num} + $context;
  
  # Check if there are any pending matches that might create overlaps
  foreach my $pending_match (@$pending_matches_ref) {
    # Skip the completed match itself
    next if $pending_match->{line_num} == $completed_match->{line_num};
    
    # Only consider matches that come after the completed match
    next if $pending_match->{line_num} <= $completed_match->{line_num};
    
    my $next_before_start = $pending_match->{line_num} - $context;
    
    # If there's potential overlap, delay display
    if ($match_after_end >= $next_before_start) {
      if ($debug) {
        print Term::ANSIColor::color("magenta");
        print "Delaying display of match at line $completed_match->{line_num} due to potential overlap with match at line $pending_match->{line_num}\n";
        print Term::ANSIColor::color("reset");
      }
      return 1;
    }
  }
  
  # Only delay if we're very close to the end of the file and there might be more matches
  # This is a more conservative approach to avoid unnecessary delays
  my $conservative_window = $match_after_end + 1; # Only delay if we're within 1 line of the after-context end
  if ($current_line_num <= $conservative_window) {
    if ($debug) {
      print Term::ANSIColor::color("magenta");
      print "Delaying display of match at line $completed_match->{line_num} - still very close to after-context end (current line: $current_line_num, after-context ends: $match_after_end)\n";
      print Term::ANSIColor::color("reset");
    }
    return 1;
  }
  
  if ($debug) {
    print Term::ANSIColor::color("magenta");
    print "Safe to display match at line $completed_match->{line_num} - no overlapping matches detected\n";
    print Term::ANSIColor::color("reset");
  }
  
  return 0;
}

=head2 _prepare_match_for_display

Prepares a match for display by detecting and handling overlapping context with subsequent matches.
Implements deduplication logic to avoid displaying same lines multiple times.

Arguments:
  - $current_match: Current match to prepare for display
  - $pending_matches_ref: Reference to array of all pending matches
  - $context: Number of context lines
  - $debug: Debug flag

Returns:
  - Hash reference with optimized match data for display

=cut

sub _prepare_match_for_display {
  my ($current_match, $pending_matches_ref, $recently_displayed_ref, $context, $debug) = @_;
  
  # Create a copy of the match to avoid modifying the original
  my $display_match = {
    line_num => $current_match->{line_num},
    content => $current_match->{content},
    before_context => [@{$current_match->{before_context}}],
    after_context => [@{$current_match->{after_context}}],
    optimized_before_context => $current_match->{optimized_before_context}
  };
  
  # Check if this match was already optimized by a previous match
  if ($current_match->{optimized_before_context}) {
    $display_match->{optimized_before_context} = $current_match->{optimized_before_context};
  }
  
  # If no context, return as-is
  if ($context == 0) {
    return $display_match;
  }
  
  # Find the next match that might have overlapping context
  my $next_match = _find_next_overlapping_match($current_match, $pending_matches_ref, $context, $debug);
  
  # Find the previous match that might have overlapping context with current match
  my $prev_match = _find_previous_overlapping_match($current_match, $recently_displayed_ref, $context, $debug);
  
  if ($next_match) {
    if ($debug) {
      print Term::ANSIColor::color("magenta");
      print "Detected overlapping context between match at line $current_match->{line_num} and line $next_match->{line_num}\n";
      print Term::ANSIColor::color("reset");
    }
    
    # Merge overlapping context sections efficiently
    my $merge_info = _merge_overlapping_context_sections($current_match, $next_match, $context, $debug);
    
    # Optimize after-context based on merge information
    $display_match->{after_context} = _optimize_after_context_for_overlap(
      $current_match->{after_context},
      $next_match,
      $context,
      $debug
    );
    
    # Store merge information for potential future optimizations
    $display_match->{merge_info} = $merge_info;
  }
  
  if ($prev_match) {
    if ($debug) {
      print Term::ANSIColor::color("magenta");
      print "Detected overlapping context between previous match at line $prev_match->{line_num} and current match at line $current_match->{line_num}\n";
      print Term::ANSIColor::color("reset");
    }
    
    # Optimize before-context to avoid duplication with previous match
    $display_match->{optimized_before_context} = _optimize_before_context_for_overlap(
      $current_match->{before_context},
      $prev_match,
      $context,
      $debug
    );
    
    # Also optimize after-context to avoid duplication with previous match's after-context
    $display_match->{after_context} = _optimize_after_context_with_previous(
      $display_match->{after_context},
      $prev_match,
      $context,
      $debug
    );
  }
  
  # If no optimization was applied, use the original before-context
  if (!$display_match->{optimized_before_context}) {
    $display_match->{optimized_before_context} = $display_match->{before_context};
  }
  
  return $display_match;
}

=head2 _find_previous_overlapping_match

Finds the previous match that has overlapping context with the current match.
This is used to optimize the current match's before-context.

Arguments:
  - $current_match: Current match being processed
  - $pending_matches_ref: Reference to array of all pending matches
  - $context: Number of context lines
  - $debug: Debug flag

Returns:
  - Reference to previous overlapping match, or undef if none found

=cut

sub _find_previous_overlapping_match {
  my ($current_match, $recently_displayed_ref, $context, $debug) = @_;
  
  my $current_before_start = $current_match->{line_num} - $context;
  
  # Find the closest previous match that might overlap
  my $closest_match = undef;
  my $closest_distance = 999999;
  
  foreach my $displayed_match (@$recently_displayed_ref) {
    # Only consider matches that come before the current match
    next if $displayed_match->{line_num} >= $current_match->{line_num};
    
    my $prev_after_end = $displayed_match->{line_num} + $context;
    
    # Check if there's overlap: previous after-context overlaps with current before-context
    if ($prev_after_end >= $current_before_start) {
      my $distance = $current_match->{line_num} - $displayed_match->{line_num};
      if ($distance < $closest_distance) {
        $closest_distance = $distance;
        $closest_match = $displayed_match;
      }
    }
  }
  
  if ($debug && $closest_match) {
    print Term::ANSIColor::color("magenta");
    print "Found previous overlapping match: prev ends context at line " . ($closest_match->{line_num} + $context) . ", current starts context at line $current_before_start\n";
    print Term::ANSIColor::color("reset");
  }
  
  return $closest_match;
}

=head2 _find_next_overlapping_match

Finds the next pending match that has overlapping context with the current match.

Arguments:
  - $current_match: Current match being processed
  - $pending_matches_ref: Reference to array of all pending matches
  - $context: Number of context lines
  - $debug: Debug flag

Returns:
  - Reference to next overlapping match, or undef if none found

=cut

sub _find_next_overlapping_match {
  my ($current_match, $pending_matches_ref, $context, $debug) = @_;
  
  my $current_after_end = $current_match->{line_num} + $context;
  
  # Find the closest subsequent match that might overlap
  my $closest_match = undef;
  my $closest_distance = 999999;
  
  foreach my $pending_match (@$pending_matches_ref) {
    # Skip the current match itself
    next if $pending_match->{line_num} == $current_match->{line_num};
    
    # Only consider matches that come after the current match
    next if $pending_match->{line_num} <= $current_match->{line_num};
    
    my $next_before_start = $pending_match->{line_num} - $context;
    
    # Check if there's overlap: current after-context overlaps with next before-context
    if ($current_after_end >= $next_before_start) {
      my $distance = $pending_match->{line_num} - $current_match->{line_num};
      if ($distance < $closest_distance) {
        $closest_distance = $distance;
        $closest_match = $pending_match;
      }
    }
  }
  
  if ($debug && $closest_match) {
    print Term::ANSIColor::color("magenta");
    print "Found overlapping match: current ends at line $current_after_end, next starts context at line " . ($closest_match->{line_num} - $context) . "\n";
    print Term::ANSIColor::color("reset");
  }
  
  return $closest_match;
}

=head2 _merge_overlapping_context_sections

Efficiently merges overlapping context sections between consecutive matches.
Maintains proper line numbering and ensures match highlighting is preserved.

Arguments:
  - $current_match: Current match being processed
  - $next_match: Next match with overlapping context
  - $context: Number of context lines
  - $debug: Debug flag

Returns:
  - Hash reference with merged context information

=cut

sub _merge_overlapping_context_sections {
  my ($current_match, $next_match, $context, $debug) = @_;
  
  my $current_after_end = $current_match->{line_num} + $context;
  my $next_before_start = $next_match->{line_num} - $context;
  
  # Calculate the overlap region
  my $overlap_start = $next_before_start;
  my $overlap_end = $current_after_end;
  
  if ($debug) {
    print Term::ANSIColor::color("magenta");
    print "Merging context: current match ends context at line $current_after_end, next starts at line $next_before_start\n";
    print "Overlap region: lines $overlap_start to $overlap_end\n";
    print Term::ANSIColor::color("reset");
  }
  
  # Return information about the merged sections
  return {
    has_overlap => ($overlap_start <= $overlap_end),
    overlap_start => $overlap_start,
    overlap_end => $overlap_end,
    current_context_cutoff => $overlap_start - 1
  };
}

=head2 _optimize_before_context_for_overlap

Optimizes before-context to avoid duplication with the previous match's after-context.
Maintains proper line numbering and formatting while eliminating redundant lines.

Arguments:
  - $before_context_ref: Reference to current match's before-context array
  - $prev_match: Previous match that has overlapping context
  - $context: Number of context lines
  - $debug: Debug flag

Returns:
  - Reference to optimized before-context array

=cut

sub _optimize_before_context_for_overlap {
  my ($before_context_ref, $prev_match, $context, $debug) = @_;
  
  my @optimized_before_context = ();
  my $prev_after_end = $prev_match->{line_num} + $context;
  my $prev_match_line = $prev_match->{line_num};
  
  # Only include before-context lines that don't overlap with previous match's after-context
  foreach my $context_line (@$before_context_ref) {
    # Skip lines that would have been displayed in previous match's after-context
    if ($context_line->{line_num} <= $prev_after_end) {
      if ($debug) {
        print Term::ANSIColor::color("magenta");
        print "Removing overlapping before-context line $context_line->{line_num} to avoid duplication with previous match's after-context\n";
        print Term::ANSIColor::color("reset");
      }
    } else {
      push @optimized_before_context, $context_line;
    }
  }
  
  if ($debug) {
    my $original_count = scalar(@$before_context_ref);
    my $optimized_count = scalar(@optimized_before_context);
    print Term::ANSIColor::color("magenta");
    print "Optimized before-context for overlap: $original_count -> $optimized_count lines (prev match ended context at line $prev_after_end)\n";
    print Term::ANSIColor::color("reset");
  }
  
  return \@optimized_before_context;
}

=head2 _optimize_after_context_with_previous

Optimizes after-context to avoid duplication with the previous match's after-context.
This handles cases where multiple matches have overlapping after-context ranges.

Arguments:
  - $after_context_ref: Reference to current match's after-context array
  - $prev_match: Previous match that might have overlapping after-context
  - $context: Number of context lines
  - $debug: Debug flag

Returns:
  - Reference to optimized after-context array

=cut

sub _optimize_after_context_with_previous {
  my ($after_context_ref, $prev_match, $context, $debug) = @_;
  
  my @optimized_after_context = ();
  my $prev_after_end = $prev_match->{line_num} + $context;
  
  # Only include after-context lines that don't overlap with previous match's after-context
  foreach my $context_line (@$after_context_ref) {
    # Skip lines that would have been displayed in previous match's after-context
    if ($context_line->{line_num} <= $prev_after_end) {
      if ($debug) {
        print Term::ANSIColor::color("magenta");
        print "Removing overlapping after-context line $context_line->{line_num} to avoid duplication with previous match's after-context\n";
        print Term::ANSIColor::color("reset");
      }
    } else {
      push @optimized_after_context, $context_line;
    }
  }
  
  if ($debug) {
    my $original_count = scalar(@$after_context_ref);
    my $optimized_count = scalar(@optimized_after_context);
    print Term::ANSIColor::color("magenta");
    print "Optimized after-context with previous: $original_count -> $optimized_count lines (prev match ended context at line $prev_after_end)\n";
    print Term::ANSIColor::color("reset");
  }
  
  return \@optimized_after_context;
}

=head2 _handle_maxline_limited_context

Handles cases where maxline limit prevents complete after-context collection.
Ensures pending matches are properly completed when maxline is reached.

Arguments:
  - $pending_matches_ref: Reference to pending matches array
  - $maxline: Maximum line limit
  - $current_line_num: Current line number
  - $debug: Debug flag

=cut

sub _handle_maxline_limited_context {
  my ($pending_matches_ref, $maxline, $current_line_num, $debug) = @_;
  
  return unless $maxline && $current_line_num >= $maxline;
  
  if ($debug && @$pending_matches_ref) {
    print Term::ANSIColor::color("yellow");
    print "Handling maxline-limited context for " . scalar(@$pending_matches_ref) . " pending matches\n";
    print Term::ANSIColor::color("reset");
  }
  
  # Complete all pending matches that are affected by maxline limit
  foreach my $pending_match (@$pending_matches_ref) {
    if ($pending_match->{after_needed} > 0) {
      my $available_after = scalar(@{$pending_match->{after_context}});
      my $needed_after = $pending_match->{after_needed};
      
      if ($debug) {
        print Term::ANSIColor::color("yellow");
        print "Match at line $pending_match->{line_num}: needed $needed_after after-context, got $available_after (limited by maxline $maxline)\n";
        print Term::ANSIColor::color("reset");
      }
      
      # Mark as complete with available after-context
      $pending_match->{after_needed} = 0;
      $pending_match->{maxline_limited} = 1;  # Flag for debugging/reporting
    }
  }
}

=head2 _force_display_oldest_matches

Forces display of oldest pending matches when queue reaches maximum capacity.
This prevents memory issues with files containing many matches.

Arguments:
  - $pending_matches_ref: Reference to pending matches array
  - $context_buffer_ref: Reference to context buffer
  - $term: Search pattern for highlighting
  - $context: Number of context lines
  - $recently_displayed_ref: Reference to recently displayed matches
  - $debug: Debug flag

=cut

sub _force_display_oldest_matches {
  my ($pending_matches_ref, $context_buffer_ref, $term, $context, $recently_displayed_ref, $debug) = @_;
  
  # Calculate how many matches to force display (display half to make room)
  my $matches_to_display = int(@$pending_matches_ref / 2);
  $matches_to_display = 1 if $matches_to_display < 1;  # Display at least one match
  
  if ($debug) {
    print Term::ANSIColor::color("magenta");
    print "Force displaying $matches_to_display oldest matches from queue of " . scalar(@$pending_matches_ref) . " matches\n";
    print Term::ANSIColor::color("reset");
  }
  
  # Sort pending matches by line number to process oldest first
  my @sorted_matches = sort { $a->{line_num} <=> $b->{line_num} } @$pending_matches_ref;
  
  # Display the oldest matches with whatever after-context they have
  for my $i (0 .. $matches_to_display - 1) {
    my $match = $sorted_matches[$i];
    
    if ($debug) {
      my $available_after = scalar(@{$match->{after_context}});
      print Term::ANSIColor::color("magenta");
      print "Force displaying match at line $match->{line_num} with $available_after after-context lines\n";
      print Term::ANSIColor::color("reset");
    }
    
    # Prepare match for display with overlapping context handling
    my $display_match = _prepare_match_for_display($match, \@sorted_matches, $recently_displayed_ref, $context, $debug);
    
    # Display the match with available after-context
    _display_match_with_context(
      $context_buffer_ref,
      $display_match->{content},
      $display_match->{line_num},
      $term,
      $context,
      $display_match->{after_context},
      $display_match->{optimized_before_context}
    );
    
    # Add to recently displayed matches for future overlap detection
    push @$recently_displayed_ref, $display_match;
    
    # Keep only recent matches to avoid memory growth
    if (@$recently_displayed_ref > 10) {
      shift @$recently_displayed_ref;
    }
  }
  
  # Remove displayed matches from pending queue
  @$pending_matches_ref = grep {
    my $match = $_;
    my $should_keep = 1;
    for my $i (0 .. $matches_to_display - 1) {
      if ($match->{line_num} == $sorted_matches[$i]->{line_num}) {
        $should_keep = 0;
        last;
      }
    }
    $should_keep;
  } @$pending_matches_ref;
  
  if ($debug) {
    print Term::ANSIColor::color("magenta");
    print "Pending matches queue reduced to " . scalar(@$pending_matches_ref) . " matches after force display\n";
    print Term::ANSIColor::color("reset");
  }
}

=head2 _optimize_after_context_for_overlap

Optimizes after-context to avoid duplication with the next match's before-context.
Maintains proper line numbering and formatting while eliminating redundant lines.

Arguments:
  - $after_context_ref: Reference to current match's after-context array
  - $next_match: Next match that has overlapping context
  - $context: Number of context lines
  - $debug: Debug flag

Returns:
  - Reference to optimized after-context array

=cut

sub _optimize_after_context_for_overlap {
  my ($after_context_ref, $next_match, $context, $debug) = @_;
  
  my @optimized_after_context = ();
  my $next_before_start = $next_match->{line_num} - $context;
  my $next_match_line = $next_match->{line_num};
  
  # Include after-context lines, but handle overlapping lines carefully
  foreach my $context_line (@$after_context_ref) {
    # Special case: if the context line is the next match itself, don't include it
    # as it will be displayed as a match line with proper highlighting
    if ($context_line->{line_num} == $next_match_line) {
      if ($debug) {
        print Term::ANSIColor::color("magenta");
        print "Skipping next match line $context_line->{line_num} from after-context (will be displayed as match)\n";
        print Term::ANSIColor::color("reset");
      }
    }
    # Include all other after-context lines - overlapping lines will be displayed here
    # and removed from the next match's before-context
    else {
      push @optimized_after_context, $context_line;
      if ($context_line->{line_num} >= $next_before_start && $debug) {
        print Term::ANSIColor::color("magenta");
        print "Including overlapping line $context_line->{line_num} in after-context (will be removed from next match's before-context)\n";
        print Term::ANSIColor::color("reset");
      }
    }
  }
  
  if ($debug) {
    my $original_count = scalar(@$after_context_ref);
    my $optimized_count = scalar(@optimized_after_context);
    print Term::ANSIColor::color("magenta");
    print "Optimized after-context for overlap: $original_count -> $optimized_count lines (next match starts context at line $next_before_start)\n";
    print Term::ANSIColor::color("reset");
  }
  
  return \@optimized_after_context;
}

=head2 _display_context_line_with_highlighting

Displays a context line with proper formatting and highlighting if it contains matches.
Ensures match highlighting is preserved in overlapping sections.

Arguments:
  - $line_num: Line number to display
  - $content: Line content
  - $term: Search pattern for highlighting

=cut

sub _display_context_line_with_highlighting {
  my ($line_num, $content, $term) = @_;
  
  buffer_output("[$line_num]\t");
  
  # Check if this context line also contains the search pattern
  my $search_pattern = get_search_pattern();
  if ($content =~ /$search_pattern/) {
    # This context line contains a match, highlight it
    print_str($content, $term);
  } else {
    # Regular context line, no highlighting needed
    buffer_output($content);
  }
  
  buffer_output("\n");
}

=head2 _display_match_with_context

Displays a match with appropriate context lines from the rolling buffer and after-context.
Enhanced to preserve match highlighting in overlapping sections.

Arguments:
  - $buffer_ref: Reference to context buffer array
  - $match_line: The line containing the match
  - $match_line_num: Line number of the match
  - $term: Search pattern for highlighting
  - $context: Number of context lines to display
  - $after_context_ref: Reference to after-context array (optional)

=cut

sub _display_match_with_context {
  my ($buffer_ref, $match_line, $match_line_num, $term, $context, $after_context_ref, $before_context_ref) = @_;
  
  if ($context) {
    # Use optimized before-context if provided, otherwise use buffer
    if ($before_context_ref && @$before_context_ref) {
      # Display optimized before-context
      foreach my $entry (@$before_context_ref) {
        _display_context_line_with_highlighting($entry->{line_num}, $entry->{content}, $term);
      }
    } elsif (defined $before_context_ref && @$before_context_ref == 0) {
      # Optimized before-context is empty, don't display any before-context
      # This handles the case where overlap optimization removed all before-context
    } else {
      # Display context before match from buffer without re-reading file sections
      my $start_context_line = $match_line_num - $context;
      
      foreach my $entry (@$buffer_ref) {
        if ($entry->{line_num} >= $start_context_line && $entry->{line_num} < $match_line_num) {
          _display_context_line_with_highlighting($entry->{line_num}, $entry->{content}, $term);
        }
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
  
  # Display after-context if provided with preserved highlighting for overlapping matches
  if ($context && $after_context_ref && @$after_context_ref) {
    foreach my $entry (@$after_context_ref) {
      _display_context_line_with_highlighting($entry->{line_num}, $entry->{content}, $term);
    }
  }
  
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