#!/usr/bin/perl

use strict;
use warnings;
use Test::More;
use File::Temp qw(tempfile tempdir);
use File::Spec;
use FindBin;
use lib "$FindBin::Bin/../lib";
use Cwd;

# Import test utilities
require "$FindBin::Bin/test_utils.pl";

# Test memory management and performance optimizations
plan tests => 6;

# Helper function to run gf command in a specific directory
sub run_gf_in_dir {
    my ($dir, @args) = @_;
    my $original_dir = Cwd::getcwd();
    chdir $dir or die "Cannot chdir to $dir: $!";
    
    # Quote arguments that might contain spaces or special characters
    my @quoted_args = map { 
        if (/\s/ || /['"\\]/) {
            s/'/'\\''/g;  # Escape single quotes
            "'$_'";
        } else {
            $_;
        }
    } @args;
    
    my $cmd = "perl -I$original_dir/lib $original_dir/bin/gf " . join(' ', @quoted_args) . " 2>&1";
    my $output = `$cmd`;
    
    chdir $original_dir or die "Cannot chdir back to $original_dir: $!";
    return $output;
}

# Test 1: Bounded pending matches queue with many matches
{
  my $temp_dir = tempdir(CLEANUP => 1);
  my ($fh, $filename) = tempfile(DIR => $temp_dir, SUFFIX => '.txt');
  
  # Create a file with many matches to test queue overflow handling
  my $lines_with_matches = 150;  # More than the default max_pending_matches (100)
  my $context_lines_between = 2;
  
  for my $i (1 .. $lines_with_matches) {
    print $fh "match_line_$i PATTERN_TO_FIND content_$i\n";
    # Add some non-matching lines between matches
    for my $j (1 .. $context_lines_between) {
      print $fh "context_line_${i}_${j} some other content\n";
    }
  }
  close $fh;
  
  # Run search with context to trigger pending matches queue
  my $output = run_gf_in_dir($temp_dir, "-s", "PATTERN_TO_FIND", "-c", "3");
  
  # Should complete without memory issues and display all matches
  like($output, qr/match_line_1.*PATTERN_TO_FIND/, "First match displayed correctly");
  like($output, qr/match_line_150.*PATTERN_TO_FIND/, "Last match displayed correctly");
  
  # Count total matches found
  my @matches = $output =~ /match_line_\d+.*PATTERN_TO_FIND/g;
  is(scalar(@matches), $lines_with_matches, "All matches found despite queue overflow");
  
  # Check for queue overflow warning in debug output
  my $debug_output = run_gf_in_dir($temp_dir, "-s", "PATTERN_TO_FIND", "-c", "3", "-d");
  like($debug_output, qr/queue overflow|Force displaying.*oldest matches/i, "Queue overflow handling triggered");
}

# Test 2: Memory usage with bounded queues and large files
{
  my $temp_dir = tempdir(CLEANUP => 1);
  my ($fh, $filename) = tempfile(DIR => $temp_dir, SUFFIX => '.txt');
  
  # Create a large file with periodic matches
  my $total_lines = 100;  # Reduced for faster testing
  my $match_frequency = 10;  # Every 10th line has a match
  
  for my $i (1 .. $total_lines) {
    if ($i % $match_frequency == 0) {
      print $fh "line_$i FINDME important_data_$i\n";
    } else {
      print $fh "line_$i regular content here\n";
    }
  }
  close $fh;
  
  # Test with high context value to stress memory management
  my $output = run_gf_in_dir($temp_dir, "-s", "FINDME", "-c", "2");
  
  # Should handle large file efficiently
  my @matches = $output =~ /line_\d+.*FINDME/g;
  my $expected_matches = int($total_lines / $match_frequency);
  is(scalar(@matches), $expected_matches, "Large file processed efficiently with bounded queue");
}

# Test 3: Maxline limits with after-context collection  
{
  my $temp_dir = tempdir(CLEANUP => 1);
  my ($fh, $filename) = tempfile(DIR => $temp_dir, SUFFIX => '.txt');
  
  # Create file with matches near maxline limit
  for my $i (1 .. 100) {
    if ($i == 45 || $i == 55) {  # Matches at lines 45 and 55
      print $fh "line_$i TARGET_PATTERN data_$i\n";
    } else {
      print $fh "line_$i regular content\n";
    }
  }
  close $fh;
  
  # Test with maxline limit that cuts off after-context
  my $output = run_gf_in_dir($temp_dir, "-s", "TARGET_PATTERN", "-c", "3", "-m", "50");
  
  # Should find first match with full context, second match should be limited
  like($output, qr/line_45.*TARGET_PATTERN/, "First match found within maxline limit");
}



done_testing();