#!/usr/bin/perl

use strict;
use warnings;
use Test::More;
use FindBin;
use lib "$FindBin::Bin/../lib";
use GF::Search qw(check_file);
use GF::PatternCache qw(initialize_pattern_cache);
use File::Temp qw(tempfile);
use Capture::Tiny qw(capture_stdout);

# Load test utilities
require "$FindBin::Bin/context_test_utils.pl";

plan tests => 12;

# Test 1: Basic overlapping context with context=2
{
  my $test_content = <<'EOF';
Line 1: Before first match
Line 2: function_alpha appears here
Line 3: Between the two matches
Line 4: function_beta appears here
Line 5: After second match
EOF

  my ($fh, $filename) = tempfile(UNLINK => 1);
  print $fh $test_content;
  close $fh;

  # Initialize pattern cache
  initialize_pattern_cache('function_', {}, {});

  my $output = capture_stdout {
    check_file($filename, 'function_', 2, 0, 0);
  };

  # Parse the output to extract line numbers
  my $parsed = parse_context_output($output);
  my @line_numbers = map { $_->{line_num} } @{$parsed->{all_lines}};
  
  # Should see lines 1,2,3,4,5 but line 3 should not be duplicated
  my %seen_lines;
  foreach my $line_num (@line_numbers) {
    $seen_lines{$line_num}++;
  }
  
  is($seen_lines{3}, 1, "Line 3 appears only once despite being in overlap region");
  ok(exists $seen_lines{1}, "Line 1 (before-context of first match) is displayed");
  ok(exists $seen_lines{2}, "Line 2 (first match) is displayed");
  ok(exists $seen_lines{4}, "Line 4 (second match) is displayed");
  ok(exists $seen_lines{5}, "Line 5 (after-context of second match) is displayed");
}

# Test 2: Overlapping context with context=1
{
  my $test_content = <<'EOF';
Line 1: function_alpha appears here
Line 2: Between matches
Line 3: function_beta appears here
EOF

  my ($fh, $filename) = tempfile(UNLINK => 1);
  print $fh $test_content;
  close $fh;

  # Initialize pattern cache
  initialize_pattern_cache('function_', {}, {});

  my $output = capture_stdout {
    check_file($filename, 'function_', 1, 0, 0);
  };

  my $parsed = parse_context_output($output);
  my @line_numbers = map { $_->{line_num} } @{$parsed->{all_lines}};
  
  # With context=1, line 2 would be after-context of first match and before-context of second
  my %seen_lines;
  foreach my $line_num (@line_numbers) {
    $seen_lines{$line_num}++;
  }
  
  is($seen_lines{2}, 1, "Line 2 appears only once in overlap scenario with context=1");
}

# Test 3: Multiple overlapping matches
{
  my $test_content = <<'EOF';
Line 1: function_alpha appears here
Line 2: Between first and second
Line 3: function_beta appears here
Line 4: Between second and third
Line 5: function_gamma appears here
Line 6: After all matches
EOF

  my ($fh, $filename) = tempfile(UNLINK => 1);
  print $fh $test_content;
  close $fh;

  # Initialize pattern cache
  initialize_pattern_cache('function_', {}, {});

  my $output = capture_stdout {
    check_file($filename, 'function_', 1, 0, 0);
  };

  my $parsed = parse_context_output($output);
  my @line_numbers = map { $_->{line_num} } @{$parsed->{all_lines}};
  
  # Check that no line numbers are duplicated
  my %seen_lines;
  foreach my $line_num (@line_numbers) {
    $seen_lines{$line_num}++;
  }
  
  my $duplicated_lines = 0;
  foreach my $count (values %seen_lines) {
    $duplicated_lines++ if $count > 1;
  }
  
  is($duplicated_lines, 0, "No lines are duplicated with multiple overlapping matches");
}

# Test 4: Adjacent matches (maximum overlap)
{
  my $test_content = <<'EOF';
Line 1: function_alpha appears here
Line 2: function_beta appears here
Line 3: After both matches
EOF

  my ($fh, $filename) = tempfile(UNLINK => 1);
  print $fh $test_content;
  close $fh;

  # Initialize pattern cache
  initialize_pattern_cache('function_', {}, {});

  my $output = capture_stdout {
    check_file($filename, 'function_', 1, 0, 0);
  };

  my $parsed = parse_context_output($output);
  my @line_numbers = map { $_->{line_num} } @{$parsed->{all_lines}};
  
  # Verify both matches are displayed and no duplication occurs
  my %seen_lines;
  foreach my $line_num (@line_numbers) {
    $seen_lines{$line_num}++;
  }
  
  ok(exists $seen_lines{1}, "First match line is displayed");
  ok(exists $seen_lines{2}, "Second match line is displayed");
  is($seen_lines{1}, 1, "First match line appears only once");
  is($seen_lines{2}, 1, "Second match line appears only once");
}

# Test 5: Overlapping context with different context values
{
  my $test_content = <<'EOF';
Line 1: Before all
Line 2: function_alpha appears here
Line 3: Between matches
Line 4: function_beta appears here
Line 5: After all
EOF

  my ($fh, $filename) = tempfile(UNLINK => 1);
  print $fh $test_content;
  close $fh;

  # Initialize pattern cache
  initialize_pattern_cache('function_', {}, {});

  # Test with context=3 (should include all lines but no duplication)
  my $output = capture_stdout {
    check_file($filename, 'function_', 3, 0, 0);
  };

  my $parsed = parse_context_output($output);
  my @line_numbers = map { $_->{line_num} } @{$parsed->{all_lines}};
  
  # Check that all lines 1-5 appear exactly once
  my %seen_lines;
  foreach my $line_num (@line_numbers) {
    $seen_lines{$line_num}++;
  }
  
  my $all_lines_once = 1;
  for my $line_num (1..5) {
    if (!exists $seen_lines{$line_num} || $seen_lines{$line_num} != 1) {
      $all_lines_once = 0;
      last;
    }
  }
  
  ok($all_lines_once, "All lines appear exactly once with large context value");
}

done_testing();