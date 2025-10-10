#!/usr/bin/env perl
# Context testing utilities for gf context flag testing

use strict;
use warnings;
use File::Temp qw(tempdir tempfile);
use File::Path qw(make_path);

=head1 NAME

context_test_utils.pl - Specialized utilities for context flag testing

=head1 DESCRIPTION

This module provides specialized testing utilities for validating the context flag
(-c) functionality of the gf tool. It includes functions for creating test files
with known content patterns, validating context output format, and checking
line numbers and content accuracy.

=cut

=head2 create_context_test_file

Creates a test file with numbered lines and specific content patterns for context testing.

Arguments:
  - $filename: Path where to create the test file
  - $total_lines: Total number of lines to create
  - $match_lines: Array reference of line numbers that should contain matches
  - $match_term: The term that should appear in match lines

Returns:
  - Hash reference with file metadata

=cut

sub create_context_test_file {
    my ($filename, $total_lines, $match_lines, $match_term) = @_;
    
    my %match_line_map = map { $_ => 1 } @$match_lines;
    my @file_content = ();
    
    for my $line_num (1 .. $total_lines) {
        my $content;
        if ($match_line_map{$line_num}) {
            $content = "Line $line_num: This line contains $match_term for testing";
        } else {
            $content = "Line $line_num: Regular content without the search term";
        }
        push @file_content, $content;
    }
    
    # Write file
    open my $fh, '>', $filename or die "Cannot write to $filename: $!";
    print $fh join("\n", @file_content) . "\n";
    close $fh;
    
    return {
        filename => $filename,
        total_lines => $total_lines,
        match_lines => $match_lines,
        match_term => $match_term,
        content => \@file_content
    };
}

=head2 create_overlapping_context_test_file

Creates a test file specifically for testing overlapping context scenarios.

Arguments:
  - $filename: Path where to create the test file
  - $match_positions: Array reference of line numbers for matches
  - $match_term: The term that should appear in match lines

Returns:
  - Hash reference with file metadata

=cut

sub create_overlapping_context_test_file {
    my ($filename, $match_positions, $match_term) = @_;
    
    # Create enough lines to test overlapping scenarios
    my $total_lines = (sort { $b <=> $a } @$match_positions)[0] + 5;
    my %match_map = map { $_ => 1 } @$match_positions;
    my @file_content = ();
    
    for my $line_num (1 .. $total_lines) {
        my $content;
        if ($match_map{$line_num}) {
            $content = "Match line $line_num: Contains $match_term here";
        } else {
            $content = "Context line $line_num: Regular content for context testing";
        }
        push @file_content, $content;
    }
    
    # Write file
    open my $fh, '>', $filename or die "Cannot write to $filename: $!";
    print $fh join("\n", @file_content) . "\n";
    close $fh;
    
    return {
        filename => $filename,
        total_lines => $total_lines,
        match_lines => $match_positions,
        match_term => $match_term,
        content => \@file_content
    };
}

=head2 create_edge_case_test_files

Creates test files for edge case scenarios (empty files, single lines, etc.).

Arguments:
  - $base_dir: Directory where to create test files

Returns:
  - Hash reference with paths to created files

=cut

sub create_edge_case_test_files {
    my ($base_dir) = @_;
    
    my %files = ();
    
    # Empty file
    $files{empty} = "$base_dir/empty.txt";
    open my $fh1, '>', $files{empty} or die "Cannot create empty file: $!";
    close $fh1;
    
    # Single line with match
    $files{single_match} = "$base_dir/single_match.txt";
    open my $fh2, '>', $files{single_match} or die "Cannot create single match file: $!";
    print $fh2 "This single line contains function keyword\n";
    close $fh2;
    
    # Single line without match
    $files{single_no_match} = "$base_dir/single_no_match.txt";
    open my $fh3, '>', $files{single_no_match} or die "Cannot create single no match file: $!";
    print $fh3 "This single line has no search term\n";
    close $fh3;
    
    # File with match at first line
    $files{match_at_start} = "$base_dir/match_at_start.txt";
    open my $fh4, '>', $files{match_at_start} or die "Cannot create match at start file: $!";
    print $fh4 "Line 1: function appears here\n";
    print $fh4 "Line 2: Context after match\n";
    print $fh4 "Line 3: More context after\n";
    print $fh4 "Line 4: Even more context\n";
    close $fh4;
    
    # File with match at last line
    $files{match_at_end} = "$base_dir/match_at_end.txt";
    open my $fh5, '>', $files{match_at_end} or die "Cannot create match at end file: $!";
    print $fh5 "Line 1: Context before match\n";
    print $fh5 "Line 2: More context before\n";
    print $fh5 "Line 3: Even more context\n";
    print $fh5 "Line 4: function appears at end\n";
    close $fh5;
    
    return \%files;
}

=head2 parse_context_output

Parses gf output and extracts structured information about matches and context.

Arguments:
  - $output: Raw output from gf command

Returns:
  - Hash reference with parsed information

=cut

sub parse_context_output {
    my ($output) = @_;
    
    my @lines = split /\n/, $output;
    my @parsed_lines = ();
    my @match_groups = ();
    my $current_group = [];
    
    for my $line (@lines) {
        # Skip empty lines (used for spacing between match groups)
        if ($line =~ /^\s*$/) {
            if (@$current_group > 0) {
                push @match_groups, $current_group;
                $current_group = [];
            }
            next;
        }
        
        # Parse line with format [line_num]\tcontent
        if ($line =~ /^\[(\d+)\]\t(.*)$/) {
            my ($line_num, $content) = ($1, $2);
            
            # Check if this line contains highlighting (indicates a match)
            my $is_match = ($content =~ /\e\[.*?m.*?\e\[.*?m/) ? 1 : 0;
            
            # Strip ANSI codes for clean content
            my $clean_content = $content;
            $clean_content =~ s/\e\[[0-9;]*m//g;
            
            my $parsed_line = {
                line_num => $line_num,
                content => $content,
                clean_content => $clean_content,
                is_match => $is_match
            };
            
            push @parsed_lines, $parsed_line;
            push @$current_group, $parsed_line;
        }
    }
    
    # Add final group if it exists
    if (@$current_group > 0) {
        push @match_groups, $current_group;
    }
    
    return {
        all_lines => \@parsed_lines,
        match_groups => \@match_groups,
        total_lines => scalar(@parsed_lines),
        total_groups => scalar(@match_groups)
    };
}

=head2 validate_context_symmetry

Validates that context is displayed symmetrically (same number of lines before and after matches).

Arguments:
  - $parsed_output: Output from parse_context_output
  - $expected_context: Expected number of context lines

Returns:
  - Hash reference with validation results

=cut

sub validate_context_symmetry {
    my ($parsed_output, $expected_context) = @_;
    
    my $results = {
        symmetric => 1,
        correct_before_count => 1,
        correct_after_count => 1,
        errors => []
    };
    
    for my $group (@{$parsed_output->{match_groups}}) {
        # Find the match line in the group
        my $match_index = -1;
        for my $i (0 .. $#{$group}) {
            if ($group->[$i]->{is_match}) {
                $match_index = $i;
                last;
            }
        }
        
        if ($match_index >= 0) {
            my $before_count = $match_index;
            my $after_count = @$group - $match_index - 1;
            my $match_line_num = $group->[$match_index]->{line_num};
            
            # Check before context count
            if ($before_count != $expected_context) {
                $results->{correct_before_count} = 0;
                push @{$results->{errors}}, 
                    "Match at line $match_line_num: Expected $expected_context before context, got $before_count";
            }
            
            # Check after context count
            if ($after_count != $expected_context) {
                $results->{correct_after_count} = 0;
                push @{$results->{errors}}, 
                    "Match at line $match_line_num: Expected $expected_context after context, got $after_count";
            }
            
            # Check symmetry
            if ($before_count != $after_count) {
                $results->{symmetric} = 0;
                push @{$results->{errors}}, 
                    "Match at line $match_line_num: Asymmetric context - before: $before_count, after: $after_count";
            }
        }
    }
    
    return $results;
}

=head2 validate_line_number_sequence

Validates that line numbers in context output form proper sequences.

Arguments:
  - $parsed_output: Output from parse_context_output

Returns:
  - Hash reference with validation results

=cut

sub validate_line_number_sequence {
    my ($parsed_output) = @_;
    
    my $results = {
        sequences_valid => 1,
        no_gaps => 1,
        errors => []
    };
    
    for my $group (@{$parsed_output->{match_groups}}) {
        if (@$group > 1) {
            # Check that line numbers form a continuous sequence
            for my $i (1 .. $#{$group}) {
                my $prev_line_num = $group->[$i-1]->{line_num};
                my $curr_line_num = $group->[$i]->{line_num};
                
                if ($curr_line_num != $prev_line_num + 1) {
                    $results->{sequences_valid} = 0;
                    $results->{no_gaps} = 0;
                    push @{$results->{errors}}, 
                        "Gap in line sequence: $prev_line_num -> $curr_line_num";
                }
            }
        }
    }
    
    return $results;
}

=head2 create_performance_test_file

Creates a large test file for performance testing of context functionality.

Arguments:
  - $filename: Path where to create the test file
  - $total_lines: Total number of lines to create
  - $match_frequency: How often matches should appear (every N lines)
  - $match_term: The term that should appear in match lines

Returns:
  - Hash reference with file metadata

=cut

sub create_performance_test_file {
    my ($filename, $total_lines, $match_frequency, $match_term) = @_;
    
    my @match_lines = ();
    
    # Write file in chunks to handle large files efficiently
    open my $fh, '>', $filename or die "Cannot write to $filename: $!";
    
    for my $line_num (1 .. $total_lines) {
        my $content;
        if ($line_num % $match_frequency == 0) {
            $content = "Line $line_num: Performance test line with $match_term keyword";
            push @match_lines, $line_num;
        } else {
            $content = "Line $line_num: Regular performance test content without search term";
        }
        print $fh "$content\n";
    }
    
    close $fh;
    
    return {
        filename => $filename,
        total_lines => $total_lines,
        match_lines => \@match_lines,
        match_term => $match_term,
        match_count => scalar(@match_lines)
    };
}

1;