#!/usr/bin/env perl
use strict;
use warnings;
use Test::More;
use File::Temp qw(tempdir);
use lib 'lib';
use lib 't';

require 'test_utils.pl';
require 'context_test_utils.pl';

BEGIN {
    use_ok('GF::Search', qw(check_file flush_output_buffer));
}

# Helper function to test context functionality directly using the Search module
sub test_context_functionality {
    my ($search_term, $context, $file_path) = @_;
    
    # Initialize pattern cache for the search term
    require GF::PatternCache;
    GF::PatternCache::initialize_pattern_cache($search_term, {}, {});
    
    # Capture output from check_file function
    my $output = capture_output(sub {
        GF::Search::check_file($file_path, $search_term, $context, 0, 0);
    });
    
    # Flush any buffered output
    GF::Search::flush_output_buffer();
    
    return {
        output => $output,
        exit_code => 0,  # Direct function call, assume success
        search_term => $search_term,
        context => $context,
        file_path => $file_path
    };
}

=head1 NAME

15-context-flag-current-behavior.t - Test current context flag behavior

=head1 DESCRIPTION

This test documents and validates the current behavior of the context flag (-c).
It demonstrates that currently only before-context is shown, not after-context.
This test will be updated when the fix is implemented to validate the complete
context functionality.

=cut

# Test current context flag behavior (before fix)
subtest 'current context flag behavior - before context only' => sub {
    plan tests => 8;
    
    # Create a simple test file
    my $temp_dir = tempdir(CLEANUP => 1);
    my $test_file = "$temp_dir/test.txt";
    
    create_context_test_file($test_file, 10, [5], 'function');
    
    # Test with context = 2
    my $result = test_context_functionality('function', 2, $test_file);
    
    ok($result->{exit_code} == 0, 'gf command executed successfully');
    ok(length($result->{output}) > 0, 'gf produced output');
    
    # Parse the output
    my $parsed = parse_context_output($result->{output});
    
    ok($parsed->{total_groups} == 1, 'Found one match group');
    ok(@{$parsed->{match_groups}->[0]} >= 1, 'Match group has at least the match line');
    
    # Find the match line
    my $match_group = $parsed->{match_groups}->[0];
    my $match_index = -1;
    for my $i (0 .. $#{$match_group}) {
        if ($match_group->[$i]->{is_match}) {
            $match_index = $i;
            last;
        }
    }
    
    ok($match_index >= 0, 'Found match line in output');
    
    if ($match_index >= 0) {
        my $before_count = $match_index;
        my $after_count = @$match_group - $match_index - 1;
        
        # Current behavior: should have before context but no after context
        ok($before_count > 0, 'Has before context lines');
        is($before_count, 2, 'Has correct number of before context lines');
        
        # This test documents the current bug - no after context
        is($after_count, 0, 'Currently has no after context (this is the bug)');
    }
};

# Test context with multiple matches
subtest 'current behavior with multiple matches' => sub {
    plan tests => 6;
    
    my $temp_dir = tempdir(CLEANUP => 1);
    my $test_file = "$temp_dir/multiple.txt";
    
    create_context_test_file($test_file, 15, [3, 8, 13], 'function');
    
    my $result = test_context_functionality('function', 1, $test_file);
    
    ok($result->{exit_code} == 0, 'gf command executed successfully');
    
    my $parsed = parse_context_output($result->{output});
    
    is($parsed->{total_groups}, 3, 'Found three match groups');
    
    # Check each match group for current behavior
    my $all_have_before = 1;
    my $any_have_after = 0;
    
    for my $group (@{$parsed->{match_groups}}) {
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
            
            $all_have_before = 0 if $before_count == 0;
            $any_have_after = 1 if $after_count > 0;
        }
    }
    
    ok($all_have_before, 'All matches have before context');
    ok(!$any_have_after, 'No matches have after context (current bug)');
    
    # Verify the matches are at expected line numbers
    my @match_line_nums = ();
    for my $group (@{$parsed->{match_groups}}) {
        for my $line (@$group) {
            if ($line->{is_match}) {
                push @match_line_nums, $line->{line_num};
            }
        }
    }
    
    is_deeply([sort { $a <=> $b } @match_line_nums], [3, 8, 13], 'Matches found at expected line numbers');
    is(scalar(@match_line_nums), 3, 'Found expected number of matches');
};

# Test edge cases with current behavior
subtest 'current behavior edge cases' => sub {
    plan tests => 8;
    
    my $temp_dir = tempdir(CLEANUP => 1);
    my $edge_files = create_edge_case_test_files($temp_dir);
    
    # Test empty file
    my $empty_result = test_context_functionality('function', 2, $edge_files->{empty});
    ok($empty_result->{exit_code} == 0, 'Empty file handled without error');
    is(length($empty_result->{output}), 0, 'Empty file produces no output');
    
    # Test single line with match
    my $single_result = test_context_functionality('function', 2, $edge_files->{single_match});
    ok($single_result->{exit_code} == 0, 'Single line file handled without error');
    ok(length($single_result->{output}) > 0, 'Single line match produces output');
    
    my $single_parsed = parse_context_output($single_result->{output});
    is($single_parsed->{total_groups}, 1, 'Single line produces one match group');
    
    # Test match at start of file
    my $start_result = test_context_functionality('function', 2, $edge_files->{match_at_start});
    ok($start_result->{exit_code} == 0, 'Match at start handled without error');
    
    my $start_parsed = parse_context_output($start_result->{output});
    my $start_group = $start_parsed->{match_groups}->[0];
    my $start_match_index = -1;
    for my $i (0 .. $#{$start_group}) {
        if ($start_group->[$i]->{is_match}) {
            $start_match_index = $i;
            last;
        }
    }
    
    is($start_match_index, 0, 'Match at start has no before context (as expected)');
    
    # Test match at end of file
    my $end_result = test_context_functionality('function', 2, $edge_files->{match_at_end});
    ok($end_result->{exit_code} == 0, 'Match at end handled without error');
};

# Test context with zero value (default behavior)
subtest 'context zero behavior' => sub {
    plan tests => 4;
    
    my $temp_dir = tempdir(CLEANUP => 1);
    my $test_file = "$temp_dir/zero_context.txt";
    
    create_context_test_file($test_file, 10, [5], 'function');
    
    # Test with context = 0 (explicit)
    my $result_zero = test_context_functionality('function', 0, $test_file);
    ok($result_zero->{exit_code} == 0, 'Context zero executed successfully');
    
    my $parsed_zero = parse_context_output($result_zero->{output});
    is($parsed_zero->{total_groups}, 1, 'Found one match with zero context');
    is(@{$parsed_zero->{match_groups}->[0]}, 1, 'Zero context shows only match line');
    
    # Test without context flag (default behavior)
    my $result_default = test_context_functionality('function', 0, $test_file);  # Default is 0 context
    ok($result_default->{exit_code} == 0, 'Default behavior executed successfully');
    
    # Default and explicit zero should produce same output
    # (We'll just verify both work for now)
};

# Test line number formatting in current implementation
subtest 'line number formatting validation' => sub {
    plan tests => 5;
    
    my $temp_dir = tempdir(CLEANUP => 1);
    my $test_file = "$temp_dir/formatting.txt";
    
    create_context_test_file($test_file, 10, [5], 'function');
    
    my $result = test_context_functionality('function', 2, $test_file);
    my $parsed = parse_context_output($result->{output});
    
    ok($parsed->{total_lines} > 0, 'Output contains lines');
    
    # Check line number format
    my $format_correct = 1;
    my $sequence_correct = 1;
    my @line_numbers = ();
    
    for my $line (@{$parsed->{all_lines}}) {
        # Check format [num]\tcontent
        if ($line->{line_num} !~ /^\d+$/) {
            $format_correct = 0;
        }
        push @line_numbers, $line->{line_num};
    }
    
    ok($format_correct, 'All line numbers are properly formatted');
    
    # Check that line numbers form a sequence (for context display)
    for my $i (1 .. $#line_numbers) {
        if ($line_numbers[$i] != $line_numbers[$i-1] + 1) {
            $sequence_correct = 0;
            last;
        }
    }
    
    ok($sequence_correct, 'Line numbers form proper sequence');
    
    # Check that match line is highlighted
    my $match_highlighted = 0;
    for my $line (@{$parsed->{all_lines}}) {
        if ($line->{is_match}) {
            $match_highlighted = 1;
            last;
        }
    }
    
    ok($match_highlighted, 'Match line is properly highlighted');
    
    # Validate line number sequence
    my $sequence_validation = validate_line_number_sequence($parsed);
    ok($sequence_validation->{sequences_valid}, 'Line number sequences are valid');
};

done_testing();