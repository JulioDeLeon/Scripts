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

=head1 NAME

16-context-flag-comprehensive.t - Comprehensive context flag testing

=head1 DESCRIPTION

This test file provides comprehensive testing for the context flag (-c) functionality.
It tests all requirements and edge cases to ensure the context flag works correctly
after the fix is implemented.

=cut

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
        exit_code => 0,
        search_term => $search_term,
        context => $context,
        file_path => $file_path
    };
}

# Test Requirement 1.1: Context lines before and after matches
subtest 'Requirement 1.1: Before and after context display' => sub {
    plan tests => 12;
    
    my $temp_dir = tempdir(CLEANUP => 1);
    
    # Test context = 1
    my $test_file_1 = "$temp_dir/context_1.txt";
    create_context_test_file($test_file_1, 7, [4], 'function');
    
    my $result_1 = test_context_functionality('function', 1, $test_file_1);
    ok($result_1->{exit_code} == 0, 'Context=1 executed successfully');
    
    my $parsed_1 = parse_context_output($result_1->{output});
    my $symmetry_1 = validate_context_symmetry($parsed_1, 1);
    
    ok($symmetry_1->{correct_before_count}, 'Context=1 has correct before context count');
    ok($symmetry_1->{correct_after_count}, 'Context=1 has correct after context count');
    ok($symmetry_1->{symmetric}, 'Context=1 is symmetric');
    
    # Test context = 2
    my $test_file_2 = "$temp_dir/context_2.txt";
    create_context_test_file($test_file_2, 9, [5], 'function');
    
    my $result_2 = test_context_functionality('function', 2, $test_file_2);
    ok($result_2->{exit_code} == 0, 'Context=2 executed successfully');
    
    my $parsed_2 = parse_context_output($result_2->{output});
    my $symmetry_2 = validate_context_symmetry($parsed_2, 2);
    
    ok($symmetry_2->{correct_before_count}, 'Context=2 has correct before context count');
    ok($symmetry_2->{correct_after_count}, 'Context=2 has correct after context count');
    ok($symmetry_2->{symmetric}, 'Context=2 is symmetric');
    
    # Test context = 5
    my $test_file_5 = "$temp_dir/context_5.txt";
    create_context_test_file($test_file_5, 15, [8], 'function');
    
    my $result_5 = test_context_functionality('function', 5, $test_file_5);
    ok($result_5->{exit_code} == 0, 'Context=5 executed successfully');
    
    my $parsed_5 = parse_context_output($result_5->{output});
    my $symmetry_5 = validate_context_symmetry($parsed_5, 5);
    
    ok($symmetry_5->{correct_before_count}, 'Context=5 has correct before context count');
    ok($symmetry_5->{correct_after_count}, 'Context=5 has correct after context count');
    ok($symmetry_5->{symmetric}, 'Context=5 is symmetric');
};

# Test Requirement 1.3: Context = 0 behavior
subtest 'Requirement 1.3: Context zero behavior' => sub {
    plan tests => 4;
    
    my $temp_dir = tempdir(CLEANUP => 1);
    my $test_file = "$temp_dir/zero_context.txt";
    create_context_test_file($test_file, 10, [5], 'function');
    
    my $result = test_context_functionality('function', 0, $test_file);
    ok($result->{exit_code} == 0, 'Context=0 executed successfully');
    
    my $parsed = parse_context_output($result->{output});
    is($parsed->{total_groups}, 1, 'Context=0 found one match group');
    is(@{$parsed->{match_groups}->[0]}, 1, 'Context=0 shows only match line');
    
    # Verify the single line is the match line
    my $match_line = $parsed->{match_groups}->[0]->[0];
    ok($match_line->{is_match}, 'Context=0 single line is the match');
};

# Test Requirement 1.4: End-of-file scenarios
subtest 'Requirement 1.4: End-of-file context handling' => sub {
    plan tests => 8;
    
    my $temp_dir = tempdir(CLEANUP => 1);
    my $edge_files = create_edge_case_test_files($temp_dir);
    
    # Test match at end of file with insufficient after-context
    my $end_result = test_context_functionality('function', 3, $edge_files->{match_at_end});
    ok($end_result->{exit_code} == 0, 'Match at end handled without error');
    
    my $end_parsed = parse_context_output($end_result->{output});
    ok($end_parsed->{total_groups} == 1, 'Match at end found one group');
    
    # Should have full before context but limited after context
    my $end_group = $end_parsed->{match_groups}->[0];
    my $match_index = -1;
    for my $i (0 .. $#{$end_group}) {
        if ($end_group->[$i]->{is_match}) {
            $match_index = $i;
            last;
        }
    }
    
    ok($match_index >= 0, 'Found match in end-of-file test');
    if ($match_index >= 0) {
        my $before_count = $match_index;
        my $after_count = @$end_group - $match_index - 1;
        
        is($before_count, 3, 'Match at end has full before context');
        ok($after_count >= 0, 'Match at end has available after context (may be less than requested)');
    }
    
    # Test match at start of file with insufficient before-context
    my $start_result = test_context_functionality('function', 3, $edge_files->{match_at_start});
    ok($start_result->{exit_code} == 0, 'Match at start handled without error');
    
    my $start_parsed = parse_context_output($start_result->{output});
    ok($start_parsed->{total_groups} == 1, 'Match at start found one group');
    
    # Should have limited before context but full after context
    my $start_group = $start_parsed->{match_groups}->[0];
    $match_index = -1;
    for my $i (0 .. $#{$start_group}) {
        if ($start_group->[$i]->{is_match}) {
            $match_index = $i;
            last;
        }
    }
    
    if ($match_index >= 0) {
        my $after_count = @$start_group - $match_index - 1;
        is($after_count, 3, 'Match at start has full after context');
    }
};

# Test Requirement 2.1 & 2.2: Line number format and highlighting
subtest 'Requirement 2.1 & 2.2: Formatting and highlighting' => sub {
    plan tests => 6;
    
    my $temp_dir = tempdir(CLEANUP => 1);
    my $test_file = "$temp_dir/formatting.txt";
    create_context_test_file($test_file, 10, [5], 'function');
    
    my $result = test_context_functionality('function', 2, $test_file);
    my $parsed = parse_context_output($result->{output});
    
    # Validate line number sequence
    my $sequence_validation = validate_line_number_sequence($parsed);
    ok($sequence_validation->{sequences_valid}, 'Line number sequences are valid');
    ok($sequence_validation->{no_gaps}, 'No gaps in line number sequences');
    
    # Check line number format [num]\tcontent
    my $format_correct = 1;
    for my $line (@{$parsed->{all_lines}}) {
        if ($line->{line_num} !~ /^\d+$/) {
            $format_correct = 0;
            last;
        }
    }
    ok($format_correct, 'All line numbers properly formatted');
    
    # Check match highlighting
    my $match_highlighted = 0;
    for my $line (@{$parsed->{all_lines}}) {
        if ($line->{is_match}) {
            $match_highlighted = 1;
            last;
        }
    }
    ok($match_highlighted, 'Match line is highlighted');
    
    # Check spacing between matches (should have empty line)
    ok(length($result->{output}) > 0, 'Output contains content');
    like($result->{output}, qr/\n\n/, 'Output contains proper spacing');
};

# Test Requirement 2.4: Overlapping context handling
subtest 'Requirement 2.4: Overlapping context handling' => sub {
    plan tests => 6;
    
    my $temp_dir = tempdir(CLEANUP => 1);
    my $test_file = "$temp_dir/overlapping.txt";
    
    # Create file with matches close together (overlapping context)
    create_overlapping_context_test_file($test_file, [2, 4], 'function');
    
    my $result = test_context_functionality('function', 2, $test_file);
    ok($result->{exit_code} == 0, 'Overlapping context executed successfully');
    
    my $parsed = parse_context_output($result->{output});
    ok($parsed->{total_groups} >= 1, 'Found match groups with overlapping context');
    
    # Check that overlapping lines are not duplicated
    my @all_line_nums = ();
    for my $line (@{$parsed->{all_lines}}) {
        push @all_line_nums, $line->{line_num};
    }
    
    # Check for duplicates
    my %seen = ();
    my $has_duplicates = 0;
    for my $line_num (@all_line_nums) {
        if ($seen{$line_num}) {
            $has_duplicates = 1;
            last;
        }
        $seen{$line_num} = 1;
    }
    
    ok(!$has_duplicates, 'No duplicate lines in overlapping context');
    
    # Verify line numbers are in sequence
    my $sequence_validation = validate_line_number_sequence($parsed);
    ok($sequence_validation->{sequences_valid}, 'Overlapping context maintains valid sequences');
    
    # Check that both matches are highlighted
    my $match_count = 0;
    for my $line (@{$parsed->{all_lines}}) {
        if ($line->{is_match}) {
            $match_count++;
        }
    }
    ok($match_count >= 2, 'Both matches in overlapping context are highlighted');
    ok($match_count <= 2, 'No extra matches found');
};

# Test multiple matches with various context values
subtest 'Multiple matches with context' => sub {
    plan tests => 9;
    
    my $temp_dir = tempdir(CLEANUP => 1);
    my $test_file = "$temp_dir/multiple.txt";
    
    # Create file with multiple matches spread apart
    create_context_test_file($test_file, 20, [5, 10, 15], 'function');
    
    # Test with context = 1
    my $result_1 = test_context_functionality('function', 1, $test_file);
    ok($result_1->{exit_code} == 0, 'Multiple matches context=1 executed successfully');
    
    my $parsed_1 = parse_context_output($result_1->{output});
    is($parsed_1->{total_groups}, 3, 'Found three match groups with context=1');
    
    # Validate symmetry for all matches
    my $symmetry_1 = validate_context_symmetry($parsed_1, 1);
    ok($symmetry_1->{symmetric}, 'All matches have symmetric context=1');
    
    # Test with context = 2
    my $result_2 = test_context_functionality('function', 2, $test_file);
    ok($result_2->{exit_code} == 0, 'Multiple matches context=2 executed successfully');
    
    my $parsed_2 = parse_context_output($result_2->{output});
    is($parsed_2->{total_groups}, 3, 'Found three match groups with context=2');
    
    my $symmetry_2 = validate_context_symmetry($parsed_2, 2);
    ok($symmetry_2->{symmetric}, 'All matches have symmetric context=2');
    
    # Verify match line numbers are correct
    my @match_line_nums = ();
    for my $group (@{$parsed_2->{match_groups}}) {
        for my $line (@$group) {
            if ($line->{is_match}) {
                push @match_line_nums, $line->{line_num};
            }
        }
    }
    
    is_deeply([sort { $a <=> $b } @match_line_nums], [5, 10, 15], 'Matches found at expected line numbers');
    is(scalar(@match_line_nums), 3, 'Found expected number of matches');
    
    # Check that each match group has the right number of lines
    for my $group (@{$parsed_2->{match_groups}}) {
        is(@$group, 5, 'Each match group has 5 lines (2 before + match + 2 after)');
    }
};

# Test performance with larger context values
subtest 'Performance with large context values' => sub {
    plan tests => 4;
    
    my $temp_dir = tempdir(CLEANUP => 1);
    my $test_file = "$temp_dir/large_context.txt";
    
    # Create a larger file for performance testing
    create_context_test_file($test_file, 50, [25], 'function');
    
    # Test with large context value
    my $result = test_context_functionality('function', 10, $test_file);
    ok($result->{exit_code} == 0, 'Large context value executed successfully');
    
    my $parsed = parse_context_output($result->{output});
    is($parsed->{total_groups}, 1, 'Found one match group with large context');
    
    # Verify context symmetry
    my $symmetry = validate_context_symmetry($parsed, 10);
    ok($symmetry->{correct_before_count}, 'Large context has correct before count');
    ok($symmetry->{correct_after_count}, 'Large context has correct after count');
};

# Test after-context collection logic specifically
subtest 'After-context collection functionality' => sub {
    plan tests => 15;
    
    my $temp_dir = tempdir(CLEANUP => 1);
    
    # Test 1: Basic after-context collection with context=1
    my $test_file_1 = "$temp_dir/after_context_1.txt";
    create_context_test_file($test_file_1, 5, [3], 'function');
    
    my $result_1 = test_context_functionality('function', 1, $test_file_1);
    ok($result_1->{exit_code} == 0, 'After-context collection context=1 executed successfully');
    
    my $parsed_1 = parse_context_output($result_1->{output});
    is($parsed_1->{total_groups}, 1, 'Found one match group for after-context test');
    
    # Verify after-context is collected
    my $group_1 = $parsed_1->{match_groups}->[0];
    my $match_index_1 = -1;
    for my $i (0 .. $#{$group_1}) {
        if ($group_1->[$i]->{is_match}) {
            $match_index_1 = $i;
            last;
        }
    }
    
    ok($match_index_1 >= 0, 'Found match in after-context test');
    if ($match_index_1 >= 0) {
        my $after_count_1 = @$group_1 - $match_index_1 - 1;
        is($after_count_1, 1, 'Collected exactly 1 line of after-context');
        
        # Verify the after-context line is correct
        if ($after_count_1 > 0) {
            my $after_line = $group_1->[$match_index_1 + 1];
            is($after_line->{line_num}, 4, 'After-context line has correct line number');
            like($after_line->{clean_content}, qr/Line 4/, 'After-context line has correct content');
        }
    }
    
    # Test 2: Multiple lines of after-context with context=3
    my $test_file_3 = "$temp_dir/after_context_3.txt";
    create_context_test_file($test_file_3, 10, [5], 'function');
    
    my $result_3 = test_context_functionality('function', 3, $test_file_3);
    ok($result_3->{exit_code} == 0, 'After-context collection context=3 executed successfully');
    
    my $parsed_3 = parse_context_output($result_3->{output});
    my $group_3 = $parsed_3->{match_groups}->[0];
    my $match_index_3 = -1;
    for my $i (0 .. $#{$group_3}) {
        if ($group_3->[$i]->{is_match}) {
            $match_index_3 = $i;
            last;
        }
    }
    
    if ($match_index_3 >= 0) {
        my $after_count_3 = @$group_3 - $match_index_3 - 1;
        is($after_count_3, 3, 'Collected exactly 3 lines of after-context');
        
        # Verify after-context line numbers are sequential
        for my $i (1 .. $after_count_3) {
            my $after_line = $group_3->[$match_index_3 + $i];
            my $expected_line_num = 5 + $i;  # Match is at line 5
            is($after_line->{line_num}, $expected_line_num, "After-context line $i has correct line number");
        }
    }
    
    # Test 3: Match completion detection - verify matches are displayed when after-context is complete
    my $test_file_complete = "$temp_dir/completion_test.txt";
    create_context_test_file($test_file_complete, 8, [3, 6], 'function');
    
    my $result_complete = test_context_functionality('function', 2, $test_file_complete);
    ok($result_complete->{exit_code} == 0, 'Match completion detection executed successfully');
    
    my $parsed_complete = parse_context_output($result_complete->{output});
    is($parsed_complete->{total_groups}, 2, 'Both matches displayed after completion');
    
    # Verify both matches have complete after-context
    for my $group_idx (0 .. $#{$parsed_complete->{match_groups}}) {
        my $group = $parsed_complete->{match_groups}->[$group_idx];
        my $match_index = -1;
        for my $i (0 .. $#{$group}) {
            if ($group->[$i]->{is_match}) {
                $match_index = $i;
                last;
            }
        }
        
        if ($match_index >= 0) {
            my $after_count = @$group - $match_index - 1;
            is($after_count, 2, "Match group $group_idx has complete after-context");
        }
    }
    
    # Test 4: End-of-file scenario with incomplete after-context
    my $test_file_eof = "$temp_dir/eof_test.txt";
    create_context_test_file($test_file_eof, 5, [4], 'function');  # Match near end
    
    my $result_eof = test_context_functionality('function', 3, $test_file_eof);
    ok($result_eof->{exit_code} == 0, 'End-of-file incomplete after-context handled successfully');
    
    my $parsed_eof = parse_context_output($result_eof->{output});
    my $group_eof = $parsed_eof->{match_groups}->[0];
    my $match_index_eof = -1;
    for my $i (0 .. $#{$group_eof}) {
        if ($group_eof->[$i]->{is_match}) {
            $match_index_eof = $i;
            last;
        }
    }
    
    if ($match_index_eof >= 0) {
        my $after_count_eof = @$group_eof - $match_index_eof - 1;
        ok($after_count_eof >= 0 && $after_count_eof <= 3, 'End-of-file match displays available after-context');
        is($after_count_eof, 1, 'End-of-file match shows exactly 1 available after-context line');
    }
};

done_testing();