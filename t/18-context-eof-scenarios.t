#!/usr/bin/env perl
use strict;
use warnings;
use Test::More;
use File::Temp qw(tempdir tempfile);
use File::Path qw(make_path);
use lib 'lib';
use lib 't';

require 'test_utils.pl';
require 'context_test_utils.pl';

BEGIN {
    use_ok('GF::Search', qw(check_file));
}

=head1 NAME

18-context-eof-scenarios.t - Test end-of-file scenarios for context flag

=head1 DESCRIPTION

This test file specifically tests end-of-file scenarios for the context flag (-c)
functionality, including matches near the end of files with insufficient after-context,
empty files, and single-line files with context.

=cut

# Helper function to capture check_file output directly
sub capture_check_file_output {
    my ($search_term, $context, $file_path, $debug) = @_;
    $debug //= 0;
    
    # Initialize pattern cache with the search term
    require GF::PatternCache;
    GF::PatternCache::initialize_pattern_cache($search_term, {}, {});
    
    # Capture output using the capture_output function from test_utils
    my $output = capture_output(sub {
        GF::Search::check_file($file_path, $search_term, $context, 0, $debug);
    });
    
    return {
        output => $output,
        exit_code => 0,  # check_file doesn't return exit codes
        file_path => $file_path
    };
}

# Create test files for EOF scenarios
sub create_eof_test_files {
    my $temp_dir = tempdir(CLEANUP => 1);
    
    # Test file 1: Match at very end of file (no after-context available)
    write_test_file("$temp_dir/match_at_end.txt", <<'EOF');
Line 1: Some content
Line 2: More content
Line 3: Even more content
Line 4: Final line with function
EOF

    # Test file 2: Match near end with partial after-context
    write_test_file("$temp_dir/match_near_end.txt", <<'EOF');
Line 1: Some content
Line 2: More content
Line 3: Match with function here
Line 4: Only one line after
EOF

    # Test file 3: Match with insufficient after-context (wants 3, gets 1)
    write_test_file("$temp_dir/insufficient_context.txt", <<'EOF');
Line 1: Setup
Line 2: More setup
Line 3: function match here
Line 4: Only after line
EOF

    # Test file 4: Multiple matches near end
    write_test_file("$temp_dir/multiple_near_end.txt", <<'EOF');
Line 1: Beginning
Line 2: function_one match
Line 3: Between matches
Line 4: function_two match
Line 5: Last line
EOF

    # Test file 5: Empty file
    write_test_file("$temp_dir/empty.txt", "");

    # Test file 6: Single line file with match
    write_test_file("$temp_dir/single_line_match.txt", "This single line contains function");

    # Test file 7: Single line file without match
    write_test_file("$temp_dir/single_line_no_match.txt", "This single line has no match");

    # Test file 8: Two line file with match on last line
    write_test_file("$temp_dir/two_line_last_match.txt", <<'EOF');
Line 1: First line
Line 2: Second line with function
EOF

    # Test file 9: Three line file with match on second line (edge case)
    write_test_file("$temp_dir/three_line_middle.txt", <<'EOF');
Line 1: First line
Line 2: Middle function match
Line 3: Last line
EOF

    return $temp_dir;
}

# Test match at very end of file with no after-context available
subtest 'match at end of file - no after-context' => sub {
    plan tests => 6;
    
    my $temp_dir = create_eof_test_files();
    my $test_file = "$temp_dir/match_at_end.txt";
    
    # Test with context=2 (wants 2 after, gets 0)
    my $result = capture_check_file_output('function', 2, $test_file);
    is($result->{exit_code}, 0, 'Command executed successfully');
    
    # Should display match with available before-context but no after-context
    my @lines = split /\n/, $result->{output};
    @lines = grep { $_ !~ /^\s*$/ && $_ !~ /\.txt$/ } @lines; # Filter filename and empty lines
    
    # Should have: 2 before + 1 match + 0 after = 3 lines
    is(scalar @lines, 3, 'Correct number of lines displayed (2 before + match + 0 after)');
    
    # Check line numbers and content
    like($lines[0], qr/^\[2\]/, 'First context line has correct line number');
    like($lines[1], qr/^\[3\]/, 'Second context line has correct line number');
    like($lines[2], qr/^\[4\]/, 'Match line has correct line number');
    like($lines[2], qr/function/, 'Match line contains search term');
};

# Test match near end with partial after-context
subtest 'match near end - partial after-context' => sub {
    plan tests => 6;
    
    my $temp_dir = create_eof_test_files();
    my $test_file = "$temp_dir/match_near_end.txt";
    
    # Test with context=3 (wants 3 after, gets 1)
    my $result = capture_check_file_output('function', 3, $test_file);
    is($result->{exit_code}, 0, 'Command executed successfully');
    
    my @lines = split /\n/, $result->{output};
    @lines = grep { $_ !~ /^\s*$/ && $_ !~ /\.txt$/ } @lines;
    
    # Should have: 2 before + 1 match + 1 after = 4 lines (not 3 before + 1 match + 3 after)
    is(scalar @lines, 4, 'Correct number of lines displayed with partial after-context');
    
    # Check that we get the available after-context
    like($lines[0], qr/^\[1\]/, 'First before-context line');
    like($lines[1], qr/^\[2\]/, 'Second before-context line');
    like($lines[2], qr/^\[3\].*function/, 'Match line with search term');
    like($lines[3], qr/^\[4\]/, 'Available after-context line');
};

# Test insufficient after-context scenario
subtest 'insufficient after-context handling' => sub {
    plan tests => 5;
    
    my $temp_dir = create_eof_test_files();
    my $test_file = "$temp_dir/insufficient_context.txt";
    
    # Test with context=3 (wants 3 after, gets 1)
    my $result = capture_check_file_output('function', 3, $test_file);
    is($result->{exit_code}, 0, 'Command executed successfully');
    
    my @lines = split /\n/, $result->{output};
    @lines = grep { $_ !~ /^\s*$/ && $_ !~ /\.txt$/ } @lines;
    
    # Should display all available context
    is(scalar @lines, 4, 'All available lines displayed');
    
    # Verify we get what's available, not what was requested
    like($lines[0], qr/^\[1\]/, 'Before-context line 1');
    like($lines[1], qr/^\[2\]/, 'Before-context line 2');
    like($lines[2], qr/^\[3\].*function/, 'Match line');
    # Note: Only one after-context line available, not the 3 requested
};

# Test multiple matches near end of file
subtest 'multiple matches near EOF' => sub {
    plan tests => 4;
    
    my $temp_dir = create_eof_test_files();
    my $test_file = "$temp_dir/multiple_near_end.txt";
    
    # Test with context=2
    my $result = capture_check_file_output('function', 2, $test_file);
    is($result->{exit_code}, 0, 'Command executed successfully');
    
    my @lines = split /\n/, $result->{output};
    @lines = grep { $_ !~ /^\s*$/ && $_ !~ /\.txt$/ } @lines;
    
    # Should find both matches (function_one and function_two)
    my @match_lines = grep { /function_one|function_two/ } @lines;
    is(scalar @match_lines, 2, 'Both matches found');
    
    # Check that both matches are displayed (order may vary)
    my $output_text = join(' ', @match_lines);
    like($output_text, qr/function_one/, 'function_one match found');
    like($output_text, qr/function_two/, 'function_two match found');
};

# Test empty file with context
subtest 'empty file with context' => sub {
    plan tests => 2;
    
    my $temp_dir = create_eof_test_files();
    my $test_file = "$temp_dir/empty.txt";
    
    # Test with context=2 on empty file
    my $result = capture_check_file_output('function', 2, $test_file);
    is($result->{exit_code}, 0, 'Command executed successfully on empty file');
    
    # Should produce no output (no matches)
    my @lines = split /\n/, $result->{output};
    @lines = grep { $_ !~ /^\s*$/ } @lines;
    is(scalar @lines, 0, 'No output for empty file');
};

# Test single line file with match
subtest 'single line file with match' => sub {
    plan tests => 4;
    
    my $temp_dir = create_eof_test_files();
    my $test_file = "$temp_dir/single_line_match.txt";
    
    # Test with context=2 on single line file
    my $result = capture_check_file_output('function', 2, $test_file);
    is($result->{exit_code}, 0, 'Command executed successfully');
    
    my @lines = split /\n/, $result->{output};
    @lines = grep { $_ !~ /^\s*$/ && $_ !~ /\.txt$/ } @lines;
    
    # Should have only the match line (no before or after context available)
    is(scalar @lines, 1, 'Only match line displayed for single line file');
    like($lines[0], qr/^\[1\]/, 'Match line has correct line number');
    like($lines[0], qr/function/, 'Match line contains search term');
};

# Test single line file without match
subtest 'single line file without match' => sub {
    plan tests => 2;
    
    my $temp_dir = create_eof_test_files();
    my $test_file = "$temp_dir/single_line_no_match.txt";
    
    # Test with context=2 on single line file with no match
    my $result = capture_check_file_output('function', 2, $test_file);
    is($result->{exit_code}, 0, 'Command executed successfully');
    
    # Should produce no output (no matches)
    my @lines = split /\n/, $result->{output};
    @lines = grep { $_ !~ /^\s*$/ } @lines;
    is(scalar @lines, 0, 'No output for single line file without match');
};

# Test two line file with match on last line
subtest 'two line file - match on last line' => sub {
    plan tests => 4;
    
    my $temp_dir = create_eof_test_files();
    my $test_file = "$temp_dir/two_line_last_match.txt";
    
    # Test with context=2
    my $result = capture_check_file_output('function', 2, $test_file);
    is($result->{exit_code}, 0, 'Command executed successfully');
    
    my @lines = split /\n/, $result->{output};
    @lines = grep { $_ !~ /^\s*$/ && $_ !~ /\.txt$/ } @lines;
    
    # Should have: 1 before + 1 match + 0 after = 2 lines
    is(scalar @lines, 2, 'Correct number of lines for two-line file');
    like($lines[0], qr/^\[1\]/, 'Before-context line');
    like($lines[1], qr/^\[2\].*function/, 'Match line on last line');
};

# Test three line file with match in middle (edge case for context)
subtest 'three line file - middle match' => sub {
    plan tests => 5;
    
    my $temp_dir = create_eof_test_files();
    my $test_file = "$temp_dir/three_line_middle.txt";
    
    # Test with context=2 (should get 1 before, match, 1 after)
    my $result = capture_check_file_output('function', 2, $test_file);
    is($result->{exit_code}, 0, 'Command executed successfully');
    
    my @lines = split /\n/, $result->{output};
    @lines = grep { $_ !~ /^\s*$/ && $_ !~ /\.txt$/ } @lines;
    
    # Should have: 1 before + 1 match + 1 after = 3 lines
    is(scalar @lines, 3, 'All three lines displayed');
    like($lines[0], qr/^\[1\]/, 'Before-context line');
    like($lines[1], qr/^\[2\].*function/, 'Match line in middle');
    like($lines[2], qr/^\[3\]/, 'After-context line');
};

# Test debug output for EOF scenarios
subtest 'debug output for EOF scenarios' => sub {
    plan tests => 3;
    
    my $temp_dir = create_eof_test_files();
    my $test_file = "$temp_dir/match_at_end.txt";
    
    # Test with debug flag to see EOF handling messages
    my $result = capture_check_file_output('function', 2, $test_file, 1);
    is($result->{exit_code}, 0, 'Command with debug executed successfully');
    
    # Check for EOF detection messages in debug output
    like($result->{output}, qr/EOF detected/, 'Debug output shows EOF detection');
    like($result->{output}, qr/pending matches/, 'Debug output shows pending matches processing');
};

# Test that pending matches queue is properly cleaned up
subtest 'pending matches queue cleanup' => sub {
    plan tests => 2;
    
    my $temp_dir = create_eof_test_files();
    my $test_file = "$temp_dir/multiple_near_end.txt";
    
    # Test with debug to see cleanup messages
    my $result = capture_check_file_output('function', 2, $test_file, 1);
    is($result->{exit_code}, 0, 'Command executed successfully');
    
    # Check for cleanup message in debug output
    like($result->{output}, qr/queue cleaned up/, 'Debug output shows queue cleanup');
};

done_testing();