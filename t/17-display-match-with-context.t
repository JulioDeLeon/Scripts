#!/usr/bin/env perl
use strict;
use warnings;
use Test::More;
use File::Temp qw(tempdir tempfile);
use lib 'lib';
use lib 't';

require 'test_utils.pl';

BEGIN {
    use_ok('GF::Search', qw(buffer_output flush_output_buffer));
    use_ok('GF::PatternCache', qw(initialize_pattern_cache));
}

=head1 NAME

17-display-match-with-context.t - Unit tests for _display_match_with_context function

=head1 DESCRIPTION

This test file contains unit tests for the enhanced _display_match_with_context function
that now supports both before and after context display. Tests cover various context
values, formatting, line number display, and edge cases.

=cut

# Helper function to capture buffered output
sub capture_display_output {
    my ($buffer_ref, $match_line, $match_line_num, $term, $context, $after_context_ref) = @_;
    
    # Initialize the pattern cache with the search term
    GF::PatternCache::initialize_pattern_cache($term, {}, {});
    
    # Clear any existing buffer
    GF::Search::flush_output_buffer();
    
    # Call the function
    GF::Search::_display_match_with_context($buffer_ref, $match_line, $match_line_num, $term, $context, $after_context_ref, undef);
    
    # Capture the buffered output
    my $output = '';
    {
        local *STDOUT;
        open STDOUT, '>', \$output or die "Cannot redirect STDOUT: $!";
        GF::Search::flush_output_buffer();
    }
    
    return $output;
}

# Test context = 0 (no context)
subtest 'context zero - no context display' => sub {
    plan tests => 4;
    
    my @buffer = ();
    my $match_line = "    return \$param * 2;  # This line contains 'function'";
    my $match_line_num = 9;
    my $term = 'function';
    my $context = 0;
    my @after_context = ();
    
    my $output = capture_display_output(\@buffer, $match_line, $match_line_num, $term, $context, \@after_context);
    
    # Should only display the match line with line number
    like($output, qr/\[9\]/, 'Match line has correct line number format');
    like($output, qr/function/, 'Match line contains search term');
    unlike($output, qr/\[8\]/, 'No before context displayed');
    unlike($output, qr/\[10\]/, 'No after context displayed');
};

# Test context = 1 with before context only
subtest 'context one - before context only' => sub {
    plan tests => 5;
    
    my @buffer = (
        { line_num => 7, content => 'sub test_function {' },
        { line_num => 8, content => '    my $param = shift;' },
        { line_num => 9, content => '    return $param * 2;  # function' }
    );
    my $match_line = "    return \$param * 2;  # This line contains 'function'";
    my $match_line_num = 9;
    my $term = 'function';
    my $context = 1;
    my @after_context = ();
    
    my $output = capture_display_output(\@buffer, $match_line, $match_line_num, $term, $context, \@after_context);
    
    like($output, qr/\[8\].*my \$param = shift/, 'Before context line displayed');
    like($output, qr/\[9\].*function/, 'Match line displayed');
    unlike($output, qr/\[7\]/, 'Only 1 line of before context shown');
    unlike($output, qr/\[10\]/, 'No after context displayed when empty');
    like($output, qr/\n\n$/, 'Extra spacing added for context display');
};

# Test context = 2 with both before and after context
subtest 'context two - before and after context' => sub {
    plan tests => 7;
    
    my @buffer = (
        { line_num => 6, content => '# Function documentation' },
        { line_num => 7, content => 'sub test_function {' },
        { line_num => 8, content => '    my $param = shift;' },
        { line_num => 9, content => '    return $param * 2;  # function' }
    );
    my $match_line = "    return \$param * 2;  # This line contains 'function'";
    my $match_line_num = 9;
    my $term = 'function';
    my $context = 2;
    my @after_context = (
        { line_num => 10, content => '}' },
        { line_num => 11, content => '# End of function' }
    );
    
    my $output = capture_display_output(\@buffer, $match_line, $match_line_num, $term, $context, \@after_context);
    
    like($output, qr/\[7\].*sub test_function/, 'First before context line displayed');
    like($output, qr/\[8\].*my \$param = shift/, 'Second before context line displayed');
    like($output, qr/\[9\].*function/, 'Match line displayed');
    like($output, qr/\[10\].*}/, 'First after context line displayed');
    like($output, qr/\[11\].*End of function/, 'Second after context line displayed');
    unlike($output, qr/\[6\]/, 'Only 2 lines of before context shown');
    like($output, qr/\n\n$/, 'Extra spacing added for context display');
};

# Test context = 3 with partial after context (end of file scenario)
subtest 'context three - partial after context' => sub {
    plan tests => 6;
    
    my @buffer = (
        { line_num => 7, content => 'Line 7' },
        { line_num => 8, content => 'Line 8' },
        { line_num => 9, content => 'Line 9' },
        { line_num => 10, content => 'Line 10 with function' }
    );
    my $match_line = "Line 10 with function";
    my $match_line_num = 10;
    my $term = 'function';
    my $context = 3;
    my @after_context = (
        { line_num => 11, content => 'Line 11' }
        # Only 1 line of after context available instead of 3
    );
    
    my $output = capture_display_output(\@buffer, $match_line, $match_line_num, $term, $context, \@after_context);
    
    like($output, qr/\[8\].*Line 8/, 'Before context line 8 displayed');
    like($output, qr/\[9\].*Line 9/, 'Before context line 9 displayed');
    like($output, qr/\[10\].*function/, 'Match line displayed');
    like($output, qr/\[11\].*Line 11/, 'Available after context displayed');
    unlike($output, qr/\[12\]/, 'No non-existent after context displayed');
    like($output, qr/\n\n$/, 'Extra spacing added for context display');
};

# Test context = 5 with no after context (empty array)
subtest 'context five - no after context available' => sub {
    plan tests => 5;
    
    my @buffer = (
        { line_num => 5, content => 'Line 5' },
        { line_num => 6, content => 'Line 6' },
        { line_num => 7, content => 'Line 7' },
        { line_num => 8, content => 'Line 8' },
        { line_num => 9, content => 'Line 9' },
        { line_num => 10, content => 'Line 10 with function' }
    );
    my $match_line = "Line 10 with function";
    my $match_line_num = 10;
    my $term = 'function';
    my $context = 5;
    my @after_context = (); # No after context available
    
    my $output = capture_display_output(\@buffer, $match_line, $match_line_num, $term, $context, \@after_context);
    
    like($output, qr/\[5\].*Line 5/, 'Before context starts at correct line');
    like($output, qr/\[9\].*Line 9/, 'Before context ends at correct line');
    like($output, qr/\[10\].*function/, 'Match line displayed');
    unlike($output, qr/\[11\]/, 'No after context displayed when empty');
    like($output, qr/\n\n$/, 'Extra spacing added for context display');
};

# Test line number formatting
subtest 'line number formatting validation' => sub {
    plan tests => 4;
    
    my @buffer = (
        { line_num => 99, content => 'Line 99' },
        { line_num => 100, content => 'Line 100 with function' }
    );
    my $match_line = "Line 100 with function";
    my $match_line_num = 100;
    my $term = 'function';
    my $context = 1;
    my @after_context = (
        { line_num => 101, content => 'Line 101' }
    );
    
    my $output = capture_display_output(\@buffer, $match_line, $match_line_num, $term, $context, \@after_context);
    
    like($output, qr/\[99\]/, 'Before context has correct line number format');
    like($output, qr/\[100\]/, 'Match line has correct line number format');
    like($output, qr/\[101\]/, 'After context has correct line number format');
    
    # Check that line numbers are in square brackets
    my @lines = split /\n/, $output;
    my $content_lines = grep { $_ =~ /\[\d+\]/ } @lines;
    is($content_lines, 3, 'All content lines have proper line number format');
};

# Test match highlighting preservation
subtest 'match highlighting preservation' => sub {
    plan tests => 3;
    
    my @buffer = (
        { line_num => 8, content => 'before line' }
    );
    my $match_line = "This line contains function keyword";
    my $match_line_num = 9;
    my $term = 'function';
    my $context = 1;
    my @after_context = (
        { line_num => 10, content => 'after line' }
    );
    
    my $output = capture_display_output(\@buffer, $match_line, $match_line_num, $term, $context, \@after_context);
    
    # The match line should be processed by print_str which adds highlighting
    like($output, qr/\[9\]/, 'Match line has line number');
    like($output, qr/function/, 'Match line contains search term');
    
    # Context lines should not have highlighting
    unlike($output, qr/\[8\].*\e\[/, 'Before context line has no ANSI codes');
};

# Test empty buffer scenario
subtest 'empty buffer scenario' => sub {
    plan tests => 3;
    
    my @buffer = (); # Empty buffer
    my $match_line = "function at start of file";
    my $match_line_num = 1;
    my $term = 'function';
    my $context = 2;
    my @after_context = (
        { line_num => 2, content => 'Line 2' },
        { line_num => 3, content => 'Line 3' }
    );
    
    my $output = capture_display_output(\@buffer, $match_line, $match_line_num, $term, $context, \@after_context);
    
    unlike($output, qr/\[0\]/, 'No invalid before context displayed');
    like($output, qr/\[1\].*function/, 'Match line displayed correctly');
    like($output, qr/\[2\].*Line 2/, 'After context displayed correctly');
};

# Test buffer with insufficient before context
subtest 'insufficient before context in buffer' => sub {
    plan tests => 4;
    
    my @buffer = (
        { line_num => 4, content => 'Line 4' } # Only 1 line in buffer, but context=3
    );
    my $match_line = "Line 5 with function";
    my $match_line_num = 5;
    my $term = 'function';
    my $context = 3;
    my @after_context = (
        { line_num => 6, content => 'Line 6' },
        { line_num => 7, content => 'Line 7' },
        { line_num => 8, content => 'Line 8' }
    );
    
    my $output = capture_display_output(\@buffer, $match_line, $match_line_num, $term, $context, \@after_context);
    
    like($output, qr/\[4\].*Line 4/, 'Available before context displayed');
    unlike($output, qr/\[2\]/, 'Non-existent before context not displayed');
    like($output, qr/\[5\].*function/, 'Match line displayed');
    like($output, qr/\[8\].*Line 8/, 'Full after context displayed');
};

# Test whitespace trimming for context = 0
subtest 'whitespace trimming for no context' => sub {
    plan tests => 2;
    
    my @buffer = ();
    my $match_line = "   function with leading/trailing spaces   ";
    my $match_line_num = 1;
    my $term = 'function';
    my $context = 0;
    my @after_context = ();
    
    my $output = capture_display_output(\@buffer, $match_line, $match_line_num, $term, $context, \@after_context);
    
    # For context = 0, whitespace should be trimmed
    like($output, qr/\[1\]/, 'Line number format preserved');
    # Note: The actual trimming happens in the function, we're testing the behavior
    unlike($output, qr/\n\n$/, 'No extra spacing for context = 0');
};

done_testing();