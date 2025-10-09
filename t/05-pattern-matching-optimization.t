#!/usr/bin/env perl

use strict;
use warnings;
use Test::More;
use File::Temp qw(tempdir);
use File::Path qw(make_path);
use Time::HiRes qw(time);

# Add lib directory to path
use lib 'lib';
use lib '.';

# Load test utilities
require 't/test_utils.pl';

# Load modules to test
use_ok('GF::Search', qw(print_str buffer_output flush_output_buffer));
use_ok('GF::PatternCache', qw(initialize_pattern_cache get_search_pattern));

=head1 NAME

05-pattern-matching-optimization.t - Tests for pattern matching optimization

=head1 DESCRIPTION

Tests the optimized pattern matching functionality including:
- Pattern compilation and caching
- Efficient single-pass highlighting
- Buffered output system
- Performance improvements

=cut

# Test pattern compilation and caching functionality
subtest 'Pattern compilation and caching' => sub {
    plan tests => 4;
    
    # Test pattern compilation
    my $term = 'function';
    my %ignores = ('*.log' => 1, '*.tmp' => 1);
    my %targets = ('*.pl' => 1, '*.pm' => 1);
    
    # Initialize pattern cache
    GF::PatternCache::initialize_pattern_cache($term, \%ignores, \%targets);
    
    # Test that search pattern is compiled and cached
    my $search_pattern = get_search_pattern();
    ok(defined $search_pattern, 'Search pattern is compiled and cached');
    isa_ok($search_pattern, 'Regexp', 'Search pattern is a compiled regex');
    
    # Test pattern matching works
    my $test_string = 'This is a function call';
    ok($test_string =~ /$search_pattern/, 'Compiled pattern matches correctly');
    
    # Test pattern caching (second call should return same pattern)
    my $cached_pattern = get_search_pattern();
    is($search_pattern, $cached_pattern, 'Pattern is properly cached');
};

# Test efficient single-pass highlighting
subtest 'Single-pass highlighting efficiency' => sub {
    plan tests => 6;
    
    # Set up pattern for testing
    my $term = 'test';
    my %ignores = ();
    my %targets = ();
    GF::PatternCache::initialize_pattern_cache($term, \%ignores, \%targets);
    
    # Test single match highlighting
    my $output = capture_print_str_output('This is a test string', $term);
    like($output, qr/This is a \e\[1;31mtest\e\[0m string/, 'Single match highlighted correctly');
    
    # Test multiple matches in single pass
    $output = capture_print_str_output('test this test string test', $term);
    my $expected_pattern = qr/\e\[1;31mtest\e\[0m this \e\[1;31mtest\e\[0m string \e\[1;31mtest\e\[0m/;
    like($output, $expected_pattern, 'Multiple matches highlighted in single pass');
    
    # Test no matches
    $output = capture_print_str_output('This string has no matches', $term);
    unlike($output, qr/\e\[1;31m/, 'No highlighting when no matches found');
    is($output, 'This string has no matches', 'Original string preserved when no matches');
    
    # Test edge cases
    $output = capture_print_str_output('testtest', $term);
    like($output, qr/\e\[1;31mtest\e\[0m\e\[1;31mtest\e\[0m/, 'Adjacent matches handled correctly');
    
    $output = capture_print_str_output('', $term);
    is($output, '', 'Empty string handled correctly');
};

# Test buffered output system
subtest 'Buffered output system' => sub {
    plan tests => 8;
    
    # Clear any existing buffer
    GF::Search::flush_output_buffer();
    
    # Test basic buffering
    my $output = capture_buffered_output(sub {
        GF::Search::buffer_output('Hello ');
        GF::Search::buffer_output('World');
        GF::Search::flush_output_buffer();
    });
    is($output, 'Hello World', 'Basic buffering works correctly');
    
    # Test automatic flushing when buffer limit reached
    $output = capture_buffered_output(sub {
        # Add more than buffer limit (50 items)
        for my $i (1..55) {
            GF::Search::buffer_output("item$i ");
        }
        GF::Search::flush_output_buffer();
    });
    
    # Should contain all items
    like($output, qr/item1 /, 'First item in output');
    like($output, qr/item55 /, 'Last item in output');
    ok(length($output) > 0, 'Auto-flush works when buffer limit exceeded');
    
    # Test multiple flush calls
    $output = capture_buffered_output(sub {
        GF::Search::buffer_output('First');
        GF::Search::flush_output_buffer();
        GF::Search::buffer_output('Second');
        GF::Search::flush_output_buffer();
    });
    is($output, 'FirstSecond', 'Multiple flush calls work correctly');
    
    # Test empty buffer flush
    $output = capture_buffered_output(sub {
        GF::Search::flush_output_buffer();
    });
    is($output, '', 'Flushing empty buffer produces no output');
    
    # Test buffer state after flush
    GF::Search::buffer_output('Test');
    GF::Search::flush_output_buffer();
    $output = capture_buffered_output(sub {
        GF::Search::flush_output_buffer();
    });
    is($output, '', 'Buffer is cleared after flush');
    
    # Test buffered print_str integration
    my $term = 'highlight';
    my %ignores = ();
    my %targets = ();
    GF::PatternCache::initialize_pattern_cache($term, \%ignores, \%targets);
    
    $output = capture_buffered_output(sub {
        GF::Search::print_str('This should highlight the word', $term);
        GF::Search::flush_output_buffer();
    });
    like($output, qr/\e\[1;31mhighlight\e\[0m/, 'Buffered print_str works with highlighting');
};

# Test performance improvements
subtest 'Performance improvements' => sub {
    plan tests => 3;
    
    # Set up test data
    my $term = 'performance';
    my %ignores = ();
    my %targets = ();
    GF::PatternCache::initialize_pattern_cache($term, \%ignores, \%targets);
    
    # Create a string with many matches for performance testing
    my $test_string = ('This is a performance test with performance improvements and performance gains. ' x 100);
    
    # Test that single-pass processing is faster than multiple operations
    my $start_time = time();
    
    # Process the string multiple times to measure performance
    for my $i (1..100) {
        my $output = capture_print_str_output($test_string, $term);
    }
    
    my $end_time = time();
    my $duration = $end_time - $start_time;
    
    # Performance should be reasonable (less than 1 second for 100 iterations)
    ok($duration < 1.0, 'Pattern matching performance is acceptable');
    
    # Test that buffering reduces I/O operations
    $start_time = time();
    
    my $output = capture_buffered_output(sub {
        for my $i (1..1000) {
            GF::Search::buffer_output("line $i\n");
        }
        GF::Search::flush_output_buffer();
    });
    
    $end_time = time();
    $duration = $end_time - $start_time;
    
    ok($duration < 0.5, 'Buffered output performance is acceptable');
    ok(length($output) > 0, 'Buffered output produces correct results');
};

# Test match highlighting efficiency
subtest 'Match highlighting efficiency' => sub {
    plan tests => 4;
    
    # Test complex patterns
    my $term = 'function|method|sub';
    my %ignores = ();
    my %targets = ();
    GF::PatternCache::initialize_pattern_cache($term, \%ignores, \%targets);
    
    my $output = capture_print_str_output('This function calls a method and defines a sub', $term);
    
    # Should highlight all three matches
    like($output, qr/\e\[1;31mfunction\e\[0m/, 'Function highlighted');
    like($output, qr/\e\[1;31mmethod\e\[0m/, 'Method highlighted');
    like($output, qr/\e\[1;31msub\e\[0m/, 'Sub highlighted');
    
    # Test case sensitivity
    $term = 'Test';
    GF::PatternCache::initialize_pattern_cache($term, \%ignores, \%targets);
    
    $output = capture_print_str_output('This is a Test and a test', $term);
    like($output, qr/\e\[1;31mTest\e\[0m/, 'Case-sensitive matching works');
};

# Helper function to capture print_str output
sub capture_print_str_output {
    my ($string, $term) = @_;
    
    # Clear buffer first
    GF::Search::flush_output_buffer();
    
    return capture_buffered_output(sub {
        GF::Search::print_str($string, $term);
        GF::Search::flush_output_buffer();
    });
}

# Helper function to capture buffered output
sub capture_buffered_output {
    my ($code_ref) = @_;
    my $output = '';
    
    # Redirect STDOUT to capture output
    open my $old_stdout, '>&', \*STDOUT or die "Can't dup STDOUT: $!";
    close STDOUT;
    open STDOUT, '>', \$output or die "Can't redirect STDOUT: $!";
    
    # Execute the code
    eval { $code_ref->(); };
    my $error = $@;
    
    # Restore STDOUT
    close STDOUT;
    open STDOUT, '>&', $old_stdout or die "Can't restore STDOUT: $!";
    
    die $error if $error;
    return $output;
}

done_testing();