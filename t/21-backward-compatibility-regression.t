#!/usr/bin/env perl

use strict;
use warnings;
use Test::More;
use lib 'lib';
use GF::Search;
use GF::PatternCache;
use File::Temp qw(tempfile);

=head1 NAME

21-backward-compatibility-regression.t - Regression tests for backward compatibility

=head1 DESCRIPTION

This test file ensures that the context flag fix does not break existing functionality.
Tests focus on the core _display_match_with_context function to verify backward compatibility.

=cut

# Initialize pattern cache for testing
GF::PatternCache::initialize_pattern_cache("target", {}, {});

plan tests => 4;

# Helper function to capture output
sub capture_output {
    my $code = shift;
    my $output = '';
    {
        local *STDOUT;
        open STDOUT, '>', \$output or die "Cannot redirect STDOUT: $!";
        $code->();
        GF::Search::flush_output_buffer();
    }
    return $output;
}

# Test 1: Test _display_match_with_context function with context=0 (backward compatibility)
{
    my @buffer = (
        { line_num => 1, content => "before line" },
        { line_num => 2, content => "match line with target" },
        { line_num => 3, content => "after line" }
    );
    
    my $output = capture_output(sub {
        # Test with context=0 (should only show match line)
        GF::Search::_display_match_with_context(
            \@buffer,
            "match line with target",
            2,
            "target",
            0,  # context = 0
            [],  # no after context
            undef  # no before context override
        );
    });
    
    # Remove ANSI color codes for comparison
    $output =~ s/\033\[[0-9;]*m//g;
    
    # Should only show the match line
    my @lines = split /\n/, $output;
    is(scalar(@lines), 1, "Context=0 produces single line output");
    like($lines[0], qr/^\[2\]\s+match line with target$/, "Context=0 output format correct");
}

# Test 2: Test _display_match_with_context function with context=1 (should show context)
{
    my @buffer = (
        { line_num => 1, content => "before line" },
        { line_num => 2, content => "match line with target" },
        { line_num => 3, content => "after line" }
    );
    
    my @after_context = (
        { line_num => 3, content => "after line" }
    );
    
    my $output = capture_output(sub {
        # Test with context=1 (should show before and after context)
        GF::Search::_display_match_with_context(
            \@buffer,
            "match line with target",
            2,
            "target",
            1,  # context = 1
            \@after_context,  # after context provided
            undef  # use buffer for before context
        );
    });
    
    # Remove ANSI color codes for comparison
    $output =~ s/\033\[[0-9;]*m//g;
    
    # Should show multiple lines (before + match + after + spacing)
    my @lines = split /\n/, $output;
    ok(scalar(@lines) > 1, "Context=1 produces multiple lines");
    like($output, qr/\[1\].*before line/, "Context=1 shows before-context");
}

done_testing();