#!/usr/bin/env perl
use strict;
use warnings;
use Test::More tests => 8;
use lib 'lib';
use lib 't';

require 'test_utils.pl';

BEGIN {
    use_ok('GF::Help', qw(show_usage show_help show_version show_usage_and_exit));
}

# Test show_usage function
subtest 'show_usage function' => sub {
    plan tests => 5;
    
    my $output = capture_output(sub { GF::Help::show_usage() });
    
    like($output, qr/gf - Friendly file search tool/, 'Contains tool description');
    like($output, qr/Usage: gf -s PATTERN/, 'Contains usage line');
    like($output, qr/Common examples:/, 'Contains examples section');
    like($output, qr/gf -s 'function'/, 'Contains basic example');
    like($output, qr/Try 'gf --help'/, 'Contains help pointer');
};

# Test show_help function completeness
subtest 'show_help function completeness' => sub {
    plan tests => 12;
    
    my $output = capture_output(sub { GF::Help::show_help() });
    
    # Check main sections
    like($output, qr/DESCRIPTION/, 'Contains DESCRIPTION section');
    like($output, qr/USAGE/, 'Contains USAGE section');
    like($output, qr/SEARCH OPTIONS/, 'Contains SEARCH OPTIONS section');
    like($output, qr/FILE FILTERING OPTIONS/, 'Contains FILE FILTERING OPTIONS section');
    like($output, qr/OUTPUT CONTROL OPTIONS/, 'Contains OUTPUT CONTROL OPTIONS section');
    like($output, qr/UTILITY OPTIONS/, 'Contains UTILITY OPTIONS section');
    like($output, qr/EXAMPLES/, 'Contains EXAMPLES section');
    like($output, qr/CONFIGURATION FILES/, 'Contains CONFIGURATION FILES section');
    like($output, qr/EXIT STATUS/, 'Contains EXIT STATUS section');
    like($output, qr/SEE ALSO/, 'Contains SEE ALSO section');
    
    # Check specific options are documented
    like($output, qr/-s, --search PATTERN/, 'Documents search option');
    like($output, qr/-t, --target PATTERN/, 'Documents target option');
};

# Test show_help examples section
subtest 'show_help examples section' => sub {
    plan tests => 6;
    
    my $output = capture_output(sub { GF::Help::show_help() });
    
    like($output, qr/Basic searches:/, 'Contains basic searches section');
    like($output, qr/File type filtering:/, 'Contains file filtering section');
    like($output, qr/Context and limits:/, 'Contains context section');
    like($output, qr/Combined options:/, 'Contains combined options section');
    like($output, qr/gf -s 'function'/, 'Contains function search example');
    like($output, qr/gf -s 'TODO\|FIXME'/, 'Contains regex example');
};

# Test show_help configuration section
subtest 'show_help configuration section' => sub {
    plan tests => 5;
    
    my $output = capture_output(sub { GF::Help::show_help() });
    
    like($output, qr/~\/\.gfconf/, 'Documents user config location');
    like($output, qr/\/etc\/gfconf/, 'Documents system config location');
    like($output, qr/ignore \*\.log/, 'Shows ignore syntax');
    like($output, qr/target \*\.pl/, 'Shows target syntax');
    like($output, qr/source \/path\/to\/other\/config/, 'Shows source syntax');
};

# Test show_version function format
subtest 'show_version function format' => sub {
    plan tests => 4;
    
    my $output = capture_output(sub { GF::Help::show_version() });
    
    like($output, qr/gf \(Friendly File Search\) version/, 'Contains version header');
    like($output, qr/Released \d{4}/, 'Contains release year');
    like($output, qr/Written by/, 'Contains author information');
    like($output, qr/For help and examples, run: gf --help/, 'Contains help pointer');
};

# Test version constants
subtest 'version constants' => sub {
    plan tests => 3;
    
    ok(defined $GF::Help::VERSION, 'VERSION constant is defined');
    ok(defined $GF::Help::VERSION_DATE, 'VERSION_DATE constant is defined');
    ok(defined $GF::Help::AUTHOR, 'AUTHOR constant is defined');
};

# Test show_usage_and_exit function output (we can't test exit without terminating)
subtest 'show_usage_and_exit function output' => sub {
    plan tests => 1;
    
    # We can test that the function exists and is callable
    # The actual exit behavior is tested in integration tests
    can_ok('GF::Help', 'show_usage_and_exit');
};

done_testing();