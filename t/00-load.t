#!/usr/bin/env perl
use strict;
use warnings;
use lib 'lib';
use Test::More tests => 4;

# Test that all modules can be loaded
BEGIN {
    use_ok('GF::Help') || print "Bail out!\n";
    use_ok('GF::Config') || print "Bail out!\n";
    use_ok('GF::Search') || print "Bail out!\n";
}

# Test that we can import the expected functions
use_ok('GF::Help', qw(show_usage show_help show_version show_usage_and_exit));

diag("Testing gf modules");