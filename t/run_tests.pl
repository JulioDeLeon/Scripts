#!/usr/bin/env perl
use strict;
use warnings;

# Simple test runner for gf test suite
print "Running gf test suite...\n\n";

my @test_files = qw(
    t/00-load.t
    t/01-help-system.t
    t/02-error-handling.t
    t/03-integration-help.t
);

my $total_tests = 0;
my $failed_tests = 0;

foreach my $test_file (@test_files) {
    print "Running $test_file...\n";
    my $result = system("perl $test_file");
    my $exit_code = $result >> 8;
    
    if ($exit_code == 0) {
        print "✓ PASSED\n";
    } else {
        print "✗ FAILED (exit code: $exit_code)\n";
        $failed_tests++;
    }
    print "\n";
    $total_tests++;
}

print "=" x 50 . "\n";
print "Test Summary:\n";
print "Total tests: $total_tests\n";
print "Passed: " . ($total_tests - $failed_tests) . "\n";
print "Failed: $failed_tests\n";

if ($failed_tests == 0) {
    print "\n🎉 All tests passed!\n";
    exit 0;
} else {
    print "\n❌ Some tests failed.\n";
    exit 1;
}