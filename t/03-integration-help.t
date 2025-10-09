#!/usr/bin/env perl
use strict;
use warnings;
use Test::More tests => 6;
use lib 'lib';
use lib 't';
use File::Temp qw(tempdir);
use Cwd qw(getcwd);

require 'test_utils.pl';

# Test help flags work correctly with main script
subtest 'help flag integration' => sub {
    plan tests => 4;
    
    # Test --help flag
    my $help_output = `perl bin/gf --help 2>&1`;
    my $help_exit_code = $? >> 8;
    
    is($help_exit_code, 0, '--help flag exits with code 0');
    like($help_output, qr/gf - Friendly file search tool/, '--help shows tool description');
    like($help_output, qr/EXAMPLES/, '--help includes examples section');
    like($help_output, qr/CONFIGURATION FILES/, '--help includes configuration section');
};

# Test version flag integration
subtest 'version flag integration' => sub {
    plan tests => 3;
    
    # Test --version flag
    my $version_output = `perl bin/gf --version 2>&1`;
    my $version_exit_code = $? >> 8;
    
    is($version_exit_code, 0, '--version flag exits with code 0');
    like($version_output, qr/gf \(Friendly File Search\) version/, '--version shows version info');
    like($version_output, qr/Written by/, '--version shows author info');
};

# Test short help flag
subtest 'short help flag integration' => sub {
    plan tests => 2;
    
    # Test -h flag
    my $help_output = `perl bin/gf -h 2>&1`;
    my $help_exit_code = $? >> 8;
    
    is($help_exit_code, 0, '-h flag exits with code 0');
    like($help_output, qr/gf - Friendly file search tool/, '-h shows help content');
};

# Test error scenarios trigger appropriate help
subtest 'error scenarios show help' => sub {
    plan tests => 4;
    
    # Test no arguments
    my $no_args_output = `perl bin/gf 2>&1`;
    my $no_args_exit_code = $? >> 8;
    
    is($no_args_exit_code, 1, 'No arguments exits with error code 1');
    like($no_args_output, qr/Error: No arguments provided/, 'No arguments shows specific error');
    like($no_args_output, qr/Usage: gf -s PATTERN/, 'No arguments shows usage');
    like($no_args_output, qr/Try 'gf --help'/, 'No arguments suggests help');
};

# Test invalid arguments trigger help
subtest 'invalid arguments show help' => sub {
    plan tests => 2;
    
    # Test invalid option
    my $invalid_output = `perl bin/gf --invalid-option 2>&1`;
    my $invalid_exit_code = $? >> 8;
    
    is($invalid_exit_code, 1, 'Invalid option exits with error code 1');
    like($invalid_output, qr/Usage: gf -s PATTERN/, 'Invalid option shows usage');
};

# Test examples in documentation work correctly
subtest 'documentation examples work' => sub {
    plan tests => 4;
    
    # Create a temporary test directory with files
    my $temp_dir = create_test_directory();
    my $original_dir = getcwd();
    
    # Change to test directory
    chdir $temp_dir or die "Cannot chdir to $temp_dir: $!";
    
    # Test basic search example (should work without errors)
    # Use -I to include the lib directory from original location
    my $search_output = `perl -I$original_dir/lib $original_dir/bin/gf -s 'function' 2>&1`;
    my $search_exit_code = $? >> 8;
    
    is($search_exit_code, 0, 'Basic search example runs without error');
    like($search_output, qr/function/, 'Basic search finds expected content');
    
    # Test file type filtering example
    my $filter_output = `perl -I$original_dir/lib $original_dir/bin/gf -s 'function' -t '*.pl' 2>&1`;
    my $filter_exit_code = $? >> 8;
    
    is($filter_exit_code, 0, 'File filtering example runs without error');
    like($filter_output, qr/main\.pl/, 'File filtering targets correct files');
    
    # Restore original directory
    chdir $original_dir or die "Cannot chdir back to $original_dir: $!";
};

done_testing();