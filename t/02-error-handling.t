#!/usr/bin/env perl
use strict;
use warnings;
use Test::More tests => 8;
use lib 'lib';
use lib 't';

require 'test_utils.pl';

BEGIN {
    use_ok('GF::Search', qw(process_args));
    use_ok('GF::Config', qw(lookup_config_file));
    use_ok('GF::Help', qw(show_usage_and_exit));
}

# Test that process_args function exists and handles arguments
subtest 'process_args function exists' => sub {
    plan tests => 1;
    
    # Test that the function exists and is callable
    can_ok('GF::Search', 'process_args');
};

# Test that error handling functions exist
subtest 'error handling functions exist' => sub {
    plan tests => 2;
    
    # Test that error handling functions are available
    can_ok('GF::Help', 'show_usage_and_exit');
    can_ok('GF::Config', 'lookup_config_file');
};

# Test configuration file error handling
subtest 'configuration file errors' => sub {
    plan tests => 4;
    
    # Create a temporary directory for testing
    my $temp_dir = create_test_directory();
    
    # Test with non-existent source file
    my $config_content = <<'EOF';
ignore *.log
source /non/existent/file.conf
target *.pl
EOF
    
    my $config_file = create_test_config($temp_dir, $config_content);
    
    # Temporarily set HOME to our test directory
    local $ENV{HOME} = $temp_dir;
    
    my %ignores = ();
    my %targets = ();
    
    my $stderr_output = '';
    {
        local *STDERR;
        open STDERR, '>', \$stderr_output or die "Can't redirect STDERR: $!";
        
        GF::Config::lookup_config_file(\%ignores, \%targets, 0);
    }
    
    like($stderr_output, qr/Warning: Source file.*does not exist/, 'Warns about non-existent source file');
    like($stderr_output, qr/Skipping this source directive/, 'Indicates skipping the directive');
    
    # Verify that valid directives still work
    ok(exists $ignores{'*.log'}, 'Valid ignore directive still processed');
    ok(exists $targets{'*.pl'}, 'Valid target directive still processed');
};

# Test configuration file with invalid directives
subtest 'invalid configuration directives' => sub {
    plan tests => 3;
    
    # Create a temporary directory for testing
    my $temp_dir = create_test_directory();
    
    # Use the invalid config fixture
    my $config_content = <<'EOF';
ignore *.log
invalid_directive some_value
target *.pl
unknown_command test
EOF
    
    my $config_file = create_test_config($temp_dir, $config_content);
    
    # Temporarily set HOME to our test directory
    local $ENV{HOME} = $temp_dir;
    
    my %ignores = ();
    my %targets = ();
    
    my $stderr_output = '';
    {
        local *STDERR;
        open STDERR, '>', \$stderr_output or die "Can't redirect STDERR: $!";
        
        GF::Config::lookup_config_file(\%ignores, \%targets, 0);
    }
    
    like($stderr_output, qr/Warning: Unknown configuration directive/, 'Warns about unknown directive');
    like($stderr_output, qr/Valid directives are: source, target, ignore/, 'Provides valid directive list');
    
    # Verify that valid directives still work despite invalid ones
    ok(exists $ignores{'*.log'} && exists $targets{'*.pl'}, 'Valid directives processed despite invalid ones');
};

# Test file permission error handling
subtest 'file permission errors' => sub {
    plan tests => 2;
    
    # Create a temporary directory for testing
    my $temp_dir = create_test_directory();
    
    # Create a config file and make it unreadable
    my $config_file = create_test_config($temp_dir, "ignore *.log\n");
    chmod 0000, $config_file;  # Remove all permissions
    
    # Temporarily set HOME to our test directory
    local $ENV{HOME} = $temp_dir;
    
    my %ignores = ();
    my %targets = ();
    
    my $stderr_output = '';
    {
        local *STDERR;
        open STDERR, '>', \$stderr_output or die "Can't redirect STDERR: $!";
        
        GF::Config::lookup_config_file(\%ignores, \%targets, 0);
    }
    
    like($stderr_output, qr/Warning: Configuration file.*is not readable/, 'Warns about unreadable config file');
    like($stderr_output, qr/Check file permissions/, 'Provides helpful suggestion');
    
    # Restore permissions for cleanup
    chmod 0644, $config_file;
};

done_testing();