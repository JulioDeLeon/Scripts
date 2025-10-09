#!/usr/bin/env perl
# Test utilities and fixtures for gf testing

use strict;
use warnings;
use File::Temp qw(tempdir tempfile);
use File::Path qw(make_path remove_tree);

=head1 NAME

test_utils.pl - Test utilities and fixtures for gf testing

=head1 DESCRIPTION

This module provides common testing utilities, fixtures, and helper functions
for testing the gf file search tool.

=cut

# Create a temporary directory structure for testing
sub create_test_directory {
    my $temp_dir = tempdir(CLEANUP => 1);
    
    # Create subdirectories
    make_path("$temp_dir/src");
    make_path("$temp_dir/docs");
    make_path("$temp_dir/logs");
    make_path("$temp_dir/temp");
    
    # Create test files with content
    write_test_file("$temp_dir/src/main.pl", <<'EOF');
#!/usr/bin/env perl
use strict;
use warnings;

# TODO: Add error handling
sub main {
    print "Hello, world!\n";
    # FIXME: This function needs improvement
    return 0;
}

main();
EOF

    write_test_file("$temp_dir/src/utils.pm", <<'EOF');
package Utils;
use strict;
use warnings;

sub function_one {
    my $param = shift;
    return $param * 2;
}

sub function_two {
    # TODO: Implement this function
    return undef;
}

1;
EOF

    write_test_file("$temp_dir/docs/README.md", <<'EOF');
# Project Documentation

This is a test project for the gf search tool.

## Functions

- function_one: Doubles input
- function_two: Not implemented yet

## TODO

- Add more documentation
- Write tests
EOF

    write_test_file("$temp_dir/logs/error.log", <<'EOF');
2024-01-01 10:00:00 ERROR: Connection failed
2024-01-01 10:01:00 INFO: Retrying connection
2024-01-01 10:02:00 ERROR: Authentication failed
EOF

    write_test_file("$temp_dir/temp/cache.tmp", <<'EOF');
temporary cache data
function cache_get() { return null; }
EOF

    return $temp_dir;
}

# Write content to a test file
sub write_test_file {
    my ($filename, $content) = @_;
    open my $fh, '>', $filename or die "Cannot write to $filename: $!";
    print $fh $content;
    close $fh;
}

# Create a temporary config file for testing
sub create_test_config {
    my ($temp_dir, $config_content) = @_;
    my $config_file = "$temp_dir/.gfconf";
    write_test_file($config_file, $config_content);
    return $config_file;
}

# Capture output from a function
sub capture_output {
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

# Capture STDERR output from a function
sub capture_stderr {
    my ($code_ref) = @_;
    my $output = '';
    
    # Redirect STDERR to capture output
    open my $old_stderr, '>&', \*STDERR or die "Can't dup STDERR: $!";
    close STDERR;
    open STDERR, '>', \$output or die "Can't redirect STDERR: $!";
    
    # Execute the code
    eval { $code_ref->(); };
    my $error = $@;
    
    # Restore STDERR
    close STDERR;
    open STDERR, '>&', $old_stderr or die "Can't restore STDERR: $!";
    
    die $error if $error;
    return $output;
}

# Strip ANSI color codes from text
sub strip_ansi_codes {
    my ($text) = @_;
    # Remove ANSI escape sequences
    $text =~ s/\e\[[0-9;]*m//g;
    return $text;
}

# Capture output and strip ANSI codes
sub capture_output_clean {
    my ($code_ref) = @_;
    my $output = capture_output($code_ref);
    return strip_ansi_codes($output);
}

1;