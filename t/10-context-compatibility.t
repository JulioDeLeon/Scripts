#!/usr/bin/env perl

use strict;
use warnings;
use Test::More;
use File::Temp qw(tempdir tempfile);
use File::Spec;
use Cwd qw(abs_path);
use IPC::Open3;
use Symbol qw(gensym);

# Test context display functionality compatibility
# Ensures context options provide same display as current implementation
# and maintain identical context line formatting and numbering

# Find the gf binary
my $gf_bin = File::Spec->catfile('bin', 'gf');
$gf_bin = abs_path($gf_bin);
if (!-x $gf_bin) {
    plan skip_all => "gf binary not found or not executable at $gf_bin";
}

# Store original directory for running gf
our $original_dir = Cwd::getcwd();

# Create temporary test directory and files
my $test_dir = tempdir(CLEANUP => 1);
my $test_file = File::Spec->catfile($test_dir, 'context_test.txt');

# Create test file with specific content for context testing
open my $fh, '>', $test_file or die "Cannot create $test_file: $!";
print $fh "Line 1: before context\n";
print $fh "Line 2: before match\n";
print $fh "Line 3: MATCH_PATTERN here\n";
print $fh "Line 4: after match\n";
print $fh "Line 5: after context\n";
print $fh "Line 6: another line\n";
print $fh "Line 7: second MATCH_PATTERN\n";
print $fh "Line 8: final line\n";
close $fh;

# Helper function to run gf command and capture output
sub run_gf_command {
    my (@args) = @_;
    
    # Change to original directory to run gf (so it can find lib/)
    my $current_dir = Cwd::getcwd();
    chdir $original_dir or die "Cannot chdir to $original_dir: $!";
    
    my $cmd = [$gf_bin, @args];
    my ($stdin, $stdout, $stderr);
    $stderr = gensym();
    
    my $pid = open3($stdin, $stdout, $stderr, @$cmd);
    
    my $output = '';
    my $error = '';
    
    if ($stdout) {
        while (my $line = <$stdout>) {
            $output .= $line;
        }
        close $stdout;
    }
    
    if ($stderr) {
        while (my $line = <$stderr>) {
            $error .= $line;
        }
        close $stderr;
    }
    
    waitpid($pid, 0);
    my $exit_code = $? >> 8;
    
    # Change back to test directory
    chdir $current_dir or die "Cannot restore directory: $!";
    
    return {
        output => $output,
        error => $error,
        exit_code => $exit_code
    };
}

plan tests => 12;

# Test 1: No context (default behavior)
{
    chdir $test_dir or die "Cannot chdir to $test_dir: $!";
    my $result = run_gf_command('--search', 'MATCH_PATTERN');
    
    ok($result->{exit_code} == 0, "No context search exits with code 0");
    like($result->{output}, qr/\[3\].*MATCH_PATTERN here/, "Shows first match with line number");
    like($result->{output}, qr/\[7\].*second MATCH_PATTERN/, "Shows second match with line number");
    unlike($result->{output}, qr/\[2\].*before match/, "No context lines shown without context option");
}

# Test 2: Context = 1 (one line before and after)
{
    chdir $test_dir or die "Cannot chdir to $test_dir: $!";
    my $result = run_gf_command('--search', 'MATCH_PATTERN', '--context', '1');
    
    ok($result->{exit_code} == 0, "Context=1 search exits with code 0");
    like($result->{output}, qr/\[2\].*before match/, "Shows 1 line before first match");
    like($result->{output}, qr/\[3\].*MATCH_PATTERN here/, "Shows first match line");
    like($result->{output}, qr/\[4\].*after match/, "Shows 1 line after first match");
    like($result->{output}, qr/\[6\].*another line/, "Shows 1 line before second match");
    like($result->{output}, qr/\[7\].*second MATCH_PATTERN/, "Shows second match line");
    like($result->{output}, qr/\[8\].*final line/, "Shows 1 line after second match");
}

# Test 3: Context = 2 (two lines before and after)
{
    chdir $test_dir or die "Cannot chdir to $test_dir: $!";
    my $result = run_gf_command('--search', 'MATCH_PATTERN', '--context', '2');
    
    ok($result->{exit_code} == 0, "Context=2 search exits with code 0");
    like($result->{output}, qr/\[1\].*before context/, "Shows 2 lines before first match");
}

# Restore original directory
chdir $original_dir or die "Cannot restore directory: $!";

done_testing();