#!/usr/bin/env perl

use strict;
use warnings;
use Test::More;
use File::Temp qw(tempdir tempfile);
use File::Spec;
use Cwd qw(abs_path);
use IPC::Open3;
use Symbol qw(gensym);

# Store original directory for running gf
our $original_dir = Cwd::getcwd();

# Test command-line interface compatibility
# Ensures all existing command-line options maintain identical behavior
# and output format remains exactly the same as current implementation

# Find the gf binary
my $gf_bin = File::Spec->catfile('bin', 'gf');
$gf_bin = abs_path($gf_bin);
if (!-x $gf_bin) {
    plan skip_all => "gf binary not found or not executable at $gf_bin";
}

# Create temporary test directory and files
my $test_dir = tempdir(CLEANUP => 1);
my $test_file1 = File::Spec->catfile($test_dir, 'test1.txt');
my $test_file2 = File::Spec->catfile($test_dir, 'test2.pl');
my $test_file3 = File::Spec->catfile($test_dir, 'ignore_me.log');
my $binary_file = File::Spec->catfile($test_dir, 'binary.bin');

# Create test files with known content
open my $fh1, '>', $test_file1 or die "Cannot create $test_file1: $!";
print $fh1 "line 1: hello world\n";
print $fh1 "line 2: test pattern here\n";
print $fh1 "line 3: another line\n";
print $fh1 "line 4: pattern match again\n";
print $fh1 "line 5: final line\n";
close $fh1;

open my $fh2, '>', $test_file2 or die "Cannot create $test_file2: $!";
print $fh2 "#!/usr/bin/perl\n";
print $fh2 "# This is a test pattern\n";
print $fh2 "use strict;\n";
print $fh2 "my \$var = 'pattern value';\n";
print $fh2 "print \"Hello World\\n\";\n";
close $fh2;

open my $fh3, '>', $test_file3 or die "Cannot create $test_file3: $!";
print $fh3 "log entry 1: pattern found\n";
print $fh3 "log entry 2: normal entry\n";
close $fh3;

# Create a binary file
open my $fh_bin, '>', $binary_file or die "Cannot create $binary_file: $!";
binmode $fh_bin;
print $fh_bin pack("C*", 0x00, 0x01, 0x02, 0x03, 0xFF, 0xFE, 0xFD);
close $fh_bin;

# Helper function to run gf command and capture output
sub run_gf_command {
    my (@args) = @_;
    
    # Change to original directory to run gf (so it can find lib/)
    my $current_dir = Cwd::getcwd();
    chdir $original_dir or die "Cannot chdir to $original_dir: $!";
    
    my $cmd = [$gf_bin, @args];
    my ($stdin, $stdout, $stderr);
    $stderr = gensym;
    
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

# We'll change to test directory for individual tests

plan tests => 40;

# Test 1: Help option (-h)
{
    my $result = run_gf_command('-h');
    ok($result->{exit_code} == 0, "Help option exits with code 0");
    like($result->{output}, qr/Usage:/, "Help output contains usage information");
    like($result->{output}, qr/--search/, "Help output contains --search option");
    like($result->{output}, qr/--target/, "Help output contains --target option");
    like($result->{output}, qr/--ignore/, "Help output contains --ignore option");
}

# Test 2: Version option (-V)
{
    my $result = run_gf_command('-V');
    ok($result->{exit_code} == 0, "Version option exits with code 0");
    like($result->{output}, qr/gf version/, "Version output contains version information");
}

# Test 3: Basic search functionality
{
    # Change to test directory for this test
    chdir $test_dir or die "Cannot chdir to $test_dir: $!";
    my $result = run_gf_command('--search', 'pattern');
    ok($result->{exit_code} == 0, "Basic search exits with code 0");
    like($result->{output}, qr/test1\.txt/, "Search finds matches in test1.txt");
    like($result->{output}, qr/test2\.pl/, "Search finds matches in test2.pl");
    like($result->{output}, qr/ignore_me\.log/, "Search finds matches in log file");
    like($result->{output}, qr/\[2\].*test pattern here/, "Search shows correct line numbers and content");
}

# Test 4: Target pattern filtering
{
    chdir $test_dir or die "Cannot chdir to $test_dir: $!";
    my $result = run_gf_command('--search', 'pattern', '--target', '\.pl$');
    ok($result->{exit_code} == 0, "Target filtering exits with code 0");
    like($result->{output}, qr/test2\.pl/, "Target filtering includes .pl files");
    unlike($result->{output}, qr/test1\.txt/, "Target filtering excludes .txt files");
    unlike($result->{output}, qr/ignore_me\.log/, "Target filtering excludes .log files");
}

# Test 5: Ignore pattern filtering
{
    chdir $test_dir or die "Cannot chdir to $test_dir: $!";
    my $result = run_gf_command('--search', 'pattern', '--ignore', '\.log$');
    ok($result->{exit_code} == 0, "Ignore filtering exits with code 0");
    like($result->{output}, qr/test1\.txt/, "Ignore filtering includes .txt files");
    like($result->{output}, qr/test2\.pl/, "Ignore filtering includes .pl files");
    unlike($result->{output}, qr/ignore_me\.log/, "Ignore filtering excludes .log files");
}

# Test 6: Context option
{
    chdir $test_dir or die "Cannot chdir to $test_dir: $!";
    my $result = run_gf_command('--search', 'test pattern here', '--context', '1');
    ok($result->{exit_code} == 0, "Context option exits with code 0");
    like($result->{output}, qr/\[1\].*hello world/, "Context shows line before match");
    like($result->{output}, qr/\[2\].*test pattern here/, "Context shows matching line");
    like($result->{output}, qr/\[3\].*another line/, "Context shows line after match");
}

# Test 7: Maxline option
{
    chdir $test_dir or die "Cannot chdir to $test_dir: $!";
    my $result = run_gf_command('--search', 'pattern', '--maxline', '3');
    ok($result->{exit_code} == 0, "Maxline option exits with code 0");
    # Should find pattern in line 2 but not in line 4 due to maxline limit
    like($result->{output}, qr/\[2\].*test pattern here/, "Maxline finds early matches");
    unlike($result->{output}, qr/\[4\].*pattern match again/, "Maxline stops before later matches");
}

# Test 8: Debug option
{
    chdir $test_dir or die "Cannot chdir to $test_dir: $!";
    my $result = run_gf_command('--search', 'pattern', '--debug');
    ok($result->{exit_code} == 0, "Debug option exits with code 0");
    # Debug output should contain magenta-colored debug information
    # We can't easily test for ANSI colors, but we can check for debug content
    like($result->{output}, qr/retTerm:/, "Debug output contains argument information");
}

# Test 9: Error handling - no search pattern
{
    my $result = run_gf_command();
    ok($result->{exit_code} == 1, "No arguments exits with error code 1");
    like($result->{error}, qr/Error: No arguments provided/, "Error message for no arguments");
}

# Test 10: Error handling - no search pattern with other options
{
    my $result = run_gf_command('--target', '\.txt$');
    ok($result->{exit_code} == 1, "Missing search pattern exits with error code 1");
    like($result->{error}, qr/Error: No search pattern provided/, "Error message for missing search pattern");
}

# Test 11: Multiple ignore patterns (comma-separated)
{
    chdir $test_dir or die "Cannot chdir to $test_dir: $!";
    my $result = run_gf_command('--search', 'pattern', '--ignore', '\.log$,\.bin$');
    ok($result->{exit_code} == 0, "Multiple ignore patterns exit with code 0");
    like($result->{output}, qr/test1\.txt/, "Multiple ignores include .txt files");
    unlike($result->{output}, qr/ignore_me\.log/, "Multiple ignores exclude .log files");
}

# Test 12: Multiple target patterns (comma-separated)
{
    chdir $test_dir or die "Cannot chdir to $test_dir: $!";
    my $result = run_gf_command('--search', 'pattern', '--target', '\.txt$,\.pl$');
    ok($result->{exit_code} == 0, "Multiple target patterns exit with code 0");
    like($result->{output}, qr/test1\.txt/, "Multiple targets include .txt files");
    like($result->{output}, qr/test2\.pl/, "Multiple targets include .pl files");
    unlike($result->{output}, qr/ignore_me\.log/, "Multiple targets exclude .log files");
}

# Restore original directory
chdir $original_dir or die "Cannot restore directory: $!";

done_testing();