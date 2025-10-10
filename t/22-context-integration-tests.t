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

# Comprehensive integration tests for context flag with other command-line options
# Tests context flag (-c) in combination with target patterns (-t), ignore patterns (-i),
# maxline limits (-m), and debug output (-d)

# Find the gf binary
my $gf_bin = File::Spec->catfile('bin', 'gf');
$gf_bin = abs_path($gf_bin);
if (!-x $gf_bin) {
    plan skip_all => "gf binary not found or not executable at $gf_bin";
}

# Create temporary test directory and files
my $test_dir = tempdir(CLEANUP => 1);
my $test_file1 = File::Spec->catfile($test_dir, 'source.txt');
my $test_file2 = File::Spec->catfile($test_dir, 'code.pl');
my $test_file3 = File::Spec->catfile($test_dir, 'data.log');
my $test_file4 = File::Spec->catfile($test_dir, 'readme.md');
my $binary_file = File::Spec->catfile($test_dir, 'binary.bin');

# Create test files with known content for context testing
open my $fh1, '>', $test_file1 or die "Cannot create $test_file1: $!";
print $fh1 "line 1: before context\n";
print $fh1 "line 2: target pattern here\n";
print $fh1 "line 3: after context\n";
print $fh1 "line 4: middle content\n";
print $fh1 "line 5: another target match\n";
print $fh1 "line 6: final context\n";
close $fh1;

open my $fh2, '>', $test_file2 or die "Cannot create $test_file2: $!";
print $fh2 "#!/usr/bin/perl\n";
print $fh2 "# Before target comment\n";
print $fh2 "use strict; # target keyword\n";
print $fh2 "# After target comment\n";
print $fh2 "my \$var = 'value';\n";
print $fh2 "print \"target output\\n\";\n";
print $fh2 "# End of file\n";
close $fh2;

open my $fh3, '>', $test_file3 or die "Cannot create $test_file3: $!";
print $fh3 "2024-01-01 09:00:00 INFO: Starting\n";
print $fh3 "2024-01-01 09:01:00 ERROR: target error occurred\n";
print $fh3 "2024-01-01 09:02:00 INFO: Recovery attempt\n";
print $fh3 "2024-01-01 09:03:00 WARN: target warning\n";
print $fh3 "2024-01-01 09:04:00 INFO: Process complete\n";
close $fh3;

open my $fh4, '>', $test_file4 or die "Cannot create $test_file4: $!";
print $fh4 "# Project Documentation\n";
print $fh4 "\n";
print $fh4 "This is the target section.\n";
print $fh4 "\n";
print $fh4 "## Features\n";
print $fh4 "- Feature 1\n";
print $fh4 "- target feature\n";
print $fh4 "- Feature 3\n";
close $fh4;

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

plan tests => 45;

# Test 1: Context with target patterns (-t)
{
    chdir $test_dir or die "Cannot chdir to $test_dir: $!";
    my $result = run_gf_command('--search', 'target', '--context', '1', '--target', '\.pl$');
    ok($result->{exit_code} == 0, "Context with target pattern exits with code 0");
    like($result->{output}, qr/code\.pl/, "Context with target finds .pl files");
    unlike($result->{output}, qr/source\.txt/, "Context with target excludes .txt files");
    unlike($result->{output}, qr/data\.log/, "Context with target excludes .log files");
    
    # Check context is displayed for .pl file matches
    like($result->{output}, qr/\[2\].*Before target comment/, "Context shows before line for .pl match");
    like($result->{output}, qr/\[3\].*target keyword/, "Context shows matching line for .pl match");
    like($result->{output}, qr/\[4\].*After target comment/, "Context shows after line for .pl match");
}

# Test 2: Context with ignore patterns (-i)
{
    chdir $test_dir or die "Cannot chdir to $test_dir: $!";
    my $result = run_gf_command('--search', 'target', '--context', '1', '--ignore', '\.log$');
    ok($result->{exit_code} == 0, "Context with ignore pattern exits with code 0");
    like($result->{output}, qr/source\.txt/, "Context with ignore includes .txt files");
    like($result->{output}, qr/code\.pl/, "Context with ignore includes .pl files");
    unlike($result->{output}, qr/data\.log/, "Context with ignore excludes .log files");
    
    # Check context is displayed for included files
    like($result->{output}, qr/\[1\].*before context/, "Context shows before line for .txt match");
    like($result->{output}, qr/\[2\].*target pattern here/, "Context shows matching line for .txt match");
    like($result->{output}, qr/\[3\].*after context/, "Context shows after line for .txt match");
}

# Test 3: Context with maxline limits (-m)
{
    chdir $test_dir or die "Cannot chdir to $test_dir: $!";
    my $result = run_gf_command('--search', 'target', '--context', '2', '--maxline', '4');
    ok($result->{exit_code} == 0, "Context with maxline exits with code 0");
    
    # Should find first target match but not later ones due to maxline limit
    like($result->{output}, qr/\[2\].*target pattern here/, "Context with maxline finds early matches");
    # Should not find matches beyond maxline limit
    unlike($result->{output}, qr/\[5\].*another target match/, "Context with maxline respects line limit");
    
    # Check that context respects maxline limit
    like($result->{output}, qr/\[1\].*before context/, "Context with maxline shows available before context");
    like($result->{output}, qr/\[3\].*after context/, "Context with maxline shows available after context");
}

# Test 4: Context with debug output (-d)
{
    chdir $test_dir or die "Cannot chdir to $test_dir: $!";
    my $result = run_gf_command('--search', 'target', '--context', '1', '--debug');
    ok($result->{exit_code} == 0, "Context with debug exits with code 0");
    
    # Debug output should contain debug information
    like($result->{output}, qr/retTerm:/, "Context with debug shows debug information");
    
    # Should still show context properly
    like($result->{output}, qr/\[1\].*before context/, "Context with debug shows before context");
    like($result->{output}, qr/\[2\].*target pattern here/, "Context with debug shows matching line");
    like($result->{output}, qr/\[3\].*after context/, "Context with debug shows after context");
}

# Test 5: Context with multiple target patterns
{
    chdir $test_dir or die "Cannot chdir to $test_dir: $!";
    my $result = run_gf_command('--search', 'target', '--context', '1', '--target', '\.pl$,\.md$');
    ok($result->{exit_code} == 0, "Context with multiple targets exits with code 0");
    like($result->{output}, qr/code\.pl/, "Context with multiple targets includes .pl files");
    like($result->{output}, qr/readme\.md/, "Context with multiple targets includes .md files");
    unlike($result->{output}, qr/source\.txt/, "Context with multiple targets excludes .txt files");
    
    # Check context for both file types
    like($result->{output}, qr/\[2\].*Before target comment/, "Context shows before line for .pl match");
    like($result->{output}, qr/\[3\].*target section/, "Context shows matching line for .md match");
}

# Test 6: Context with multiple ignore patterns
{
    chdir $test_dir or die "Cannot chdir to $test_dir: $!";
    my $result = run_gf_command('--search', 'target', '--context', '1', '--ignore', '\.log$,\.bin$');
    ok($result->{exit_code} == 0, "Context with multiple ignores exits with code 0");
    like($result->{output}, qr/source\.txt/, "Context with multiple ignores includes .txt files");
    like($result->{output}, qr/code\.pl/, "Context with multiple ignores includes .pl files");
    unlike($result->{output}, qr/data\.log/, "Context with multiple ignores excludes .log files");
}

# Test 7: Context with target and ignore patterns combined
{
    chdir $test_dir or die "Cannot chdir to $test_dir: $!";
    my $result = run_gf_command('--search', 'target', '--context', '1', '--target', '\.txt$,\.pl$', '--ignore', '\.pl$');
    ok($result->{exit_code} == 0, "Context with target and ignore exits with code 0");
    like($result->{output}, qr/source\.txt/, "Context with target and ignore includes .txt files");
    unlike($result->{output}, qr/code\.pl/, "Context with target and ignore excludes .pl files (ignore overrides target)");
}

# Test 8: Context with maxline and target patterns
{
    chdir $test_dir or die "Cannot chdir to $test_dir: $!";
    my $result = run_gf_command('--search', 'target', '--context', '1', '--maxline', '3', '--target', '\.pl$');
    ok($result->{exit_code} == 0, "Context with maxline and target exits with code 0");
    like($result->{output}, qr/code\.pl/, "Context with maxline and target finds .pl files");
    like($result->{output}, qr/\[3\].*target keyword/, "Context with maxline and target finds match within limit");
}

# Test 9: Context with debug and target patterns
{
    chdir $test_dir or die "Cannot chdir to $test_dir: $!";
    my $result = run_gf_command('--search', 'target', '--context', '1', '--debug', '--target', '\.txt$');
    ok($result->{exit_code} == 0, "Context with debug and target exits with code 0");
    like($result->{output}, qr/retTerm:/, "Context with debug and target shows debug info");
    like($result->{output}, qr/source\.txt/, "Context with debug and target finds .txt files");
    unlike($result->{output}, qr/code\.pl/, "Context with debug and target excludes .pl files");
}

# Test 10: Context with all options combined
{
    chdir $test_dir or die "Cannot chdir to $test_dir: $!";
    my $result = run_gf_command('--search', 'target', '--context', '1', '--target', '\.txt$,\.pl$', '--ignore', '\.pl$', '--maxline', '5', '--debug');
    ok($result->{exit_code} == 0, "Context with all options exits with code 0");
    like($result->{output}, qr/retTerm:/, "Context with all options shows debug info");
    like($result->{output}, qr/source\.txt/, "Context with all options finds .txt files");
    unlike($result->{output}, qr/code\.pl/, "Context with all options respects ignore over target");
    like($result->{output}, qr/\[1\].*before context/, "Context with all options shows before context");
    like($result->{output}, qr/\[2\].*target pattern here/, "Context with all options shows matching line");
    like($result->{output}, qr/\[3\].*after context/, "Context with all options shows after context");
}

# Test 11: Context with higher values and target patterns
{
    chdir $test_dir or die "Cannot chdir to $test_dir: $!";
    my $result = run_gf_command('--search', 'target', '--context', '3', '--target', '\.txt$');
    ok($result->{exit_code} == 0, "Context 3 with target exits with code 0");
    like($result->{output}, qr/source\.txt/, "Context 3 with target finds .txt files");
    
    # Check that higher context values work with target patterns
    like($result->{output}, qr/\[1\].*before context/, "Context 3 shows line 1");
    like($result->{output}, qr/\[2\].*target pattern here/, "Context 3 shows matching line 2");
    like($result->{output}, qr/\[3\].*after context/, "Context 3 shows line 3");
    like($result->{output}, qr/\[4\].*middle content/, "Context 3 shows line 4");
    like($result->{output}, qr/\[5\].*another target match/, "Context 3 shows line 5");
}

# Test 12: Context zero with other options (should behave like no context)
{
    chdir $test_dir or die "Cannot chdir to $test_dir: $!";
    my $result = run_gf_command('--search', 'target', '--context', '0', '--target', '\.txt$');
    ok($result->{exit_code} == 0, "Context 0 with target exits with code 0");
    like($result->{output}, qr/\[2\].*target pattern here/, "Context 0 shows matching line");
    unlike($result->{output}, qr/\[1\].*before context/, "Context 0 does not show before context");
    unlike($result->{output}, qr/\[3\].*after context/, "Context 0 does not show after context");
}

# Restore original directory
chdir $original_dir or die "Cannot restore directory: $!";

done_testing();