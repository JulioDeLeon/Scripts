#!/usr/bin/env perl

use strict;
use warnings;
use Test::More;
use File::Temp qw(tempdir tempfile);
use File::Spec;
use Cwd qw(abs_path);

# Test configuration file compatibility
# Ensures configuration file format and locations remain unchanged
# and all configuration directives work as before

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
my $config_file = File::Spec->catfile($test_dir, '.gfconf');
my $test_file1 = File::Spec->catfile($test_dir, 'test.txt');
my $test_file2 = File::Spec->catfile($test_dir, 'test.pl');
my $test_file3 = File::Spec->catfile($test_dir, 'test.log');

# Create test files
open my $fh1, '>', $test_file1 or die "Cannot create $test_file1: $!";
print $fh1 "This is a test file with pattern\n";
close $fh1;

open my $fh2, '>', $test_file2 or die "Cannot create $test_file2: $!";
print $fh2 "#!/usr/bin/perl\n";
print $fh2 "# This has the pattern too\n";
close $fh2;

open my $fh3, '>', $test_file3 or die "Cannot create $test_file3: $!";
print $fh3 "Log entry with pattern\n";
close $fh3;

# Helper function to run gf command and capture output
sub run_gf_command {
    my (@args) = @_;
    
    # Change to original directory to run gf (so it can find lib/)
    my $current_dir = Cwd::getcwd();
    chdir $original_dir or die "Cannot chdir to $original_dir: $!";
    
    my $cmd = [$gf_bin, @args];
    my ($stdin, $stdout, $stderr);
    $stderr = Symbol::gensym();
    
    my $pid = IPC::Open3::open3($stdin, $stdout, $stderr, @$cmd);
    
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

use IPC::Open3;
use Symbol qw(gensym);

plan tests => 12;

# Test 1: Basic configuration file with ignore patterns
{
    # Create config file with ignore patterns
    open my $config_fh, '>', $config_file or die "Cannot create config file: $!";
    print $config_fh "ignore *.log\n";
    print $config_fh "ignore *.tmp\n";
    close $config_fh;
    
    chdir $test_dir or die "Cannot chdir to $test_dir: $!";
    my $result = run_gf_command('--search', 'pattern');
    
    ok($result->{exit_code} == 0, "Search with ignore config exits with code 0");
    like($result->{output}, qr/test\.txt/, "Config allows .txt files");
    like($result->{output}, qr/test\.pl/, "Config allows .pl files");
    unlike($result->{output}, qr/test\.log/, "Config ignores .log files");
}

# Test 2: Configuration file with target patterns
{
    # Create config file with target patterns
    open my $config_fh, '>', $config_file or die "Cannot create config file: $!";
    print $config_fh "target *.pl\n";
    print $config_fh "target *.pm\n";
    close $config_fh;
    
    chdir $test_dir or die "Cannot chdir to $test_dir: $!";
    my $result = run_gf_command('--search', 'pattern');
    
    ok($result->{exit_code} == 0, "Search with target config exits with code 0");
    unlike($result->{output}, qr/test\.txt/, "Config excludes .txt files");
    like($result->{output}, qr/test\.pl/, "Config targets .pl files");
    unlike($result->{output}, qr/test\.log/, "Config excludes .log files");
}

# Test 3: Mixed configuration with both ignore and target patterns
{
    # Create config file with both ignore and target patterns
    open my $config_fh, '>', $config_file or die "Cannot create config file: $!";
    print $config_fh "target *.txt\n";
    print $config_fh "target *.pl\n";
    print $config_fh "ignore *.log\n";
    close $config_fh;
    
    chdir $test_dir or die "Cannot chdir to $test_dir: $!";
    my $result = run_gf_command('--search', 'pattern');
    
    ok($result->{exit_code} == 0, "Search with mixed config exits with code 0");
    like($result->{output}, qr/test\.txt/, "Mixed config includes .txt files");
    like($result->{output}, qr/test\.pl/, "Mixed config includes .pl files");
    unlike($result->{output}, qr/test\.log/, "Mixed config ignores .log files");
}

# Test 4: Command-line options override config
{
    # Create config file that would normally ignore .txt files
    open my $config_fh, '>', $config_file or die "Cannot create config file: $!";
    print $config_fh "ignore *.txt\n";
    close $config_fh;
    
    chdir $test_dir or die "Cannot chdir to $test_dir: $!";
    # Use command-line target to override config ignore
    my $result = run_gf_command('--search', 'pattern', '--target', '*.txt');
    
    ok($result->{exit_code} == 0, "Command-line override exits with code 0");
    like($result->{output}, qr/test\.txt/, "Command-line target overrides config ignore");
}

# Clean up config file
unlink $config_file if -f $config_file;

# Restore original directory
chdir $original_dir or die "Cannot restore directory: $!";

done_testing();