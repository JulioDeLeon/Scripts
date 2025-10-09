#!/usr/bin/env perl

use strict;
use warnings;
use Test::More;
use File::Temp qw(tempdir tempfile);
use File::Spec;
use Cwd qw(abs_path);
use IPC::Open3;
use Symbol qw(gensym);

# Comprehensive regression tests
# Create test suite comparing old vs new implementation output
# Test all command-line options for identical behavior
# Verify configuration file processing remains unchanged

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

# Create a comprehensive test file structure
my $test_file1 = File::Spec->catfile($test_dir, 'test.txt');
my $test_file2 = File::Spec->catfile($test_dir, 'script.pl');
my $test_file3 = File::Spec->catfile($test_dir, 'data.log');
my $test_file4 = File::Spec->catfile($test_dir, 'README.md');
my $config_file = File::Spec->catfile($test_dir, '.gfconf');

# Create test files with known content
open my $fh1, '>', $test_file1 or die "Cannot create $test_file1: $!";
print $fh1 "Line 1: This is a test file\n";
print $fh1 "Line 2: Contains search_term for testing\n";
print $fh1 "Line 3: Another line with content\n";
print $fh1 "Line 4: Final search_term occurrence\n";
close $fh1;

open my $fh2, '>', $test_file2 or die "Cannot create $test_file2: $!";
print $fh2 "#!/usr/bin/perl\n";
print $fh2 "use strict;\n";
print $fh2 "my \$search_term = 'value';\n";
print $fh2 "print \"Hello World\\n\";\n";
close $fh2;

open my $fh3, '>', $test_file3 or die "Cannot create $test_file3: $!";
print $fh3 "2024-01-01 INFO: Application started\n";
print $fh3 "2024-01-01 ERROR: search_term not found\n";
print $fh3 "2024-01-01 INFO: Processing complete\n";
close $fh3;

open my $fh4, '>', $test_file4 or die "Cannot create $test_file4: $!";
print $fh4 "# Test Project\n";
print $fh4 "\n";
print $fh4 "This project contains search_term examples.\n";
print $fh4 "\n";
print $fh4 "## Usage\n";
print $fh4 "Run the search_term command.\n";
close $fh4;

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

plan tests => 20;

# Test 1: Basic functionality regression
{
    chdir $test_dir or die "Cannot chdir to $test_dir: $!";
    my $result = run_gf_command('--search', 'search_term');
    
    ok($result->{exit_code} == 0, "Basic search functionality works");
    like($result->{output}, qr/test\.txt/, "Finds matches in .txt files");
    like($result->{output}, qr/script\.pl/, "Finds matches in .pl files");
    like($result->{output}, qr/data\.log/, "Finds matches in .log files");
    like($result->{output}, qr/README\.md/, "Finds matches in .md files");
}

# Test 2: Line number format regression
{
    chdir $test_dir or die "Cannot chdir to $test_dir: $!";
    my $result = run_gf_command('--search', 'search_term');
    
    like($result->{output}, qr/\[\d+\]/, "Line numbers are formatted with brackets");
    like($result->{output}, qr/\[2\].*search_term/, "Correct line number for first match");
    like($result->{output}, qr/\[4\].*search_term/, "Correct line number for second match");
}

# Test 3: Target pattern regression
{
    chdir $test_dir or die "Cannot chdir to $test_dir: $!";
    my $result = run_gf_command('--search', 'search_term', '--target', '*.pl');
    
    ok($result->{exit_code} == 0, "Target pattern functionality works");
    like($result->{output}, qr/script\.pl/, "Target pattern includes .pl files");
    unlike($result->{output}, qr/test\.txt/, "Target pattern excludes .txt files");
}

# Test 4: Ignore pattern regression
{
    chdir $test_dir or die "Cannot chdir to $test_dir: $!";
    my $result = run_gf_command('--search', 'search_term', '--ignore', '*.log');
    
    ok($result->{exit_code} == 0, "Ignore pattern functionality works");
    unlike($result->{output}, qr/data\.log/, "Ignore pattern excludes .log files");
    like($result->{output}, qr/test\.txt/, "Ignore pattern includes .txt files");
}

# Test 5: Context functionality regression
{
    chdir $test_dir or die "Cannot chdir to $test_dir: $!";
    my $result = run_gf_command('--search', 'search_term', '--context', '1', '--target', '*.txt');
    
    ok($result->{exit_code} == 0, "Context functionality works");
    # Context should show lines around matches
    like($result->{output}, qr/\[1\].*This is a test file/, "Context shows line before match");
    like($result->{output}, qr/\[2\].*search_term/, "Context shows match line");
    like($result->{output}, qr/\[3\].*Another line/, "Context shows line after match");
}

# Test 6: Configuration file regression
{
    # Create config file
    open my $config_fh, '>', $config_file or die "Cannot create config file: $!";
    print $config_fh "ignore *.log\n";
    print $config_fh "target *.txt\n";
    print $config_fh "target *.md\n";
    close $config_fh;
    
    chdir $test_dir or die "Cannot chdir to $test_dir: $!";
    my $result = run_gf_command('--search', 'search_term');
    
    ok($result->{exit_code} == 0, "Configuration file functionality works");
    like($result->{output}, qr/test\.txt/, "Config includes targeted .txt files");
    like($result->{output}, qr/README\.md/, "Config includes targeted .md files");
    unlike($result->{output}, qr/data\.log/, "Config ignores .log files");
    
    # Clean up config file
    unlink $config_file;
}

# Restore original directory
chdir $original_dir or die "Cannot restore directory: $!";

done_testing();