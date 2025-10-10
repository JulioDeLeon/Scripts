#!/usr/bin/env perl

use strict;
use warnings;
use Test::More;
use File::Temp qw(tempdir tempfile);
use File::Spec;
use Cwd qw(abs_path);
use IPC::Open3;
use Symbol qw(gensym);
use Time::HiRes qw(time);

# Store original directory for running gf
our $original_dir = Cwd::getcwd();

# Performance benchmarks for context feature
# Benchmarks context vs non-context performance and tests performance
# with large files and high context values

# Find the gf binary
my $gf_bin = File::Spec->catfile('bin', 'gf');
$gf_bin = abs_path($gf_bin);
if (!-x $gf_bin) {
    plan skip_all => "gf binary not found or not executable at $gf_bin";
}

# Create temporary test directory
my $test_dir = tempdir(CLEANUP => 1);

# Helper function to run gf command and measure execution time
sub run_gf_command_timed {
    my (@args) = @_;
    
    # Change to original directory to run gf (so it can find lib/)
    my $current_dir = Cwd::getcwd();
    chdir $original_dir or die "Cannot chdir to $original_dir: $!";
    
    my $start_time = time();
    
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
    
    my $end_time = time();
    my $execution_time = $end_time - $start_time;
    
    # Change back to test directory
    chdir $current_dir or die "Cannot restore directory: $!";
    
    return {
        output => $output,
        error => $error,
        exit_code => $exit_code,
        execution_time => $execution_time
    };
}

# Helper function to create large test file
sub create_large_test_file {
    my ($filename, $lines, $match_frequency) = @_;
    my $full_path = File::Spec->catfile($test_dir, $filename);
    
    open my $fh, '>', $full_path or die "Cannot create $full_path: $!";
    
    for my $i (1..$lines) {
        if ($i % $match_frequency == 0) {
            print $fh "line $i: this line contains target match for testing\n";
        } else {
            print $fh "line $i: this is regular content without the search term\n";
        }
    }
    
    close $fh;
    return $full_path;
}

# Helper function to run multiple iterations and get average time
sub benchmark_command {
    my ($iterations, @args) = @_;
    my $total_time = 0;
    my $successful_runs = 0;
    
    for my $i (1..$iterations) {
        my $result = run_gf_command_timed(@args);
        if ($result->{exit_code} == 0) {
            $total_time += $result->{execution_time};
            $successful_runs++;
        }
    }
    
    return $successful_runs > 0 ? $total_time / $successful_runs : 0;
}

plan tests => 24;

# Test 1: Benchmark small file - context vs no context
{
    my $small_file = create_large_test_file('small.txt', 100, 10); # 100 lines, match every 10th
    
    chdir $test_dir or die "Cannot chdir to $test_dir: $!";
    
    my $no_context_time = benchmark_command(3, '--search', 'target', '--context', '0');
    my $context_time = benchmark_command(3, '--search', 'target', '--context', '3');
    
    ok($no_context_time > 0, "No context benchmark completed for small file");
    ok($context_time > 0, "Context benchmark completed for small file");
    
    # Context should not be significantly slower (allow up to 3x slower)
    my $slowdown_ratio = $context_time / $no_context_time;
    ok($slowdown_ratio < 3.0, "Context performance acceptable for small file (slowdown: " . sprintf("%.2f", $slowdown_ratio) . "x)");
    
    diag("Small file (100 lines): No context: " . sprintf("%.4f", $no_context_time) . "s, Context: " . sprintf("%.4f", $context_time) . "s");
}

# Test 2: Benchmark medium file - context vs no context
{
    my $medium_file = create_large_test_file('medium.txt', 1000, 50); # 1000 lines, match every 50th
    
    chdir $test_dir or die "Cannot chdir to $test_dir: $!";
    
    my $no_context_time = benchmark_command(3, '--search', 'target', '--context', '0');
    my $context_time = benchmark_command(3, '--search', 'target', '--context', '5');
    
    ok($no_context_time > 0, "No context benchmark completed for medium file");
    ok($context_time > 0, "Context benchmark completed for medium file");
    
    # Context should not be significantly slower (allow up to 4x slower)
    my $slowdown_ratio = $context_time / $no_context_time;
    ok($slowdown_ratio < 4.0, "Context performance acceptable for medium file (slowdown: " . sprintf("%.2f", $slowdown_ratio) . "x)");
    
    diag("Medium file (1000 lines): No context: " . sprintf("%.4f", $no_context_time) . "s, Context: " . sprintf("%.4f", $context_time) . "s");
}

# Test 3: Benchmark large file - context vs no context
{
    my $large_file = create_large_test_file('large.txt', 10000, 100); # 10000 lines, match every 100th
    
    chdir $test_dir or die "Cannot chdir to $test_dir: $!";
    
    my $no_context_time = benchmark_command(2, '--search', 'target', '--context', '0');
    my $context_time = benchmark_command(2, '--search', 'target', '--context', '5');
    
    ok($no_context_time > 0, "No context benchmark completed for large file");
    ok($context_time > 0, "Context benchmark completed for large file");
    
    # Context should not be significantly slower (allow up to 5x slower for large files)
    my $slowdown_ratio = $context_time / $no_context_time;
    ok($slowdown_ratio < 5.0, "Context performance acceptable for large file (slowdown: " . sprintf("%.2f", $slowdown_ratio) . "x)");
    
    diag("Large file (10000 lines): No context: " . sprintf("%.4f", $no_context_time) . "s, Context: " . sprintf("%.4f", $context_time) . "s");
}

# Test 4: Benchmark different context values
{
    my $test_file = create_large_test_file('context_values.txt', 2000, 40); # 2000 lines, match every 40th
    
    chdir $test_dir or die "Cannot chdir to $test_dir: $!";
    
    my $context1_time = benchmark_command(3, '--search', 'target', '--context', '1');
    my $context5_time = benchmark_command(3, '--search', 'target', '--context', '5');
    my $context10_time = benchmark_command(3, '--search', 'target', '--context', '10');
    
    ok($context1_time > 0, "Context 1 benchmark completed");
    ok($context5_time > 0, "Context 5 benchmark completed");
    ok($context10_time > 0, "Context 10 benchmark completed");
    
    # Higher context values should not be dramatically slower
    my $ratio_5_to_1 = $context5_time / $context1_time;
    my $ratio_10_to_1 = $context10_time / $context1_time;
    
    ok($ratio_5_to_1 < 2.0, "Context 5 not dramatically slower than context 1 (ratio: " . sprintf("%.2f", $ratio_5_to_1) . ")");
    ok($ratio_10_to_1 < 3.0, "Context 10 not dramatically slower than context 1 (ratio: " . sprintf("%.2f", $ratio_10_to_1) . ")");
    
    diag("Context values: C1: " . sprintf("%.4f", $context1_time) . "s, C5: " . sprintf("%.4f", $context5_time) . "s, C10: " . sprintf("%.4f", $context10_time) . "s");
}

# Test 5: Benchmark high match frequency (stress test)
{
    my $high_match_file = create_large_test_file('high_match.txt', 1000, 5); # 1000 lines, match every 5th (200 matches)
    
    chdir $test_dir or die "Cannot chdir to $test_dir: $!";
    
    my $no_context_time = benchmark_command(2, '--search', 'target', '--context', '0');
    my $context_time = benchmark_command(2, '--search', 'target', '--context', '3');
    
    ok($no_context_time > 0, "No context benchmark completed for high match frequency");
    ok($context_time > 0, "Context benchmark completed for high match frequency");
    
    # Even with many matches, context should not be excessively slow
    my $slowdown_ratio = $context_time / $no_context_time;
    ok($slowdown_ratio < 6.0, "Context performance acceptable with high match frequency (slowdown: " . sprintf("%.2f", $slowdown_ratio) . "x)");
    
    diag("High match frequency (200 matches): No context: " . sprintf("%.4f", $no_context_time) . "s, Context: " . sprintf("%.4f", $context_time) . "s");
}

# Test 6: Memory usage test with large context values
{
    my $memory_test_file = create_large_test_file('memory_test.txt', 5000, 100); # 5000 lines, match every 100th
    
    chdir $test_dir or die "Cannot chdir to $test_dir: $!";
    
    # Test very high context value to ensure memory management works
    my $result = run_gf_command_timed('--search', 'target', '--context', '50');
    
    ok($result->{exit_code} == 0, "High context value (50) completes successfully");
    ok($result->{execution_time} < 10.0, "High context value completes in reasonable time");
    
    diag("High context value (50): " . sprintf("%.4f", $result->{execution_time}) . "s");
}

# Test 7: Performance with maxline limits
{
    my $maxline_test_file = create_large_test_file('maxline_test.txt', 10000, 50); # 10000 lines, match every 50th
    
    chdir $test_dir or die "Cannot chdir to $test_dir: $!";
    
    my $no_maxline_time = benchmark_command(2, '--search', 'target', '--context', '3');
    my $maxline_time = benchmark_command(2, '--search', 'target', '--context', '3', '--maxline', '1000');
    
    ok($no_maxline_time > 0, "No maxline benchmark completed");
    ok($maxline_time > 0, "Maxline benchmark completed");
    
    # Maxline should improve performance by limiting processing
    ok($maxline_time <= $no_maxline_time, "Maxline improves or maintains performance");
    
    diag("Maxline performance: No limit: " . sprintf("%.4f", $no_maxline_time) . "s, With limit: " . sprintf("%.4f", $maxline_time) . "s");
}

# Test 8: Performance consistency test
{
    my $consistency_file = create_large_test_file('consistency.txt', 2000, 25); # 2000 lines, match every 25th
    
    chdir $test_dir or die "Cannot chdir to $test_dir: $!";
    
    # Run multiple iterations to check consistency
    my @times;
    for my $i (1..5) {
        my $result = run_gf_command_timed('--search', 'target', '--context', '3');
        if ($result->{exit_code} == 0) {
            push @times, $result->{execution_time};
        }
    }
    
    ok(@times >= 3, "At least 3 successful consistency test runs");
    
    if (@times >= 3) {
        # Calculate standard deviation
        my $mean = (sum(@times)) / @times;
        my $variance = (sum(map { ($_ - $mean) ** 2 } @times)) / @times;
        my $std_dev = sqrt($variance);
        my $coefficient_of_variation = $std_dev / $mean;
        
        # Performance should be reasonably consistent (CV < 0.5)
        ok($coefficient_of_variation < 0.5, "Performance is consistent (CV: " . sprintf("%.3f", $coefficient_of_variation) . ")");
        
        diag("Consistency test - Mean: " . sprintf("%.4f", $mean) . "s, StdDev: " . sprintf("%.4f", $std_dev) . "s, CV: " . sprintf("%.3f", $coefficient_of_variation));
    }
}

# Helper function for sum
sub sum {
    my $total = 0;
    $total += $_ for @_;
    return $total;
}

# Restore original directory
chdir $original_dir or die "Cannot restore directory: $!";

done_testing();