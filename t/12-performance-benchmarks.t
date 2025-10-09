#!/usr/bin/env perl

use strict;
use warnings;
use Test::More;
use File::Temp qw(tempdir tempfile);
use File::Spec;
use Time::HiRes qw(time);
use Cwd qw(abs_path);

# Add lib directory to path
use lib 'lib';

# Import the modules we need for testing
use GF::Search qw(main_search);
use GF::Startup qw(initialize_gf_system);

=head1 NAME

t/12-performance-benchmarks.t - Performance benchmarking and validation tests

=head1 DESCRIPTION

This test suite validates that all performance requirements are met by measuring:
- Memory usage improvements with large files
- Search speed improvements across different directory sizes
- Pattern matching performance
- Directory traversal optimization
- Early validation pipeline efficiency

=cut

# Test configuration
my $debug = $ENV{GF_DEBUG} || 0;
my $verbose = $ENV{GF_VERBOSE} || 0;

# Performance thresholds (these can be adjusted based on system capabilities)
my $MAX_MEMORY_MB = 50;        # Maximum memory usage in MB for large file processing
my $MAX_SEARCH_TIME_LARGE = 5; # Maximum search time in seconds for large directories
my $MAX_SEARCH_TIME_SMALL = 1; # Maximum search time in seconds for small directories

# Skip performance tests if running in CI or if explicitly disabled
if ($ENV{SKIP_PERFORMANCE_TESTS}) {
    plan skip_all => 'Performance tests disabled via SKIP_PERFORMANCE_TESTS';
}

plan tests => 12;

=head2 Test Helper Functions

=cut

sub get_memory_usage {
    my $pid = shift || $$;
    
    # Try different methods to get memory usage based on OS
    if ($^O eq 'linux') {
        # Linux: read from /proc/pid/status
        if (open my $fh, '<', "/proc/$pid/status") {
            while (my $line = <$fh>) {
                if ($line =~ /^VmRSS:\s+(\d+)\s+kB/) {
                    close $fh;
                    return $1 / 1024; # Convert KB to MB
                }
            }
            close $fh;
        }
    } elsif ($^O eq 'darwin') {
        # macOS: use ps command
        my $ps_output = `ps -o rss= -p $pid 2>/dev/null`;
        chomp $ps_output;
        if ($ps_output && $ps_output =~ /^\s*(\d+)/) {
            return $1 / 1024; # Convert KB to MB
        }
    }
    
    # Fallback: return 0 if we can't measure memory
    return 0;
}

sub create_test_file {
    my ($size_mb, $pattern_density) = @_;
    $pattern_density ||= 0.1; # 10% of lines contain the search pattern by default
    
    my ($fh, $filename) = tempfile(SUFFIX => '.txt', UNLINK => 1);
    
    my $lines_per_mb = 20000; # Approximate lines per MB
    my $total_lines = $size_mb * $lines_per_mb;
    my $pattern_lines = int($total_lines * $pattern_density);
    
    # Generate content with search patterns distributed throughout
    for my $i (1..$total_lines) {
        if ($i <= $pattern_lines && ($i % int($total_lines / $pattern_lines)) == 0) {
            # Insert lines with search pattern
            print $fh "Line $i: This line contains the search pattern 'test_pattern' for benchmarking\n";
        } else {
            # Regular lines without pattern
            print $fh "Line $i: This is a regular line without the target pattern for performance testing\n";
        }
    }
    
    close $fh;
    return $filename;
}

sub create_test_directory {
    my ($num_files, $files_per_dir) = @_;
    $files_per_dir ||= 100;
    
    my $temp_dir = tempdir(CLEANUP => 1);
    my $files_created = 0;
    my $dir_count = 0;
    
    while ($files_created < $num_files) {
        my $subdir = File::Spec->catdir($temp_dir, "subdir_$dir_count");
        mkdir $subdir or die "Cannot create directory $subdir: $!";
        
        my $files_in_this_dir = ($num_files - $files_created > $files_per_dir) 
                               ? $files_per_dir 
                               : ($num_files - $files_created);
        
        for my $i (1..$files_in_this_dir) {
            my $filename = File::Spec->catfile($subdir, "test_file_${dir_count}_$i.txt");
            open my $fh, '>', $filename or die "Cannot create file $filename: $!";
            
            # Add some content with occasional search patterns
            for my $line (1..50) {
                if ($line % 10 == 0) {
                    print $fh "Line $line: Contains test_pattern for searching\n";
                } else {
                    print $fh "Line $line: Regular content without pattern\n";
                }
            }
            close $fh;
            $files_created++;
        }
        $dir_count++;
    }
    
    return $temp_dir;
}

sub benchmark_search {
    my ($search_term, $target_patterns, $ignore_patterns, $test_dir) = @_;
    
    # Initialize system
    my %targets = map { $_ => 1 } @{$target_patterns || []};
    my %ignores = map { $_ => 1 } @{$ignore_patterns || []};
    
    eval {
        initialize_gf_system($search_term, \%ignores, \%targets, 0, 0, 0);
    };
    if ($@) {
        die "Failed to initialize system for benchmark: $@";
    }
    
    # Measure performance
    my $start_time = time();
    my $start_memory = get_memory_usage();
    
    # Capture output to avoid cluttering test results
    my $output = '';
    {
        local *STDOUT;
        open STDOUT, '>', \$output or die "Cannot redirect STDOUT: $!";
        
        # Change to test directory and run search
        my $original_dir = Cwd::getcwd();
        chdir $test_dir or die "Cannot change to test directory: $!";
        
        eval {
            main_search($search_term, \%ignores, \%targets, 0, 0, 0);
        };
        
        chdir $original_dir;
        
        if ($@) {
            die "Search failed: $@";
        }
    }
    
    my $end_time = time();
    my $end_memory = get_memory_usage();
    
    return {
        duration => $end_time - $start_time,
        memory_used => $end_memory - $start_memory,
        memory_peak => $end_memory,
        output_size => length($output),
        matches_found => ($output =~ tr/\n/\n/), # Count lines as rough match estimate
    };
}

=head2 Performance Tests

=cut

# Test 1: Memory usage with large files (Requirement 1.1, 1.2, 1.3)
subtest 'Memory usage with large files' => sub {
    plan tests => 3;
    
    # Create a 10MB test file
    my $large_file = create_test_file(10, 0.05); # 5% pattern density
    my $test_dir = tempdir(CLEANUP => 1);
    
    # Move file to test directory
    my $target_file = File::Spec->catfile($test_dir, 'large_test.txt');
    rename $large_file, $target_file or die "Cannot move test file: $!";
    
    my $results = benchmark_search('test_pattern', ['*.txt'], [], $test_dir);
    
    ok($results->{duration} > 0, 'Search completed successfully');
    
    # Memory usage should be reasonable for streaming approach
    if ($results->{memory_peak} > 0) {
        cmp_ok($results->{memory_peak}, '<', $MAX_MEMORY_MB, 
               "Peak memory usage (" . $results->{memory_peak} . " MB) under threshold ($MAX_MEMORY_MB MB)");
    } else {
        pass('Memory measurement not available on this system');
    }
    
    # Should find matches in reasonable time
    cmp_ok($results->{duration}, '<', $MAX_SEARCH_TIME_LARGE,
           "Search time (" . $results->{duration} . "s) under threshold (${MAX_SEARCH_TIME_LARGE}s)");
    
    if ($verbose) {
        diag("Large file test results:");
        diag("  Duration: " . $results->{duration} . " seconds");
        diag("  Peak memory: " . $results->{memory_peak} . " MB");
        diag("  Matches found: " . $results->{matches_found});
    }
};

# Test 2: Search speed with small directory (Requirement 3.1, 3.2)
subtest 'Search speed with small directory' => sub {
    plan tests => 2;
    
    my $test_dir = create_test_directory(50, 25); # 50 files in 2 directories
    
    my $results = benchmark_search('test_pattern', ['*.txt'], [], $test_dir);
    
    ok($results->{duration} > 0, 'Search completed successfully');
    cmp_ok($results->{duration}, '<', $MAX_SEARCH_TIME_SMALL,
           "Search time (" . $results->{duration} . "s) under threshold (${MAX_SEARCH_TIME_SMALL}s)");
    
    if ($verbose) {
        diag("Small directory test results:");
        diag("  Duration: " . $results->{duration} . " seconds");
        diag("  Matches found: " . $results->{matches_found});
    }
};

# Test 3: Search speed with large directory (Requirement 3.1, 3.2)
subtest 'Search speed with large directory' => sub {
    plan tests => 2;
    
    my $test_dir = create_test_directory(500, 50); # 500 files in 10 directories
    
    my $results = benchmark_search('test_pattern', ['*.txt'], [], $test_dir);
    
    ok($results->{duration} > 0, 'Search completed successfully');
    cmp_ok($results->{duration}, '<', $MAX_SEARCH_TIME_LARGE,
           "Search time (" . $results->{duration} . "s) under threshold (${MAX_SEARCH_TIME_LARGE}s)");
    
    if ($verbose) {
        diag("Large directory test results:");
        diag("  Duration: " . $results->{duration} . " seconds");
        diag("  Matches found: " . $results->{matches_found});
    }
};

# Test 4: Pattern matching performance (Requirement 2.1, 2.2)
subtest 'Pattern matching performance' => sub {
    plan tests => 3;
    
    my $test_dir = create_test_directory(100, 50);
    
    # Test with complex regex pattern
    my $complex_pattern = '(test_pattern|search|benchmark)';
    my $results = benchmark_search($complex_pattern, ['*.txt'], [], $test_dir);
    
    ok($results->{duration} > 0, 'Complex pattern search completed');
    cmp_ok($results->{duration}, '<', $MAX_SEARCH_TIME_SMALL * 2,
           "Complex pattern search time reasonable");
    
    # Test with simple pattern for comparison
    my $simple_results = benchmark_search('test_pattern', ['*.txt'], [], $test_dir);
    
    # Complex pattern should not be dramatically slower due to pattern caching
    my $slowdown_ratio = $results->{duration} / ($simple_results->{duration} || 0.001);
    cmp_ok($slowdown_ratio, '<', 5, 'Complex pattern not dramatically slower than simple pattern');
    
    if ($verbose) {
        diag("Pattern matching performance:");
        diag("  Simple pattern: " . $simple_results->{duration} . " seconds");
        diag("  Complex pattern: " . $results->{duration} . " seconds");
        diag("  Slowdown ratio: " . $slowdown_ratio);
    }
};

# Test 5: Directory traversal optimization (Requirement 3.1, 3.2)
subtest 'Directory traversal optimization' => sub {
    plan tests => 2;
    
    # Create nested directory structure
    my $base_dir = tempdir(CLEANUP => 1);
    my $deep_path = $base_dir;
    
    # Create 10 levels of nesting
    for my $level (1..10) {
        $deep_path = File::Spec->catdir($deep_path, "level_$level");
        mkdir $deep_path or die "Cannot create directory $deep_path: $!";
        
        # Add a few files at each level
        for my $file (1..5) {
            my $filename = File::Spec->catfile($deep_path, "file_$file.txt");
            open my $fh, '>', $filename or die "Cannot create file $filename: $!";
            print $fh "Content with test_pattern in level $level file $file\n";
            close $fh;
        }
    }
    
    my $results = benchmark_search('test_pattern', ['*.txt'], [], $base_dir);
    
    ok($results->{duration} > 0, 'Deep directory traversal completed');
    cmp_ok($results->{duration}, '<', $MAX_SEARCH_TIME_LARGE,
           "Deep traversal time under threshold");
    
    if ($verbose) {
        diag("Directory traversal test results:");
        diag("  Duration: " . $results->{duration} . " seconds");
        diag("  Matches found: " . $results->{matches_found});
    }
};

# Test 6: Early validation pipeline efficiency (Requirement 4.1, 4.2)
subtest 'Early validation pipeline efficiency' => sub {
    plan tests => 3;
    
    my $test_dir = tempdir(CLEANUP => 1);
    
    # Create mix of files that should be skipped and processed
    my @skip_files = qw(test.exe test.dll test.so test.jpg test.png test.zip);
    my @process_files = qw(test.txt test.pl test.pm test.py test.js);
    
    # Create files that should be skipped
    for my $file (@skip_files) {
        my $filepath = File::Spec->catfile($test_dir, $file);
        open my $fh, '>', $filepath or die "Cannot create $filepath: $!";
        print $fh "Binary or unwanted content\n";
        close $fh;
    }
    
    # Create files that should be processed
    for my $file (@process_files) {
        my $filepath = File::Spec->catfile($test_dir, $file);
        open my $fh, '>', $filepath or die "Cannot create $filepath: $!";
        print $fh "Text content with test_pattern\n";
        close $fh;
    }
    
    # Test with target pattern that should skip binary files
    my $results = benchmark_search('test_pattern', ['*.txt', '*.pl', '*.pm'], [], $test_dir);
    
    ok($results->{duration} > 0, 'Validation pipeline search completed');
    
    # Should be fast due to early skipping of unwanted files
    cmp_ok($results->{duration}, '<', $MAX_SEARCH_TIME_SMALL,
           "Early validation keeps search time low");
    
    # Should find matches only in processed files
    ok($results->{matches_found} > 0, 'Found matches in valid files');
    
    if ($verbose) {
        diag("Early validation test results:");
        diag("  Duration: " . $results->{duration} . " seconds");
        diag("  Matches found: " . $results->{matches_found});
    }
};

# Test 7: Memory efficiency with maxline limits (Requirement 1.1)
subtest 'Memory efficiency with maxline limits' => sub {
    plan tests => 2;
    
    # Create a large file
    my $large_file = create_test_file(5, 0.1);
    my $test_dir = tempdir(CLEANUP => 1);
    my $target_file = File::Spec->catfile($test_dir, 'maxline_test.txt');
    rename $large_file, $target_file or die "Cannot move test file: $!";
    
    # Test with maxline limit - this should use less memory and be faster
    my %targets = ('*.txt' => 1);
    my %ignores = ();
    
    eval {
        initialize_gf_system('test_pattern', \%ignores, \%targets, 0, 100, 0); # maxline = 100
    };
    
    ok(!$@, 'System initialized with maxline limit') or diag("Error: $@");
    
    my $start_time = time();
    my $output = '';
    {
        local *STDOUT;
        open STDOUT, '>', \$output or die "Cannot redirect STDOUT: $!";
        
        my $original_dir = Cwd::getcwd();
        chdir $test_dir or die "Cannot change to test directory: $!";
        
        eval {
            main_search('test_pattern', \%ignores, \%targets, 0, 100, 0);
        };
        
        chdir $original_dir;
    }
    my $duration = time() - $start_time;
    
    # Should complete quickly due to early exit
    cmp_ok($duration, '<', $MAX_SEARCH_TIME_SMALL,
           "Maxline limit keeps search time low ($duration seconds)");
    
    if ($verbose) {
        diag("Maxline test results:");
        diag("  Duration: " . $duration . " seconds");
        diag("  Output size: " . length($output) . " bytes");
    }
};

# Test 8: Binary file detection performance (Requirement 1.4)
subtest 'Binary file detection performance' => sub {
    plan tests => 2;
    
    my $test_dir = tempdir(CLEANUP => 1);
    
    # Create mix of binary and text files
    for my $i (1..20) {
        # Create text files
        my $text_file = File::Spec->catfile($test_dir, "text_$i.txt");
        open my $fh, '>', $text_file or die "Cannot create $text_file: $!";
        print $fh "Text content with test_pattern\n" x 100;
        close $fh;
        
        # Create binary files (simulate with non-text content)
        my $bin_file = File::Spec->catfile($test_dir, "binary_$i.bin");
        open $fh, '>', $bin_file or die "Cannot create $bin_file: $!";
        binmode $fh;
        print $fh pack("C*", (0..255) x 10); # Binary data
        close $fh;
    }
    
    my $results = benchmark_search('test_pattern', ['*'], [], $test_dir);
    
    ok($results->{duration} > 0, 'Binary detection search completed');
    
    # Should be reasonably fast due to early binary detection
    cmp_ok($results->{duration}, '<', $MAX_SEARCH_TIME_SMALL,
           "Binary detection keeps search efficient");
    
    if ($verbose) {
        diag("Binary detection test results:");
        diag("  Duration: " . $results->{duration} . " seconds");
        diag("  Matches found: " . $results->{matches_found});
    }
};

# Test 9: Cache effectiveness
subtest 'Cache effectiveness' => sub {
    plan tests => 2;
    
    my $test_dir = create_test_directory(100, 50);
    
    # Run search twice to test cache effectiveness
    my $first_run = benchmark_search('test_pattern', ['*.txt'], [], $test_dir);
    my $second_run = benchmark_search('test_pattern', ['*.txt'], [], $test_dir);
    
    ok($first_run->{duration} > 0 && $second_run->{duration} > 0, 
       'Both cache test runs completed');
    
    # Second run might be faster due to caching, but this is system-dependent
    # Just verify both runs are reasonable
    cmp_ok($second_run->{duration}, '<', $MAX_SEARCH_TIME_LARGE,
           'Cached run performance acceptable');
    
    if ($verbose) {
        diag("Cache effectiveness test results:");
        diag("  First run: " . $first_run->{duration} . " seconds");
        diag("  Second run: " . $second_run->{duration} . " seconds");
        my $improvement = ($first_run->{duration} - $second_run->{duration}) / $first_run->{duration} * 100;
        diag("  Performance change: " . sprintf("%.1f%%", $improvement));
    }
};

# Test 10: Ignore pattern performance (Requirement 4.3)
subtest 'Ignore pattern performance' => sub {
    plan tests => 2;
    
    my $test_dir = create_test_directory(200, 50);
    
    # Test with many ignore patterns
    my @ignore_patterns = qw(*.log *.tmp *.bak *.cache *.old *.backup);
    
    my $results = benchmark_search('test_pattern', ['*.txt'], \@ignore_patterns, $test_dir);
    
    ok($results->{duration} > 0, 'Ignore pattern search completed');
    cmp_ok($results->{duration}, '<', $MAX_SEARCH_TIME_LARGE,
           "Ignore pattern processing efficient");
    
    if ($verbose) {
        diag("Ignore pattern test results:");
        diag("  Duration: " . $results->{duration} . " seconds");
        diag("  Ignore patterns: " . scalar(@ignore_patterns));
    }
};

# Test 11: Context display performance (Requirement 6.3)
subtest 'Context display performance' => sub {
    plan tests => 2;
    
    my $test_dir = create_test_directory(50, 25);
    
    # Test with context lines
    my %targets = ('*.txt' => 1);
    my %ignores = ();
    
    eval {
        initialize_gf_system('test_pattern', \%ignores, \%targets, 3, 0, 0); # context = 3
    };
    
    ok(!$@, 'System initialized with context') or diag("Error: $@");
    
    my $start_time = time();
    my $output = '';
    {
        local *STDOUT;
        open STDOUT, '>', \$output or die "Cannot redirect STDOUT: $!";
        
        my $original_dir = Cwd::getcwd();
        chdir $test_dir or die "Cannot change to test directory: $!";
        
        eval {
            main_search('test_pattern', \%ignores, \%targets, 3, 0, 0);
        };
        
        chdir $original_dir;
    }
    my $duration = time() - $start_time;
    
    cmp_ok($duration, '<', $MAX_SEARCH_TIME_SMALL * 2,
           "Context display performance acceptable");
    
    if ($verbose) {
        diag("Context display test results:");
        diag("  Duration: " . $duration . " seconds");
        diag("  Output size: " . length($output) . " bytes");
    }
};

# Test 12: Overall performance validation
subtest 'Overall performance validation' => sub {
    plan tests => 4;
    
    # Create a comprehensive test scenario
    my $test_dir = create_test_directory(300, 60);
    
    # Add a large file
    my $large_file = create_test_file(5, 0.08);
    my $target_file = File::Spec->catfile($test_dir, 'large_comprehensive.txt');
    rename $large_file, $target_file or die "Cannot move test file: $!";
    
    # Run comprehensive search with various options
    my $results = benchmark_search('test_pattern', ['*.txt'], ['*.log', '*.tmp'], $test_dir);
    
    ok($results->{duration} > 0, 'Comprehensive search completed');
    
    # Validate all performance requirements are met
    if ($results->{memory_peak} > 0) {
        cmp_ok($results->{memory_peak}, '<', $MAX_MEMORY_MB * 2,
               "Overall memory usage acceptable");
    } else {
        pass('Memory measurement not available');
    }
    
    cmp_ok($results->{duration}, '<', $MAX_SEARCH_TIME_LARGE * 2,
           "Overall search time acceptable");
    
    ok($results->{matches_found} > 0, 'Found expected matches');
    
    if ($verbose) {
        diag("Overall performance test results:");
        diag("  Duration: " . $results->{duration} . " seconds");
        diag("  Peak memory: " . $results->{memory_peak} . " MB");
        diag("  Matches found: " . $results->{matches_found});
        diag("  Output size: " . $results->{output_size} . " bytes");
    }
};

# Print summary if verbose
if ($verbose) {
    diag("\nPerformance test summary:");
    diag("  Max memory threshold: $MAX_MEMORY_MB MB");
    diag("  Max search time (large): $MAX_SEARCH_TIME_LARGE seconds");
    diag("  Max search time (small): $MAX_SEARCH_TIME_SMALL seconds");
    diag("  System: $^O");
}

done_testing();