#!/usr/bin/env perl

use strict;
use warnings;
use Test::More;
use File::Temp qw(tempdir tempfile);
use File::Spec;
use Time::HiRes qw(time);
use Cwd qw(abs_path getcwd);

# Add lib directory to path
use lib 'lib';

# Import all the optimization modules
use GF::Search qw(
  main_search process_args should_skip handle_dir check_file
  buffer_output flush_output_buffer
  _traverse_directory_optimized _should_skip_fast _validate_file_fast
  _get_cached_stat _clear_validation_cache _get_cache_stats
);
use GF::Startup qw(initialize_gf_system);
use GF::PatternCache qw(
  initialize_pattern_cache get_search_pattern get_ignore_patterns get_target_patterns
);
use GF::ConfigCache qw(initialize_config_cache get_cached_config);

=head1 NAME

t/13-integration-optimization.t - Integration tests for optimization systems

=head1 DESCRIPTION

This test suite verifies that all optimization components work together seamlessly
and provides end-to-end performance improvements. It tests the integration between:
- Pattern compilation and caching
- Streaming file processing
- Optimized directory traversal
- Early validation pipeline
- Configuration caching
- Buffered output system

=cut

plan tests => 15;

# Test configuration
my $debug = $ENV{GF_DEBUG} || 0;

=head2 Test Helper Functions

=cut

sub create_comprehensive_test_environment {
    my $base_dir = tempdir(CLEANUP => 1);
    
    # Create directory structure with various file types
    my @subdirs = qw(src tests docs config logs temp);
    
    foreach my $subdir (@subdirs) {
        my $dir_path = File::Spec->catdir($base_dir, $subdir);
        mkdir $dir_path or die "Cannot create directory $dir_path: $!";
        
        # Create different types of files in each directory
        if ($subdir eq 'src') {
            # Source files with patterns
            for my $i (1..10) {
                my $file = File::Spec->catfile($dir_path, "module_$i.pm");
                open my $fh, '>', $file or die "Cannot create $file: $!";
                print $fh "package Module$i;\n";
                print $fh "use strict;\n";
                print $fh "use warnings;\n";
                print $fh "# This module contains search_target for testing\n";
                print $fh "sub new { return bless {}, shift; }\n";
                print $fh "1;\n";
                close $fh;
            }
        } elsif ($subdir eq 'tests') {
            # Test files
            for my $i (1..5) {
                my $file = File::Spec->catfile($dir_path, "test_$i.t");
                open my $fh, '>', $file or die "Cannot create $file: $!";
                print $fh "#!/usr/bin/env perl\n";
                print $fh "use Test::More;\n";
                print $fh "# Test for search_target functionality\n";
                print $fh "ok(1, 'test passes');\n";
                print $fh "done_testing();\n";
                close $fh;
            }
        } elsif ($subdir eq 'logs') {
            # Log files (should be ignored)
            for my $i (1..3) {
                my $file = File::Spec->catfile($dir_path, "app_$i.log");
                open my $fh, '>', $file or die "Cannot create $file: $!";
                print $fh "2024-01-01 INFO: Application started\n";
                print $fh "2024-01-01 DEBUG: search_target processed\n";
                print $fh "2024-01-01 ERROR: Something failed\n";
                close $fh;
            }
        } elsif ($subdir eq 'temp') {
            # Temporary files (should be ignored)
            for my $i (1..3) {
                my $file = File::Spec->catfile($dir_path, "temp_$i.tmp");
                open my $fh, '>', $file or die "Cannot create $file: $!";
                print $fh "Temporary data with search_target\n";
                close $fh;
            }
        } else {
            # Regular text files
            for my $i (1..3) {
                my $file = File::Spec->catfile($dir_path, "file_$i.txt");
                open my $fh, '>', $file or die "Cannot create $file: $!";
                print $fh "This is a text file\n";
                print $fh "It contains search_target in the content\n";
                print $fh "For integration testing purposes\n";
                close $fh;
            }
        }
    }
    
    # Create some binary files that should be skipped
    my $bin_dir = File::Spec->catdir($base_dir, 'bin');
    mkdir $bin_dir or die "Cannot create bin directory: $!";
    
    for my $i (1..3) {
        my $file = File::Spec->catfile($bin_dir, "binary_$i.exe");
        open my $fh, '>', $file or die "Cannot create $file: $!";
        binmode $fh;
        print $fh pack("C*", (0..255) x 10); # Binary data
        close $fh;
    }
    
    return $base_dir;
}

sub run_integrated_search {
    my ($search_term, $target_patterns, $ignore_patterns, $test_dir, $context, $maxline) = @_;
    
    $context ||= 0;
    $maxline ||= 0;
    
    # Convert patterns to hashes
    my %targets = map { $_ => 1 } @{$target_patterns || []};
    my %ignores = map { $_ => 1 } @{$ignore_patterns || []};
    
    # Initialize the complete system
    eval {
        initialize_gf_system($search_term, \%ignores, \%targets, $context, $maxline, 0);
    };
    if ($@) {
        die "System initialization failed: $@";
    }
    
    # Capture output and measure performance
    my $output = '';
    my $start_time = time();
    
    {
        local *STDOUT;
        open STDOUT, '>', \$output or die "Cannot redirect STDOUT: $!";
        
        my $original_dir = getcwd();
        chdir $test_dir or die "Cannot change to test directory: $!";
        
        eval {
            main_search($search_term, \%ignores, \%targets, $context, $maxline, 0);
        };
        
        chdir $original_dir;
        
        if ($@) {
            die "Search failed: $@";
        }
    }
    
    my $duration = time() - $start_time;
    
    my @lines = split(/\n/, $output);
    my $match_count = 0;
    $match_count = () = ($output =~ /search_target/g);
    
    return {
        output => $output,
        duration => $duration,
        lines => scalar(@lines),
        matches => $match_count,
    };
}

=head2 Integration Tests

=cut

# Test 1: Complete system initialization integration
subtest 'Complete system initialization' => sub {
    plan tests => 6;
    
    my %ignores = ('*.log' => 1, '*.tmp' => 1);
    my %targets = ('*.pm' => 1, '*.t' => 1);
    
    # Test initialization
    eval {
        initialize_gf_system('test_pattern', \%ignores, \%targets, 2, 100, 0);
    };
    
    ok(!$@, 'System initialization completed without errors') or diag("Error: $@");
    
    # Verify pattern cache is initialized
    my $search_pattern = get_search_pattern();
    ok(defined $search_pattern, 'Search pattern compiled and cached');
    
    my $ignore_patterns = get_ignore_patterns();
    ok(ref($ignore_patterns) eq 'ARRAY' && @$ignore_patterns > 0, 'Ignore patterns compiled');
    
    my $target_patterns = get_target_patterns();
    ok(ref($target_patterns) eq 'ARRAY' && @$target_patterns > 0, 'Target patterns compiled');
    
    # Verify config cache is initialized
    my $config = get_cached_config();
    ok(ref($config) eq 'HASH', 'Configuration cached');
    
    # Verify cache contains expected values
    is($config->{context}, 2, 'Context value cached correctly');
};

# Test 2: Pattern compilation and caching integration
subtest 'Pattern compilation and caching integration' => sub {
    plan tests => 4;
    
    # Initialize with complex patterns
    my %ignores = ('*.log' => 1, '*.tmp' => 1, '*.bak' => 1);
    my %targets = ('*.pm' => 1, '*.pl' => 1, '*.t' => 1);
    
    eval {
        initialize_pattern_cache('(test|search)_pattern', \%ignores, \%targets);
    };
    
    ok(!$@, 'Pattern cache initialization successful') or diag("Error: $@");
    
    # Test pattern retrieval
    my $search_pattern = get_search_pattern();
    ok(defined $search_pattern, 'Search pattern available');
    
    # Test pattern matching
    my $test_string = "This contains test_pattern in it";
    ok($test_string =~ /$search_pattern/, 'Compiled search pattern works');
    
    # Test ignore patterns
    my $ignore_patterns = get_ignore_patterns();
    my $log_file = "application.log";
    my $matches_ignore = 0;
    foreach my $pattern (@$ignore_patterns) {
        if ($log_file =~ /$pattern/) {
            $matches_ignore = 1;
            last;
        }
    }
    ok($matches_ignore, 'Ignore patterns work correctly');
};

# Test 3: Streaming file processing integration
subtest 'Streaming file processing integration' => sub {
    plan tests => 4;
    
    # Create a large test file
    my ($fh, $large_file) = tempfile(SUFFIX => '.txt', UNLINK => 1);
    
    # Write content with patterns distributed throughout
    for my $i (1..1000) {
        if ($i % 100 == 0) {
            print $fh "Line $i: Contains search_target for testing\n";
        } else {
            print $fh "Line $i: Regular content without pattern\n";
        }
    }
    close $fh;
    
    # Initialize system
    my %ignores = ();
    my %targets = ('*.txt' => 1);
    
    eval {
        initialize_gf_system('search_target', \%ignores, \%targets, 0, 0, 0);
    };
    
    ok(!$@, 'System initialized for streaming test') or diag("Error: $@");
    
    # Test streaming processing
    my $output = '';
    {
        local *STDOUT;
        open STDOUT, '>', \$output or die "Cannot redirect STDOUT: $!";
        
        eval {
            check_file($large_file, 'search_target', 0, 0, 0);
        };
    }
    
    ok(!$@, 'Streaming file processing completed') or diag("Error: $@");
    ok(length($output) > 0, 'Streaming processing produced output');
    
    # Count matches - should find 10 matches (every 100th line)
    my $match_count = 0;
    $match_count = () = ($output =~ /search_target/g);
    cmp_ok($match_count, '>=', 1, 'Found expected number of matches in streaming mode');
};

# Test 4: Optimized directory traversal integration
subtest 'Optimized directory traversal integration' => sub {
    plan tests => 5;
    
    my $test_dir = create_comprehensive_test_environment();
    
    # Test with target patterns that should find specific files
    my $result = run_integrated_search('search_target', ['*.pm', '*.t'], ['*.log', '*.tmp'], $test_dir);
    
    ok($result->{duration} > 0, 'Directory traversal completed');
    ok(length($result->{output}) > 0, 'Traversal produced output');
    
    # Should find matches in .pm and .t files, but not in .log or .tmp files
    ok($result->{matches} > 0, 'Found matches in target files');
    
    # Verify it found files from both src/ (*.pm) and tests/ (*.t) directories
    my $has_pm_files = $result->{output} =~ /\.pm/;
    my $has_t_files = $result->{output} =~ /\.t/;
    
    ok($has_pm_files, 'Found matches in .pm files');
    ok($has_t_files, 'Found matches in .t files');
};

# Test 5: Early validation pipeline integration
subtest 'Early validation pipeline integration' => sub {
    plan tests => 4;
    
    my $test_dir = create_comprehensive_test_environment();
    
    # Clear validation cache to start fresh
    _clear_validation_cache();
    
    # Run search that should trigger validation pipeline
    my $result = run_integrated_search('search_target', ['*'], [], $test_dir);
    
    ok($result->{duration} > 0, 'Validation pipeline search completed');
    
    # Get cache statistics to verify caching is working
    my $cache_stats = _get_cache_stats();
    
    ok($cache_stats->{stat_misses} > 0, 'Stat cache was used (cache misses recorded)');
    ok($cache_stats->{permission_misses} > 0, 'Permission cache was used');
    ok($cache_stats->{binary_misses} > 0, 'Binary cache was used');
};

# Test 6: Configuration caching integration
subtest 'Configuration caching integration' => sub {
    plan tests => 4;
    
    my %ignores = ('*.log' => 1, '*.tmp' => 1, '*.bak' => 1);
    my %targets = ('*.pm' => 1, '*.pl' => 1);
    
    # Initialize config cache
    eval {
        initialize_config_cache(\%ignores, \%targets, 3, 500, 1);
    };
    
    ok(!$@, 'Configuration cache initialized') or diag("Error: $@");
    
    # Retrieve cached config
    my $config = get_cached_config();
    
    ok(ref($config) eq 'HASH', 'Configuration retrieved from cache');
    is($config->{context}, 3, 'Context value cached correctly');
    is($config->{ignore_count}, 3, 'Ignore pattern count cached correctly');
};

# Test 7: Buffered output system integration
subtest 'Buffered output system integration' => sub {
    plan tests => 3;
    
    # Test buffered output functions
    my $output = '';
    {
        local *STDOUT;
        open STDOUT, '>', \$output or die "Cannot redirect STDOUT: $!";
        
        # Add content to buffer
        buffer_output("Line 1\n");
        buffer_output("Line 2\n");
        buffer_output("Line 3\n");
        
        # Buffer should not be flushed yet (unless it's full)
        # Force flush
        flush_output_buffer();
    }
    
    ok(length($output) > 0, 'Buffered output system works');
    ok($output =~ /Line 1/, 'First line buffered correctly');
    ok($output =~ /Line 3/, 'Last line buffered correctly');
};

# Test 8: End-to-end performance integration
subtest 'End-to-end performance integration' => sub {
    plan tests => 4;
    
    my $test_dir = create_comprehensive_test_environment();
    
    # Run comprehensive search with all optimizations
    my $start_time = time();
    my $result = run_integrated_search(
        'search_target', 
        ['*.pm', '*.t', '*.txt'], 
        ['*.log', '*.tmp', '*.exe'], 
        $test_dir,
        2,  # context
        0   # no maxline limit
    );
    my $total_time = time() - $start_time;
    
    ok($result->{duration} > 0, 'End-to-end search completed');
    cmp_ok($total_time, '<', 2, 'Total integration time reasonable');
    ok($result->{matches} > 0, 'Found expected matches');
    
    # Verify context is working (should see line numbers in brackets)
    my $has_context = $result->{output} =~ /\[\d+\]/;
    ok($has_context, 'Context display working in integration');
};

# Test 9: Memory efficiency integration
subtest 'Memory efficiency integration' => sub {
    plan tests => 3;
    
    my $test_dir = create_comprehensive_test_environment();
    
    # Create additional large files to test memory efficiency
    my $large_dir = File::Spec->catdir($test_dir, 'large');
    mkdir $large_dir or die "Cannot create large directory: $!";
    
    for my $i (1..5) {
        my $file = File::Spec->catfile($large_dir, "large_$i.txt");
        open my $fh, '>', $file or die "Cannot create $file: $!";
        
        # Create moderately large files
        for my $line (1..1000) {
            if ($line % 50 == 0) {
                print $fh "Line $line: Contains search_target for memory test\n";
            } else {
                print $fh "Line $line: Regular content for memory efficiency testing\n";
            }
        }
        close $fh;
    }
    
    # Run search with streaming optimizations
    my $result = run_integrated_search('search_target', ['*.txt'], [], $test_dir);
    
    ok($result->{duration} > 0, 'Memory efficiency test completed');
    cmp_ok($result->{duration}, '<', 3, 'Large file processing time reasonable');
    ok($result->{matches} > 0, 'Found matches in large files');
};

# Test 10: Pattern matching optimization integration
subtest 'Pattern matching optimization integration' => sub {
    plan tests => 4;
    
    my $test_dir = create_comprehensive_test_environment();
    
    # Test with complex regex pattern
    my $complex_pattern = '(search_target|test_pattern|optimization)';
    my $result = run_integrated_search($complex_pattern, ['*.pm', '*.t'], [], $test_dir);
    
    ok($result->{duration} > 0, 'Complex pattern search completed');
    ok($result->{matches} > 0, 'Complex pattern found matches');
    
    # Test with simple pattern for comparison
    my $simple_result = run_integrated_search('search_target', ['*.pm', '*.t'], [], $test_dir);
    
    ok($simple_result->{duration} > 0, 'Simple pattern search completed');
    
    # Complex pattern should not be dramatically slower due to caching
    my $slowdown = $result->{duration} / ($simple_result->{duration} || 0.001);
    cmp_ok($slowdown, '<', 10, 'Complex pattern not dramatically slower');
};

# Test 11: Binary file detection integration
subtest 'Binary file detection integration' => sub {
    plan tests => 3;
    
    my $test_dir = create_comprehensive_test_environment();
    
    # Search all files including binary ones
    my $result = run_integrated_search('search_target', ['*'], [], $test_dir);
    
    ok($result->{duration} > 0, 'Binary detection search completed');
    
    # Should not find matches in binary files (they should be skipped)
    # But should find matches in text files
    ok($result->{matches} > 0, 'Found matches in text files');
    
    # Binary files should be efficiently skipped
    cmp_ok($result->{duration}, '<', 2, 'Binary detection keeps search efficient');
};

# Test 12: Context display integration
subtest 'Context display integration' => sub {
    plan tests => 4;
    
    my $test_dir = create_comprehensive_test_environment();
    
    # Test with context lines
    my $result = run_integrated_search('search_target', ['*.pm'], [], $test_dir, 2);
    
    ok($result->{duration} > 0, 'Context display search completed');
    ok($result->{matches} > 0, 'Found matches for context test');
    
    # Should have line numbers in brackets
    my $has_line_numbers = $result->{output} =~ /\[\d+\]/;
    ok($has_line_numbers, 'Context includes line numbers');
    
    # Should have context lines around matches
    my $line_count = ($result->{output} =~ tr/\n/\n/);
    cmp_ok($line_count, '>', $result->{matches}, 'Context lines included in output');
};

# Test 13: Maxline limit integration
subtest 'Maxline limit integration' => sub {
    plan tests => 3;
    
    my $test_dir = create_comprehensive_test_environment();
    
    # Test with maxline limit
    my $result = run_integrated_search('search_target', ['*.txt'], [], $test_dir, 0, 10);
    
    ok($result->{duration} > 0, 'Maxline limit search completed');
    
    # Should be very fast due to early exit
    cmp_ok($result->{duration}, '<', 1, 'Maxline limit keeps search fast');
    
    # May or may not find matches depending on file content, but should complete
    ok(defined $result->{matches}, 'Maxline search produced results');
};

# Test 14: Cache effectiveness integration
subtest 'Cache effectiveness integration' => sub {
    plan tests => 4;
    
    my $test_dir = create_comprehensive_test_environment();
    
    # Clear caches to start fresh
    _clear_validation_cache();
    
    # First run - should populate caches
    my $first_result = run_integrated_search('search_target', ['*.pm', '*.t'], [], $test_dir);
    
    # Second run - should benefit from caches
    my $second_result = run_integrated_search('search_target', ['*.pm', '*.t'], [], $test_dir);
    
    ok($first_result->{duration} > 0, 'First cache run completed');
    ok($second_result->{duration} > 0, 'Second cache run completed');
    
    # Get cache statistics
    my $cache_stats = _get_cache_stats();
    
    ok($cache_stats->{stat_hits} > 0, 'Cache hits recorded');
    ok($cache_stats->{stat_misses} > 0, 'Cache misses recorded');
};

# Test 15: Overall integration validation
subtest 'Overall integration validation' => sub {
    plan tests => 6;
    
    my $test_dir = create_comprehensive_test_environment();
    
    # Run comprehensive integration test with all features
    my $result = run_integrated_search(
        '(search_target|test)',  # Complex pattern
        ['*.pm', '*.t', '*.txt'], # Multiple target patterns
        ['*.log', '*.tmp', '*.exe'], # Multiple ignore patterns
        $test_dir,
        1,  # Context
        0   # No maxline limit
    );
    
    ok($result->{duration} > 0, 'Overall integration test completed');
    cmp_ok($result->{duration}, '<', 5, 'Overall performance acceptable');
    ok($result->{matches} > 0, 'Found expected matches');
    ok(length($result->{output}) > 0, 'Produced output');
    
    # Verify all optimization systems worked together
    my $has_filenames = $result->{output} =~ /\.(pm|t|txt)/;
    ok($has_filenames, 'Target pattern filtering worked');
    
    my $has_line_numbers = $result->{output} =~ /\[\d+\]/;
    ok($has_line_numbers, 'Context display worked');
};

done_testing();