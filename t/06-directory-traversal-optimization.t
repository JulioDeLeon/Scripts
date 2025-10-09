#!/usr/bin/env perl

use strict;
use warnings;
use Test::More;
use File::Temp qw(tempdir);
use File::Spec;
use Cwd qw(abs_path);

# Add lib directory to path
use lib 'lib';

# Import modules for testing
use GF::Search qw(handle_dir _traverse_directory_optimized _construct_path_optimized _should_skip_fast _quick_extension_check);
use GF::PatternCache qw(initialize_pattern_cache);
use GF::Startup qw(initialize_gf_system);

# Test data
my $test_term = "test_pattern";
my %test_ignores = (
    "*.log" => 1,
    "*.tmp" => 1,
    "node_modules" => 1
);
my %test_targets = (
    "*.pl" => 1,
    "*.pm" => 1,
    "*.t" => 1
);

# Initialize pattern cache for testing
initialize_pattern_cache($test_term, \%test_ignores, \%test_targets);

# Test path construction optimization
subtest 'Path Construction Optimization' => sub {
    plan tests => 4;
    
    # Test basic path construction
    my $result = _construct_path_optimized("/home/user", "file.txt");
    is($result, "/home/user/file.txt", "Basic path construction works");
    
    # Test path normalization (double slashes)
    $result = _construct_path_optimized("/home/user/", "file.txt");
    is($result, "/home/user/file.txt", "Path normalization removes double slashes");
    
    # Test with subdirectory
    $result = _construct_path_optimized("/home/user", "subdir/file.txt");
    like($result, qr{/home/user/subdir/file\.txt$}, "Subdirectory path construction works");
    
    # Test with current directory
    $result = _construct_path_optimized(".", "file.txt");
    like($result, qr{file\.txt$}, "Current directory path construction works");
};

# Test quick extension checking
subtest 'Quick Extension Check Performance' => sub {
    plan tests => 8;
    
    # Test binary extensions that should be skipped
    is(_quick_extension_check("test.exe"), 1, "Executable files are skipped");
    is(_quick_extension_check("lib.dll"), 1, "DLL files are skipped");
    is(_quick_extension_check("image.jpg"), 1, "Image files are skipped");
    is(_quick_extension_check("archive.zip"), 1, "Archive files are skipped");
    
    # Test text extensions that should not be skipped
    is(_quick_extension_check("script.pl"), 0, "Perl files are not skipped");
    is(_quick_extension_check("module.pm"), 0, "Perl modules are not skipped");
    is(_quick_extension_check("readme.txt"), 0, "Text files are not skipped");
    is(_quick_extension_check("config.conf"), 0, "Config files are not skipped");
};

# Test fast skip functionality
subtest 'Fast Skip Pattern Matching' => sub {
    plan tests => 6;
    
    # Create temporary test files for realistic testing
    my $temp_dir = tempdir(CLEANUP => 1);
    
    # Test files that should be skipped due to ignore patterns
    my $log_file = File::Spec->catfile($temp_dir, "test.log");
    my $tmp_file = File::Spec->catfile($temp_dir, "temp.tmp");
    my $node_modules = File::Spec->catfile($temp_dir, "node_modules");
    
    # Create test files
    open(my $fh, '>', $log_file) or die "Cannot create test file: $!";
    print $fh "test content\n";
    close($fh);
    
    open($fh, '>', $tmp_file) or die "Cannot create test file: $!";
    print $fh "test content\n";
    close($fh);
    
    mkdir($node_modules) or die "Cannot create test directory: $!";
    
    # Test ignore pattern matching
    is(_should_skip_fast($log_file, \%test_ignores, \%test_targets, 0), 1, 
       "Log files are skipped due to ignore pattern");
    is(_should_skip_fast($tmp_file, \%test_ignores, \%test_targets, 0), 1, 
       "Temp files are skipped due to ignore pattern");
    is(_should_skip_fast($node_modules, \%test_ignores, \%test_targets, 0), 1, 
       "node_modules directory is skipped due to ignore pattern");
    
    # Test files that should not be skipped
    my $perl_file = File::Spec->catfile($temp_dir, "test.pl");
    open($fh, '>', $perl_file) or die "Cannot create test file: $!";
    print $fh "test content\n";
    close($fh);
    
    is(_should_skip_fast($perl_file, \%test_ignores, \%test_targets, 0), 0, 
       "Perl files are not skipped when matching target pattern");
    
    # Test file that doesn't match target patterns
    my $other_file = File::Spec->catfile($temp_dir, "test.py");  # Use .py which won't match *.pl, *.pm, *.t
    open($fh, '>', $other_file) or die "Cannot create test file: $!";
    print $fh "test content\n";
    close($fh);
    
    is(_should_skip_fast($other_file, \%test_ignores, \%test_targets, 0), 1, 
       "Files not matching target patterns are skipped");
    
    # Test with no target patterns (should not skip) - use a .txt file to avoid extension skip
    my $txt_file = File::Spec->catfile($temp_dir, "test.txt");
    open($fh, '>', $txt_file) or die "Cannot create test file: $!";
    print $fh "test content\n";
    close($fh);
    
    my %empty_targets = ();
    is(_should_skip_fast($txt_file, \%test_ignores, \%empty_targets, 0), 0, 
       "Files are not skipped when no target patterns are defined");
};

# Test directory traversal performance and correctness
subtest 'Directory Traversal Optimization' => sub {
    plan tests => 5;
    
    # Create a test directory structure
    my $temp_dir = tempdir(CLEANUP => 1);
    my $subdir = File::Spec->catdir($temp_dir, "subdir");
    mkdir($subdir) or die "Cannot create subdirectory: $!";
    
    # Create test files
    my $perl_file = File::Spec->catfile($temp_dir, "test.pl");
    my $log_file = File::Spec->catfile($temp_dir, "test.log");
    my $sub_perl_file = File::Spec->catfile($subdir, "sub.pl");
    
    open(my $fh, '>', $perl_file) or die "Cannot create test file: $!";
    print $fh "my \$test_pattern = 'found';\n";
    close($fh);
    
    open($fh, '>', $log_file) or die "Cannot create test file: $!";
    print $fh "test_pattern in log file\n";
    close($fh);
    
    open($fh, '>', $sub_perl_file) or die "Cannot create test file: $!";
    print $fh "another test_pattern here\n";
    close($fh);
    
    # Test that traversal works without chdir operations
    my $original_cwd = Cwd::getcwd();
    
    # Capture output to verify search results
    my $output = '';
    {
        local *STDOUT;
        open(STDOUT, '>', \$output) or die "Cannot redirect STDOUT: $!";
        
        # Initialize system for this test
        eval {
            initialize_gf_system($test_term, \%test_ignores, \%test_targets, 0, 0, 0);
        };
        
        # Run directory traversal
        _traverse_directory_optimized($temp_dir, $test_term, \%test_ignores, \%test_targets, {}, 0, 0, 0);
        
        # Flush any buffered output
        require GF::Search;
        GF::Search::flush_output_buffer();
    }
    
    # Verify we're still in the same directory (no chdir was used)
    is(Cwd::getcwd(), $original_cwd, "Current directory unchanged after traversal");
    
    # Verify that appropriate files were processed
    like($output, qr/test\.pl/, "Perl file in root directory was processed");
    like($output, qr/sub\.pl/, "Perl file in subdirectory was processed");
    unlike($output, qr/test\.log/, "Log file was skipped due to ignore pattern");
    
    # Verify that the traversal completed without errors
    ok(1, "Directory traversal completed successfully");
};

# Test pattern evaluation order for performance
subtest 'Pattern Evaluation Order Optimization' => sub {
    plan tests => 4;
    
    # Test that ignore patterns are checked before target patterns
    # This is important for performance as ignore patterns are more likely to match
    
    my $temp_dir = tempdir(CLEANUP => 1);
    my $test_file = File::Spec->catfile($temp_dir, "test.log");
    
    open(my $fh, '>', $test_file) or die "Cannot create test file: $!";
    print $fh "test content\n";
    close($fh);
    
    # File matches both ignore pattern (*.log) and could match target pattern
    # But ignore should take precedence for fast failure
    my %mixed_targets = ("*.log" => 1, "*.pl" => 1);  # Include .log in targets too
    
    is(_should_skip_fast($test_file, \%test_ignores, \%mixed_targets, 0), 1, 
       "Ignore patterns take precedence over target patterns");
    
    # Test with a file that only matches target pattern
    my $perl_file = File::Spec->catfile($temp_dir, "test.pl");
    open($fh, '>', $perl_file) or die "Cannot create test file: $!";
    print $fh "test content\n";
    close($fh);
    
    is(_should_skip_fast($perl_file, \%test_ignores, \%test_targets, 0), 0, 
       "Files matching target patterns are not skipped");
    
    # Test extension check happens before pattern matching
    my $exe_file = File::Spec->catfile($temp_dir, "test.exe");
    open($fh, '>', $exe_file) or die "Cannot create test file: $!";
    print $fh "binary content\n";
    close($fh);
    
    is(_should_skip_fast($exe_file, {}, {}, 0), 1, 
       "Extension check skips files before pattern matching");
    
    # Test that non-existent files are handled gracefully
    my $nonexistent = File::Spec->catfile($temp_dir, "nonexistent.pl");
    is(_should_skip_fast($nonexistent, \%test_ignores, \%test_targets, 0), 1, 
       "Non-existent files are skipped");
};

done_testing();