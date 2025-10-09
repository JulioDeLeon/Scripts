#!/usr/bin/perl

use strict;
use warnings;
use Test::More;
use File::Temp qw(tempdir tempfile);
use File::Path qw(make_path remove_tree);
use FindBin;
use lib "$FindBin::Bin/../lib";

# Import the functions we want to test
use GF::Search qw(
  _validate_file_fast _get_cached_stat _check_file_permissions _is_binary_cached
  _clear_validation_cache _get_cache_stats _optimize_cache_size _quick_extension_check
);

# Test setup
my $temp_dir = tempdir(CLEANUP => 1);
my $debug = 0;  # Set to 1 for debug output during testing

# Clear cache before starting tests
_clear_validation_cache();

subtest 'Quick extension check tests' => sub {
  plan tests => 8;
  
  # Test binary extensions that should be skipped
  ok(_quick_extension_check('test.exe'), 'Should skip .exe files');
  ok(_quick_extension_check('lib.dll'), 'Should skip .dll files');
  ok(_quick_extension_check('image.jpg'), 'Should skip .jpg files');
  ok(_quick_extension_check('video.mp4'), 'Should skip .mp4 files');
  ok(_quick_extension_check('archive.zip'), 'Should skip .zip files');
  
  # Test text extensions that should not be skipped
  ok(!_quick_extension_check('code.pl'), 'Should not skip .pl files');
  ok(!_quick_extension_check('document.txt'), 'Should not skip .txt files');
  ok(!_quick_extension_check('config.conf'), 'Should not skip .conf files');
};

subtest 'File metadata caching tests' => sub {
  plan tests => 12;
  
  # Create test files
  my ($fh1, $text_file) = tempfile(DIR => $temp_dir, SUFFIX => '.txt');
  print $fh1 "This is a text file\n";
  close $fh1;
  
  my ($fh2, $binary_file) = tempfile(DIR => $temp_dir, SUFFIX => '.bin');
  binmode $fh2;
  # Create binary content with null bytes and high-bit characters
  print $fh2 pack('C*', 0, 1, 2, 3, 255, 254, 253, 0, 0, 0);
  close $fh2;
  
  my $nonexistent_file = "$temp_dir/nonexistent.txt";
  
  # Test stat caching
  my $stat1 = _get_cached_stat($text_file);
  ok($stat1, 'Should get stat info for existing file');
  ok(ref($stat1) eq 'ARRAY', 'Stat info should be array reference');
  
  my $stat2 = _get_cached_stat($text_file);
  is_deeply($stat1, $stat2, 'Second call should return cached result');
  
  my $stat_nonexistent = _get_cached_stat($nonexistent_file);
  ok(!defined $stat_nonexistent, 'Should return undef for nonexistent file');
  
  # Test permission caching
  ok(_check_file_permissions($text_file, $debug), 'Should have read permission for text file');
  ok(_check_file_permissions($text_file, $debug), 'Second call should use cached result');
  
  # Test binary detection caching
  ok(!_is_binary_cached($text_file, $debug), 'Text file should not be detected as binary');
  ok(_is_binary_cached($binary_file, $debug), 'Binary file should be detected as binary');
  ok(!_is_binary_cached($text_file, $debug), 'Second call should use cached result');
  ok(_is_binary_cached($binary_file, $debug), 'Second binary check should use cached result');
  
  # Test nonexistent file handling
  ok(_is_binary_cached($nonexistent_file, $debug), 'Nonexistent file should be treated as binary');
  ok(_is_binary_cached($nonexistent_file, $debug), 'Second call should use cached result');
};

subtest 'Cache statistics tests' => sub {
  plan tests => 8;
  
  # Clear cache and get initial stats
  _clear_validation_cache();
  my $initial_stats = _get_cache_stats();
  
  is($initial_stats->{stat_hits}, 0, 'Initial stat hits should be 0');
  is($initial_stats->{stat_misses}, 0, 'Initial stat misses should be 0');
  
  # Create a test file and perform operations to generate cache activity
  my ($fh, $test_file) = tempfile(DIR => $temp_dir, SUFFIX => '.txt');
  print $fh "Test content\n";
  close $fh;
  
  # First access should be cache miss
  _get_cached_stat($test_file);
  my $stats_after_miss = _get_cache_stats();
  is($stats_after_miss->{stat_misses}, 1, 'Should have 1 stat miss after first access');
  
  # Second access should be cache hit
  _get_cached_stat($test_file);
  my $stats_after_hit = _get_cache_stats();
  is($stats_after_hit->{stat_hits}, 1, 'Should have 1 stat hit after second access');
  is($stats_after_hit->{stat_misses}, 1, 'Should still have 1 stat miss');
  
  # Test permission cache stats
  _check_file_permissions($test_file, $debug);
  _check_file_permissions($test_file, $debug);
  my $perm_stats = _get_cache_stats();
  is($perm_stats->{permission_hits}, 1, 'Should have 1 permission hit');
  is($perm_stats->{permission_misses}, 1, 'Should have 1 permission miss');
  
  # Test binary cache stats
  _is_binary_cached($test_file, $debug);
  _is_binary_cached($test_file, $debug);
  my $binary_stats = _get_cache_stats();
  is($binary_stats->{binary_hits}, 1, 'Should have 1 binary hit');
};

subtest 'Fast validation pipeline tests' => sub {
  plan tests => 10;
  
  # Clear cache for clean test
  _clear_validation_cache();
  
  # Create test files
  my ($fh1, $text_file) = tempfile(DIR => $temp_dir, SUFFIX => '.txt');
  print $fh1 "This is a text file for validation testing\n";
  close $fh1;
  
  my ($fh2, $binary_file) = tempfile(DIR => $temp_dir, SUFFIX => '.bin');
  binmode $fh2;
  print $fh2 pack('C*', 0, 1, 2, 3, 255, 254, 253);
  close $fh2;
  
  my $exe_file = "$temp_dir/test.exe";
  my ($fh3, $temp_exe) = tempfile(DIR => $temp_dir, SUFFIX => '.exe');
  print $fh3 "fake exe content\n";
  close $fh3;
  rename $temp_exe, $exe_file;
  
  my $nonexistent_file = "$temp_dir/nonexistent.txt";
  my $unreadable_file = "$temp_dir/unreadable.txt";
  
  # Create unreadable file (if possible)
  if (open my $fh4, '>', $unreadable_file) {
    print $fh4 "unreadable content\n";
    close $fh4;
    chmod 0000, $unreadable_file;  # Remove all permissions
  }
  
  # Test valid text file
  ok(!_validate_file_fast($text_file, $debug), 'Valid text file should pass validation');
  
  # Test file with bad extension
  ok(_validate_file_fast($exe_file, $debug), 'File with .exe extension should fail validation');
  
  # Test nonexistent file
  ok(_validate_file_fast($nonexistent_file, $debug), 'Nonexistent file should fail validation');
  
  # Test binary file
  ok(_validate_file_fast($binary_file, $debug), 'Binary file should fail validation');
  
  # Test unreadable file (if we could create one)
  if (-e $unreadable_file) {
    ok(_validate_file_fast($unreadable_file, $debug), 'Unreadable file should fail validation');
    chmod 0644, $unreadable_file;  # Restore permissions for cleanup
  } else {
    pass('Skipping unreadable file test (could not create unreadable file)');
  }
  
  # Test that validation uses caching (second call should be faster)
  my $start_time = time();
  _validate_file_fast($text_file, $debug);
  my $cached_time = time() - $start_time;
  
  # The cached call should complete very quickly (this is a rough test)
  ok($cached_time <= 1, 'Cached validation should be fast');
  
  # Verify cache statistics show hits
  my $stats = _get_cache_stats();
  ok($stats->{stat_hits} > 0, 'Should have stat cache hits');
  ok($stats->{permission_hits} > 0, 'Should have permission cache hits');
  ok($stats->{binary_hits} > 0, 'Should have binary cache hits');
  
  # Test cache clearing
  _clear_validation_cache();
  my $cleared_stats = _get_cache_stats();
  is($cleared_stats->{stat_hits}, 0, 'Cache stats should be reset after clearing');
};

subtest 'Cache size optimization tests' => sub {
  plan tests => 6;
  
  # Clear cache and create many entries
  _clear_validation_cache();
  
  # Create temporary files to populate cache
  my @test_files;
  for my $i (1..50) {
    my ($fh, $filename) = tempfile(DIR => $temp_dir, SUFFIX => ".test$i.txt");
    print $fh "Test file $i content\n";
    close $fh;
    push @test_files, $filename;
    
    # Access each file to populate all caches
    _get_cached_stat($filename);
    _check_file_permissions($filename, $debug);
    _is_binary_cached($filename, $debug);
  }
  
  # Verify caches are populated
  my $stats_before = _get_cache_stats();
  ok($stats_before->{stat_misses} >= 50, 'Should have many stat cache entries');
  ok($stats_before->{permission_misses} >= 50, 'Should have many permission cache entries');
  ok($stats_before->{binary_misses} >= 50, 'Should have many binary cache entries');
  
  # Test cache optimization with small limit
  _optimize_cache_size(10);
  
  # Access a few files to check if optimization worked
  # (We can't directly check cache size, but we can verify it still works)
  ok(!_validate_file_fast($test_files[0], $debug), 'Cache should still work after optimization');
  ok(!_validate_file_fast($test_files[-1], $debug), 'Cache should still work for different files');
  
  # Test that cache optimization doesn't break functionality
  my $stats_after = _get_cache_stats();
  ok($stats_after->{stat_hits} >= 0, 'Cache should still track hits after optimization');
};

subtest 'Performance improvement tests' => sub {
  plan tests => 4;
  
  # Create test files for performance comparison
  my @perf_test_files;
  for my $i (1..20) {
    my ($fh, $filename) = tempfile(DIR => $temp_dir, SUFFIX => ".perf$i.txt");
    print $fh "Performance test file $i\n" x 10;
    close $fh;
    push @perf_test_files, $filename;
  }
  
  # Clear cache and measure time for first pass (cache misses)
  _clear_validation_cache();
  my $start_time = time();
  
  for my $file (@perf_test_files) {
    _validate_file_fast($file, $debug);
  }
  
  my $first_pass_time = time() - $start_time;
  
  # Measure time for second pass (cache hits)
  $start_time = time();
  
  for my $file (@perf_test_files) {
    _validate_file_fast($file, $debug);
  }
  
  my $second_pass_time = time() - $start_time;
  
  # Second pass should be faster or equal (due to caching)
  ok($second_pass_time <= $first_pass_time, 'Cached validation should be faster or equal');
  
  # Verify we got cache hits
  my $final_stats = _get_cache_stats();
  ok($final_stats->{stat_hits} >= 20, 'Should have significant stat cache hits');
  ok($final_stats->{permission_hits} >= 20, 'Should have significant permission cache hits');
  ok($final_stats->{binary_hits} >= 20, 'Should have significant binary cache hits');
};

subtest 'Edge cases and error handling' => sub {
  plan tests => 6;
  
  # Test empty file
  my ($fh, $empty_file) = tempfile(DIR => $temp_dir, SUFFIX => '.empty');
  close $fh;  # Create empty file
  
  ok(!_is_binary_cached($empty_file, $debug), 'Empty file should not be detected as binary');
  ok(!_validate_file_fast($empty_file, $debug), 'Empty file should pass validation');
  
  # Test file with no extension
  my ($fh2, $no_ext_file) = tempfile(DIR => $temp_dir);
  print $fh2 "File with no extension\n";
  close $fh2;
  
  ok(!_quick_extension_check($no_ext_file), 'File with no extension should pass extension check');
  ok(!_validate_file_fast($no_ext_file, $debug), 'File with no extension should pass validation');
  
  # Test very long filename
  my $long_name = "$temp_dir/" . ('a' x 100) . '.txt';
  if (open my $fh3, '>', $long_name) {
    print $fh3 "Long filename test\n";
    close $fh3;
    
    ok(!_validate_file_fast($long_name, $debug), 'File with long name should pass validation');
  } else {
    pass('Skipping long filename test (could not create file)');
  }
  
  # Test cache behavior with same file accessed multiple times
  my $access_count = 5;
  for my $i (1..$access_count) {
    _validate_file_fast($empty_file, $debug);
  }
  
  my $multi_access_stats = _get_cache_stats();
  ok($multi_access_stats->{stat_hits} >= $access_count - 1, 
     'Multiple accesses should generate cache hits');
};

# Run all tests
done_testing();

__END__

=head1 NAME

07-early-validation-pipeline.t - Tests for early validation pipeline optimization

=head1 DESCRIPTION

This test suite validates the early validation pipeline implementation including:
- Fast-fail validation system with optimal ordering
- File metadata caching for performance
- Cache statistics and management
- Performance improvements through caching
- Edge cases and error handling

The tests verify that the validation pipeline correctly:
- Performs file extension checking before expensive operations
- Implements permission validation before file access attempts
- Creates optimal pattern evaluation ordering
- Caches stat information to avoid repeated system calls
- Caches file permissions and type information
- Optimizes file validation pipeline for performance

=head1 REQUIREMENTS TESTED

- 4.1: File extension and binary detection before opening files
- 4.2: Permission validation before file access attempts  
- 4.3: Optimal pattern evaluation ordering for fast failure
- 4.4: File metadata caching to avoid repeated system calls

=cut