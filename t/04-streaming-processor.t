#!/usr/bin/env perl
use strict;
use warnings;
use Test::More tests => 12;
use File::Temp qw(tempdir tempfile);
use File::Path qw(make_path);
use lib 'lib';
use lib 't';

require 'test_utils.pl';

BEGIN {
    use_ok('GF::Search', qw(check_file is_binary_file));
    use_ok('GF::PatternCache', qw(initialize_pattern_cache));
}

# Test binary file detection
subtest 'binary file detection' => sub {
    plan tests => 4;
    
    my $temp_dir = tempdir(CLEANUP => 1);
    
    # Create a text file
    my $text_file = "$temp_dir/test.txt";
    write_test_file($text_file, "This is a text file\nwith multiple lines\n");
    
    # Create a binary file (simulate with null bytes)
    my $binary_file = "$temp_dir/test.bin";
    open my $fh, '>', $binary_file or die "Cannot create binary file: $!";
    binmode $fh;
    print $fh "\x00\x01\x02\x03\xFF\xFE";
    close $fh;
    
    # Test binary detection
    is(GF::Search::is_binary_file($text_file), 0, 'Text file detected as non-binary');
    is(GF::Search::is_binary_file($binary_file), 1, 'Binary file detected correctly');
    is(GF::Search::is_binary_file('/nonexistent/file'), 0, 'Non-existent file returns 0');
    
    # Test with empty file
    my $empty_file = "$temp_dir/empty.txt";
    write_test_file($empty_file, "");
    is(GF::Search::is_binary_file($empty_file), 0, 'Empty file detected as non-binary');
};

# Test context buffer management
subtest 'context buffer management' => sub {
    plan tests => 6;
    
    # Test the internal context buffer function through check_file
    my $temp_dir = tempdir(CLEANUP => 1);
    my $test_file = "$temp_dir/context_test.txt";
    
    # Create a file with numbered lines for context testing
    my $content = join("\n", map { "Line $_" } (1..10)) . "\n";
    write_test_file($test_file, $content);
    
    # Initialize pattern cache for "Line 5"
    GF::PatternCache::initialize_pattern_cache("Line 5", {}, {});
    
    # Capture output from check_file with context
    my $output = capture_output(sub {
        GF::Search::check_file($test_file, "Line 5", 2, 0, 0);
    });
    
    # Verify context is included
    like($output, qr/Line 3/, 'Context before match included');
    like($output, qr/Line 4/, 'Context before match included');
    like($output, qr/Line 5/, 'Match line included');
    like($output, qr/\[5\]/, 'Line number displayed correctly');
    
    # Test without context
    GF::PatternCache::initialize_pattern_cache("Line 7", {}, {});
    my $output_no_context = capture_output(sub {
        GF::Search::check_file($test_file, "Line 7", 0, 0, 0);
    });
    
    unlike($output_no_context, qr/\[6\]\s+Line 6/, 'No context lines when context=0');
    like($output_no_context, qr/\[7\]\s+Line 7/, 'Match line still included');
};

# Test early exit with maxline limits
subtest 'early exit with maxline limits' => sub {
    plan tests => 4;
    
    my $temp_dir = tempdir(CLEANUP => 1);
    my $test_file = "$temp_dir/maxline_test.txt";
    
    # Create a file with 100 lines
    my $content = join("\n", map { "Content line $_" } (1..100)) . "\n";
    write_test_file($test_file, $content);
    
    # Initialize pattern cache for "Content"
    GF::PatternCache::initialize_pattern_cache("Content", {}, {});
    
    # Test with maxline limit of 10
    my $output_limited = capture_output(sub {
        GF::Search::check_file($test_file, "Content", 0, 10, 0);
    });
    
    # Should find matches in first 10 lines only
    like($output_limited, qr/\[1\]\s+Content line 1/, 'First line match found');
    like($output_limited, qr/\[10\]\s+Content line 10/, 'Tenth line match found');
    unlike($output_limited, qr/Content line 11/, 'Eleventh line not processed');
    unlike($output_limited, qr/Content line 50/, 'Later lines not processed');
};

# Test streaming vs memory usage (conceptual test)
subtest 'streaming file processing behavior' => sub {
    plan tests => 3;
    
    my $temp_dir = tempdir(CLEANUP => 1);
    my $large_file = "$temp_dir/large_test.txt";
    
    # Create a moderately large file (1000 lines)
    my @lines = map { "Test line $_ with some content to search" } (1..1000);
    $lines[500] = "SPECIAL_MARKER line 501 with unique content";
    my $content = join("\n", @lines) . "\n";
    write_test_file($large_file, $content);
    
    # Initialize pattern cache
    GF::PatternCache::initialize_pattern_cache("SPECIAL_MARKER", {}, {});
    
    # Test that streaming processing works
    my $output = capture_output(sub {
        GF::Search::check_file($large_file, "SPECIAL_MARKER", 0, 0, 0);
    });
    
    like($output, qr/SPECIAL_MARKER/, 'Found match in large file');
    like($output, qr/\[501\]/, 'Correct line number displayed');
    like($output, qr/large_test\.txt/, 'Filename displayed');
};

# Test binary file early detection integration
subtest 'binary file early detection integration' => sub {
    plan tests => 3;
    
    my $temp_dir = tempdir(CLEANUP => 1);
    
    # Create a binary file
    my $binary_file = "$temp_dir/test.bin";
    open my $fh, '>', $binary_file or die "Cannot create binary file: $!";
    binmode $fh;
    print $fh "\x00\x01\x02SEARCH_TERM\x03\xFF\xFE";
    close $fh;
    
    # Initialize pattern cache
    GF::PatternCache::initialize_pattern_cache("SEARCH_TERM", {}, {});
    
    # Test that binary file is skipped
    my $output = capture_output(sub {
        GF::Search::check_file($binary_file, "SEARCH_TERM", 0, 0, 0);
    });
    
    is($output, '', 'Binary file produces no output');
    
    # Test with debug mode to verify skipping message
    my $debug_output = capture_output(sub {
        GF::Search::check_file($binary_file, "SEARCH_TERM", 0, 0, 1);
    });
    
    like($debug_output, qr/skipping binary file/, 'Debug message shows binary file skipped');
    unlike($debug_output, qr/SEARCH_TERM/, 'Search term not found in binary file output');
};

# Test file handle error handling
subtest 'file handle error handling' => sub {
    plan tests => 2;
    
    # Test with non-existent file
    my $stderr_output = capture_stderr(sub {
        GF::Search::check_file('/nonexistent/file.txt', 'test', 0, 0, 0);
    });
    
    like($stderr_output, qr/Warning: Could not open file/, 'Error message for non-existent file');
    like($stderr_output, qr/nonexistent\/file\.txt/, 'Filename included in error message');
};

# Test multiple matches in single file
subtest 'multiple matches in single file' => sub {
    plan tests => 4;
    
    my $temp_dir = tempdir(CLEANUP => 1);
    my $test_file = "$temp_dir/multi_match.txt";
    
    my $content = <<'EOF';
function first_function() {
    return "hello";
}

function second_function() {
    return "world";
}

function third_function() {
    return "test";
}
EOF
    
    write_test_file($test_file, $content);
    
    # Initialize pattern cache for "function"
    GF::PatternCache::initialize_pattern_cache("function", {}, {});
    
    my $output = capture_output(sub {
        GF::Search::check_file($test_file, "function", 0, 0, 0);
    });
    
    # Should find all three function declarations
    like($output, qr/\[1\]\s+function first_function/, 'First function found');
    like($output, qr/\[5\]\s+function second_function/, 'Second function found');
    like($output, qr/\[9\]\s+function third_function/, 'Third function found');
    
    # Filename should only appear once
    my @filename_matches = $output =~ /multi_match\.txt/g;
    is(scalar @filename_matches, 1, 'Filename header appears only once');
};

# Test context with matches near file boundaries
subtest 'context with boundary conditions' => sub {
    plan tests => 4;
    
    my $temp_dir = tempdir(CLEANUP => 1);
    my $test_file = "$temp_dir/boundary_test.txt";
    
    my $content = <<'EOF';
Line 1
MATCH at line 2
Line 3
Line 4
Line 5
MATCH at line 6
Line 7
EOF
    
    write_test_file($test_file, $content);
    
    # Initialize pattern cache for "MATCH"
    GF::PatternCache::initialize_pattern_cache("MATCH", {}, {});
    
    my $output = capture_output(sub {
        GF::Search::check_file($test_file, "MATCH", 1, 0, 0);
    });
    
    # Check context around first match (line 2)
    like($output, qr/\[1\]\s+Line 1/, 'Context before first match');
    like($output, qr/\[2\]\s+MATCH at line 2/, 'First match line');
    
    # Check context around second match (line 6)
    like($output, qr/\[5\]\s+Line 5/, 'Context before second match');
    like($output, qr/\[6\]\s+MATCH at line 6/, 'Second match line');
};

# Test empty file handling
subtest 'empty file handling' => sub {
    plan tests => 2;
    
    my $temp_dir = tempdir(CLEANUP => 1);
    my $empty_file = "$temp_dir/empty.txt";
    write_test_file($empty_file, "");
    
    # Initialize pattern cache
    GF::PatternCache::initialize_pattern_cache("anything", {}, {});
    
    my $output = capture_output(sub {
        GF::Search::check_file($empty_file, "anything", 0, 0, 0);
    });
    
    is($output, '', 'Empty file produces no output');
    
    # Test that empty file is not detected as binary
    is(GF::Search::is_binary_file($empty_file), 0, 'Empty file not detected as binary');
};

# Test maxline with context interaction
subtest 'maxline with context interaction' => sub {
    plan tests => 3;
    
    my $temp_dir = tempdir(CLEANUP => 1);
    my $test_file = "$temp_dir/maxline_context.txt";
    
    # Create file with match near maxline limit
    my $content = join("\n", map { "Line $_" } (1..20)) . "\n";
    write_test_file($test_file, $content);
    
    # Initialize pattern cache for "Line 8"
    GF::PatternCache::initialize_pattern_cache("Line 8", {}, {});
    
    # Test with maxline=10 and context=2
    my $output = capture_output(sub {
        GF::Search::check_file($test_file, "Line 8", 2, 10, 0);
    });
    
    like($output, qr/Line 6/, 'Context before match within maxline');
    like($output, qr/Line 8/, 'Match line found');
    
    # Test that lines beyond maxline are not processed
    GF::PatternCache::initialize_pattern_cache("Line 15", {}, {});
    my $output_beyond = capture_output(sub {
        GF::Search::check_file($test_file, "Line 15", 0, 10, 0);
    });
    
    is($output_beyond, '', 'Lines beyond maxline not processed');
};

done_testing();