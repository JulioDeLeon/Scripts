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

# Test context display formatting and edge cases
# Tests various context values (1, 3, 5, 10), binary file detection,
# permission errors, empty files, and single-line files

# Find the gf binary
my $gf_bin = File::Spec->catfile('bin', 'gf');
$gf_bin = abs_path($gf_bin);
if (!-x $gf_bin) {
    plan skip_all => "gf binary not found or not executable at $gf_bin";
}

# Create temporary test directory and files
my $test_dir = tempdir(CLEANUP => 1);

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

# Helper function to create test file with specific content
sub create_test_file {
    my ($filename, $content) = @_;
    my $full_path = File::Spec->catfile($test_dir, $filename);
    open my $fh, '>', $full_path or die "Cannot create $full_path: $!";
    print $fh $content;
    close $fh;
    return $full_path;
}

plan tests => 55;

# Test 1: Context value 1 - basic formatting
{
    my $test_file = create_test_file('context1.txt', <<'EOF');
line 1: before
line 2: target match
line 3: after
line 4: more content
EOF
    
    chdir $test_dir or die "Cannot chdir to $test_dir: $!";
    my $result = run_gf_command('--search', 'target', '--context', '1');
    ok($result->{exit_code} == 0, "Context 1 exits with code 0");
    
    # Check line number formatting
    like($result->{output}, qr/\[1\].*before/, "Context 1 shows line 1 with correct format");
    like($result->{output}, qr/\[2\].*target match/, "Context 1 shows line 2 with correct format");
    like($result->{output}, qr/\[3\].*after/, "Context 1 shows line 3 with correct format");
    unlike($result->{output}, qr/\[4\].*more content/, "Context 1 does not show line 4");
}

# Test 2: Context value 3 - extended formatting
{
    my $test_file = create_test_file('context3.txt', <<'EOF');
line 1: context before
line 2: more before
line 3: just before
line 4: target match here
line 5: just after
line 6: more after
line 7: context after
line 8: beyond context
EOF
    
    chdir $test_dir or die "Cannot chdir to $test_dir: $!";
    my $result = run_gf_command('--search', 'target', '--context', '3');
    ok($result->{exit_code} == 0, "Context 3 exits with code 0");
    
    # Check all context lines are shown
    like($result->{output}, qr/\[1\].*context before/, "Context 3 shows line 1");
    like($result->{output}, qr/\[2\].*more before/, "Context 3 shows line 2");
    like($result->{output}, qr/\[3\].*just before/, "Context 3 shows line 3");
    like($result->{output}, qr/\[4\].*target match here/, "Context 3 shows matching line 4");
    like($result->{output}, qr/\[5\].*just after/, "Context 3 shows line 5");
    like($result->{output}, qr/\[6\].*more after/, "Context 3 shows line 6");
    like($result->{output}, qr/\[7\].*context after/, "Context 3 shows line 7");
    unlike($result->{output}, qr/\[8\].*beyond context/, "Context 3 does not show line 8");
}

# Test 3: Context value 5 - larger context
{
    my $test_file = create_test_file('context5.txt', <<'EOF');
line 1: far before
line 2: before 4
line 3: before 3
line 4: before 2
line 5: before 1
line 6: target match
line 7: after 1
line 8: after 2
line 9: after 3
line 10: after 4
line 11: far after
line 12: beyond
EOF
    
    chdir $test_dir or die "Cannot chdir to $test_dir: $!";
    my $result = run_gf_command('--search', 'target', '--context', '5');
    ok($result->{exit_code} == 0, "Context 5 exits with code 0");
    
    # Check all 5 before and 5 after lines
    like($result->{output}, qr/\[1\].*far before/, "Context 5 shows line 1");
    like($result->{output}, qr/\[2\].*before 4/, "Context 5 shows line 2");
    like($result->{output}, qr/\[3\].*before 3/, "Context 5 shows line 3");
    like($result->{output}, qr/\[4\].*before 2/, "Context 5 shows line 4");
    like($result->{output}, qr/\[5\].*before 1/, "Context 5 shows line 5");
    like($result->{output}, qr/\[6\].*target match/, "Context 5 shows matching line 6");
    like($result->{output}, qr/\[7\].*after 1/, "Context 5 shows line 7");
    like($result->{output}, qr/\[8\].*after 2/, "Context 5 shows line 8");
    like($result->{output}, qr/\[9\].*after 3/, "Context 5 shows line 9");
    like($result->{output}, qr/\[10\].*after 4/, "Context 5 shows line 10");
    like($result->{output}, qr/\[11\].*far after/, "Context 5 shows line 11");
    unlike($result->{output}, qr/\[12\].*beyond/, "Context 5 does not show line 12");
}

# Test 4: Context value 10 - very large context
{
    my $content = '';
    for my $i (1..25) {
        if ($i == 13) {
            $content .= "line $i: target match in middle\n";
        } else {
            $content .= "line $i: content line $i\n";
        }
    }
    my $test_file = create_test_file('context10.txt', $content);
    
    chdir $test_dir or die "Cannot chdir to $test_dir: $!";
    my $result = run_gf_command('--search', 'target', '--context', '10');
    ok($result->{exit_code} == 0, "Context 10 exits with code 0");
    
    # Check 10 lines before and after (lines 3-23)
    like($result->{output}, qr/\[3\].*content line 3/, "Context 10 shows line 3");
    like($result->{output}, qr/\[12\].*content line 12/, "Context 10 shows line 12");
    like($result->{output}, qr/\[13\].*target match in middle/, "Context 10 shows matching line 13");
    like($result->{output}, qr/\[14\].*content line 14/, "Context 10 shows line 14");
    like($result->{output}, qr/\[23\].*content line 23/, "Context 10 shows line 23");
    unlike($result->{output}, qr/\[1\].*content line 1/, "Context 10 does not show line 1");
    unlike($result->{output}, qr/\[2\].*content line 2/, "Context 10 does not show line 2");
    unlike($result->{output}, qr/\[24\].*content line 24/, "Context 10 does not show line 24");
    unlike($result->{output}, qr/\[25\].*content line 25/, "Context 10 does not show line 25");
}

# Test 5: Binary file detection with context
{
    my $binary_file = File::Spec->catfile($test_dir, 'binary.bin');
    open my $fh_bin, '>', $binary_file or die "Cannot create $binary_file: $!";
    binmode $fh_bin;
    # Create binary content with some text that would match
    print $fh_bin "text before\x00\x01\x02target\x03\x04\x05text after\n";
    close $fh_bin;
    
    chdir $test_dir or die "Cannot chdir to $test_dir: $!";
    my $result = run_gf_command('--search', 'target', '--context', '2');
    ok($result->{exit_code} == 0, "Context with binary file exits with code 0");
    
    # Binary files should be detected and skipped
    unlike($result->{output}, qr/binary\.bin/, "Context skips binary files");
}

# Test 6: Empty file with context
{
    my $empty_file = create_test_file('empty.txt', '');
    
    chdir $test_dir or die "Cannot chdir to $test_dir: $!";
    my $result = run_gf_command('--search', 'target', '--context', '3');
    ok($result->{exit_code} == 0, "Context with empty file exits with code 0");
    
    # Empty file should not produce any output
    unlike($result->{output}, qr/empty\.txt/, "Context handles empty files gracefully");
}

# Test 7: Single-line file with context
{
    my $single_line_file = create_test_file('single.txt', 'single line with target match');
    
    chdir $test_dir or die "Cannot chdir to $test_dir: $!";
    my $result = run_gf_command('--search', 'target', '--context', '3');
    ok($result->{exit_code} == 0, "Context with single-line file exits with code 0");
    
    # Should show the single line without context
    like($result->{output}, qr/single\.txt/, "Context finds single-line file");
    like($result->{output}, qr/\[1\].*single line with target match/, "Context shows single line correctly");
    # No additional context lines should be shown
    unlike($result->{output}, qr/\[2\]/, "Context does not show non-existent line 2");
}

# Test 8: File with only target match at beginning
{
    my $test_file = create_test_file('match_start.txt', <<'EOF');
target match at start
line 2: after content
line 3: more after
line 4: even more
EOF
    
    chdir $test_dir or die "Cannot chdir to $test_dir: $!";
    my $result = run_gf_command('--search', 'target', '--context', '2');
    ok($result->{exit_code} == 0, "Context with match at start exits with code 0");
    
    # Should show match and after-context, no before-context available
    like($result->{output}, qr/\[1\].*target match at start/, "Context shows match at line 1");
    like($result->{output}, qr/\[2\].*after content/, "Context shows line 2");
    like($result->{output}, qr/\[3\].*more after/, "Context shows line 3");
    unlike($result->{output}, qr/\[0\]/, "Context does not show non-existent line 0");
}

# Test 9: File with target match at end
{
    my $test_file = create_test_file('match_end.txt', <<'EOF');
line 1: before content
line 2: more before
line 3: target match at end
EOF
    
    chdir $test_dir or die "Cannot chdir to $test_dir: $!";
    my $result = run_gf_command('--search', 'target', '--context', '2');
    ok($result->{exit_code} == 0, "Context with match at end exits with code 0");
    
    # Should show before-context and match, no after-context available
    like($result->{output}, qr/\[1\].*before content/, "Context shows line 1");
    like($result->{output}, qr/\[2\].*more before/, "Context shows line 2");
    like($result->{output}, qr/\[3\].*target match at end/, "Context shows match at line 3");
    unlike($result->{output}, qr/\[4\]/, "Context does not show non-existent line 4");
}

# Test 10: Multiple matches with overlapping context
{
    my $test_file = create_test_file('overlapping.txt', <<'EOF');
line 1: before first
line 2: first target match
line 3: between matches
line 4: second target match
line 5: after second
EOF
    
    chdir $test_dir or die "Cannot chdir to $test_dir: $!";
    my $result = run_gf_command('--search', 'target', '--context', '2');
    ok($result->{exit_code} == 0, "Context with overlapping matches exits with code 0");
    
    # Should handle overlapping context appropriately
    like($result->{output}, qr/\[1\].*before first/, "Context shows line 1");
    like($result->{output}, qr/\[2\].*first target match/, "Context shows first match");
    like($result->{output}, qr/\[3\].*between matches/, "Context shows line 3");
    like($result->{output}, qr/\[4\].*second target match/, "Context shows second match");
    like($result->{output}, qr/\[5\].*after second/, "Context shows line 5");
}

# Test 11: Context with special characters and formatting
{
    my $test_file = create_test_file('special_chars.txt', <<'EOF');
line 1: before with "quotes"
line 2: target with $pecial ch@rs & symbols
line 3: after with [brackets] and {braces}
EOF
    
    chdir $test_dir or die "Cannot chdir to $test_dir: $!";
    my $result = run_gf_command('--search', 'target', '--context', '1');
    ok($result->{exit_code} == 0, "Context with special characters exits with code 0");
    
    # Should preserve special characters in output
    like($result->{output}, qr/\[1\].*"quotes"/, "Context preserves quotes");
    like($result->{output}, qr/\[2\].*\$pecial ch\@rs & symbols/, "Context preserves special characters");
    like($result->{output}, qr/\[3\].*\[brackets\] and \{braces\}/, "Context preserves brackets and braces");
}

# Test 12: Context with very long lines
{
    my $long_line = "x" x 1000;
    my $test_file = create_test_file('long_lines.txt', <<EOF);
line 1: before long line
line 2: $long_line target match in very long line $long_line
line 3: after long line
EOF
    
    chdir $test_dir or die "Cannot chdir to $test_dir: $!";
    my $result = run_gf_command('--search', 'target', '--context', '1');
    ok($result->{exit_code} == 0, "Context with long lines exits with code 0");
    
    # Should handle long lines appropriately
    like($result->{output}, qr/\[1\].*before long line/, "Context shows line before long line");
    like($result->{output}, qr/\[2\].*target match in very long line/, "Context shows long line with match");
    like($result->{output}, qr/\[3\].*after long line/, "Context shows line after long line");
}

# Test 13: Context with tabs and whitespace
{
    my $test_file = create_test_file('whitespace.txt', <<'EOF');
	line 1: before with tab
    line 2: target with spaces
		line 3: after with tab and spaces
EOF
    
    chdir $test_dir or die "Cannot chdir to $test_dir: $!";
    my $result = run_gf_command('--search', 'target', '--context', '1');
    ok($result->{exit_code} == 0, "Context with whitespace exits with code 0");
    
    # Should preserve whitespace formatting
    like($result->{output}, qr/\[1\].*before with tab/, "Context preserves tabs");
    like($result->{output}, qr/\[2\].*target with spaces/, "Context preserves spaces");
    like($result->{output}, qr/\[3\].*after with tab/, "Context preserves mixed whitespace");
}

# Restore original directory
chdir $original_dir or die "Cannot restore directory: $!";

done_testing();