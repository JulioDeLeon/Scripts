package GF::PatternCache;

use strict;
use warnings;
use Exporter qw(import);

our @EXPORT_OK = qw(
  initialize_pattern_cache
  get_search_pattern
  get_ignore_patterns
  get_target_patterns
  is_cache_initialized
);

=head1 NAME

GF::PatternCache - Pattern compilation and caching system for the gf file search tool

=head1 SYNOPSIS

    use GF::PatternCache qw(initialize_pattern_cache get_search_pattern);
    
    # Initialize patterns once at startup
    initialize_pattern_cache($search_term, \%ignores, \%targets);
    
    # Use compiled patterns throughout the application
    my $search_pattern = get_search_pattern();
    my $ignore_patterns = get_ignore_patterns();

=head1 DESCRIPTION

This module provides pattern compilation and caching functionality to optimize
regex performance by compiling patterns once at startup and reusing them
throughout the application lifecycle.

=cut

# Global pattern cache variables
my $search_pattern;
my @ignore_patterns;
my @target_patterns;
my $cache_initialized = 0;

=head2 initialize_pattern_cache

Compiles and caches all regex patterns for reuse throughout the application.
This should be called once at startup after configuration loading.

Arguments:
  - $search_term: The search pattern string
  - $ignores_ref: Hash reference of ignore patterns
  - $targets_ref: Hash reference of target patterns

=cut

sub initialize_pattern_cache {
  my ($search_term, $ignores_ref, $targets_ref) = @_;
  
  # Compile search pattern
  if (defined $search_term && $search_term ne '') {
    eval {
      $search_pattern = qr/$search_term/;
    };
    if ($@) {
      die "Error compiling search pattern '$search_term': $@";
    }
  }
  
  # Compile ignore patterns
  @ignore_patterns = ();
  if ($ignores_ref && ref($ignores_ref) eq 'HASH') {
    foreach my $ignore_pattern (keys %$ignores_ref) {
      next if !defined $ignore_pattern || $ignore_pattern eq '';
      
      # Convert glob pattern to regex
      my $regex_pattern = _glob_to_regex($ignore_pattern);
      
      eval {
        push @ignore_patterns, qr/$regex_pattern/i;
      };
      if ($@) {
        warn "Warning: Could not compile ignore pattern '$ignore_pattern': $@";
      }
    }
  }
  
  # Compile target patterns
  @target_patterns = ();
  if ($targets_ref && ref($targets_ref) eq 'HASH') {
    foreach my $target_pattern (keys %$targets_ref) {
      next if !defined $target_pattern || $target_pattern eq '';
      
      # Convert glob pattern to regex
      my $regex_pattern = _glob_to_regex($target_pattern);
      
      eval {
        push @target_patterns, qr/$regex_pattern/i;
      };
      if ($@) {
        warn "Warning: Could not compile target pattern '$target_pattern': $@";
      }
    }
  }
  
  $cache_initialized = 1;
}

=head2 get_search_pattern

Returns the compiled search pattern.

Returns:
  - Compiled regex pattern for search term

=cut

sub get_search_pattern {
  return $search_pattern;
}

=head2 get_ignore_patterns

Returns array reference of compiled ignore patterns.

Returns:
  - Array reference of compiled ignore regex patterns

=cut

sub get_ignore_patterns {
  return \@ignore_patterns;
}

=head2 get_target_patterns

Returns array reference of compiled target patterns.

Returns:
  - Array reference of compiled target regex patterns

=cut

sub get_target_patterns {
  return \@target_patterns;
}

=head2 is_cache_initialized

Checks if the pattern cache has been initialized.

Returns:
  - 1 if cache is initialized, 0 otherwise

=cut

sub is_cache_initialized {
  return $cache_initialized;
}

=head2 _glob_to_regex (private)

Converts glob patterns to regex patterns.

Arguments:
  - $glob_pattern: Glob pattern string

Returns:
  - Regex pattern string

=cut

sub _glob_to_regex {
  my ($glob_pattern) = @_;
  
  # Escape regex special characters except * and ?
  my $regex_pattern = $glob_pattern;
  $regex_pattern =~ s/([.+^\${}()|\\])/\\$1/g;
  
  # Convert glob wildcards to regex
  $regex_pattern =~ s/\*/.*?/g;  # * becomes .*?
  $regex_pattern =~ s/\?/./g;    # ? becomes .
  
  return $regex_pattern;
}

1;

__END__

=head1 AUTHOR

Julio de Leon

=head1 COPYRIGHT AND LICENSE

This software is copyright (c) 2024 by Julio de Leon.

=cut
</content>
</invoke>