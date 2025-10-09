package GF::ConfigCache;

use strict;
use warnings;
use Exporter qw(import);

our @EXPORT_OK = qw(
  initialize_config_cache
  get_cached_config
  is_config_cached
  clear_config_cache
);

=head1 NAME

GF::ConfigCache - Configuration caching system for the gf file search tool

=head1 SYNOPSIS

    use GF::ConfigCache qw(initialize_config_cache get_cached_config);
    
    # Initialize configuration cache once at startup
    initialize_config_cache(\%ignores, \%targets, $context, $maxline, $debug);
    
    # Retrieve cached configuration
    my $config = get_cached_config();

=head1 DESCRIPTION

This module provides configuration caching functionality to avoid repeated
parsing of configuration files and command-line arguments.

=cut

# Global configuration cache
my %config_cache;
my $cache_initialized = 0;

=head2 initialize_config_cache

Caches configuration parameters to avoid repeated parsing.

Arguments:
  - $ignores_ref: Hash reference of ignore patterns
  - $targets_ref: Hash reference of target patterns
  - $context: Number of context lines
  - $maxline: Maximum lines to read per file
  - $debug: Debug flag

=cut

sub initialize_config_cache {
  my ($ignores_ref, $targets_ref, $context, $maxline, $debug) = @_;
  
  # Deep copy the pattern hashes to avoid reference issues
  my %cached_ignores = %{$ignores_ref || {}};
  my %cached_targets = %{$targets_ref || {}};
  
  %config_cache = (
    ignores => \%cached_ignores,
    targets => \%cached_targets,
    context => $context || 0,
    maxline => $maxline || 0,
    debug => $debug || 0,
    ignore_count => scalar(keys %cached_ignores),
    target_count => scalar(keys %cached_targets),
  );
  
  $cache_initialized = 1;
}

=head2 get_cached_config

Returns the cached configuration.

Returns:
  - Hash reference containing cached configuration parameters

=cut

sub get_cached_config {
  return \%config_cache;
}

=head2 is_config_cached

Checks if configuration has been cached.

Returns:
  - 1 if configuration is cached, 0 otherwise

=cut

sub is_config_cached {
  return $cache_initialized;
}

=head2 clear_config_cache

Clears the configuration cache. Useful for testing or reinitialization.

=cut

sub clear_config_cache {
  %config_cache = ();
  $cache_initialized = 0;
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