package GF::Startup;

use strict;
use warnings;
use Exporter qw(import);

our @EXPORT_OK = qw(initialize_gf_system);

=head1 NAME

GF::Startup - Startup initialization system for the gf file search tool

=head1 SYNOPSIS

    use GF::Startup qw(initialize_gf_system);
    
    # Initialize the entire gf system once at startup
    initialize_gf_system($search_term, \%ignores, \%targets, $context, $maxline, $debug);

=head1 DESCRIPTION

This module provides centralized startup initialization that coordinates
pattern compilation, configuration caching, and system setup to optimize
performance throughout the application lifecycle.

=cut

use GF::PatternCache qw(initialize_pattern_cache is_cache_initialized);
use GF::ConfigCache qw(initialize_config_cache is_config_cached);

=head2 initialize_gf_system

Performs complete system initialization including pattern compilation and
configuration caching. This should be called once at startup after all
configuration loading is complete.

Arguments:
  - $search_term: The search pattern string
  - $ignores_ref: Hash reference of ignore patterns
  - $targets_ref: Hash reference of target patterns
  - $context: Number of context lines
  - $maxline: Maximum lines to read per file
  - $debug: Debug flag

Returns:
  - 1 on success, dies on error

=cut

sub initialize_gf_system {
  my ($search_term, $ignores_ref, $targets_ref, $context, $maxline, $debug) = @_;
  
  # Validate required parameters
  if (!defined $search_term || $search_term eq '') {
    die "Error: Search term is required for system initialization";
  }
  
  # Initialize configuration cache
  eval {
    initialize_config_cache($ignores_ref, $targets_ref, $context, $maxline, $debug);
  };
  if ($@) {
    die "Error initializing configuration cache: $@";
  }
  
  # Initialize pattern cache
  eval {
    initialize_pattern_cache($search_term, $ignores_ref, $targets_ref);
  };
  if ($@) {
    die "Error initializing pattern cache: $@";
  }
  
  # Verify initialization was successful
  if (!is_config_cached()) {
    die "Error: Configuration cache initialization failed";
  }
  
  if (!is_cache_initialized()) {
    die "Error: Pattern cache initialization failed";
  }
  
  # Print debug information about initialization
  if ($debug) {
    require Term::ANSIColor;
    print Term::ANSIColor::color("green");
    print "GF System Initialization Complete:\n";
    print "  - Pattern cache initialized with search pattern\n";
    print "  - " . scalar(keys %{$ignores_ref || {}}) . " ignore patterns compiled\n";
    print "  - " . scalar(keys %{$targets_ref || {}}) . " target patterns compiled\n";
    print "  - Configuration cache initialized\n";
    print Term::ANSIColor::color("reset");
  }
  
  return 1;
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