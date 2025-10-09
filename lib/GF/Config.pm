package GF::Config;

use strict;
use warnings;
use Exporter qw(import);

our @EXPORT_OK = qw(lookup_config_file print_debug);

=head1 NAME

GF::Config - Configuration file handling for the gf file search tool

=head1 SYNOPSIS

    use GF::Config qw(lookup_config_file print_debug);
    
    my ($ignores_ref, $targets_ref) = lookup_config_file(\%ignores, \%targets, $debug);

=head1 DESCRIPTION

This module handles configuration file processing for the gf tool, including
reading configuration files, parsing directives, and managing ignore/target patterns.

=cut

sub print_debug {
  my ($debug, $str) = @_;
  if ($debug) {
    require Term::ANSIColor;
    print Term::ANSIColor::color("magenta");
    print $str;
    print Term::ANSIColor::color("reset");
  }
}

=head2 lookup_config_file

Processes gf configuration files to load ignore and target patterns.

Arguments:
  - $ignores_ref: Hash reference to store ignore patterns
  - $targets_ref: Hash reference to store target patterns  
  - $debug: Debug flag for verbose output

Returns:
  - Updated ignore and target pattern hashes

Configuration file locations (checked in order):
  - ~/.gfconf (user-specific)
  - /etc/gfconf (system-wide)

=cut

sub lookup_config_file {
  my ($ignores_ref, $targets_ref, $debug) = @_;
  
  my @conffile = ( $ENV{"HOME"}."/\.gfconf", "/etc/gfconf");
  
  foreach my $file (@conffile) {
    if( -e $file ) {
      print_debug($debug, "looking at file: $file\n");
      
      # Check if we can read the configuration file
      if (! -r $file) {
        print STDERR "Warning: Configuration file '$file' exists but is not readable.\n";
        print STDERR "Check file permissions to use configuration settings.\n";
        next;
      }
      
      my $fh;
      if (!open($fh, "<", $file)) {
        print STDERR "Warning: Could not open configuration file '$file': $!\n";
        print STDERR "Continuing without this configuration file.\n";
        next;
      }
      
      my $line_num = 0;
      foreach my $line (<$fh>){
        $line_num++;
        chomp($line);
        
        # Skip empty lines and comments
        next if $line =~ /^\s*$/ || $line =~ /^\s*#/;
        
        if( $line =~ /^source\s/ip){
          my $tar = ${^POSTMATCH};
          $tar =~ s/^\s+|\s+$//g;
          print_debug($debug, "Found source file [$tar]\n");
          
          # Validate source file exists
          if (! -e $tar) {
            print STDERR "Warning: Source file '$tar' specified in '$file' (line $line_num) does not exist.\n";
            print STDERR "Skipping this source directive.\n";
            next;
          }
          
          push(@conffile, $tar);
        } elsif ($line =~ /^target\s/ip) {
          my $tar = ${^POSTMATCH};
          $tar =~ s/^\s+|\s+$//g;
          print_debug($debug, "Found target pattern [$tar]\n");
          $targets_ref->{$tar} = 1;
        } elsif ($line =~ /^ignore\s/ip) {
          my $tar = ${^POSTMATCH};
          $tar =~ s/^\s+|\s+$//g;
          print_debug($debug, "Found ignore pattern [$tar]\n");
          $ignores_ref->{$tar} = 1;
        } else {
          # Unknown configuration directive
          print STDERR "Warning: Unknown configuration directive in '$file' (line $line_num): $line\n";
          print STDERR "Valid directives are: source, target, ignore\n";
        }
      }
      close $fh;
    }
  }
  
  return ($ignores_ref, $targets_ref);
}

1;

__END__

=head1 AUTHOR

Julio de Leon

=head1 COPYRIGHT AND LICENSE

This software is copyright (c) 2024 by Julio de Leon.

=cut