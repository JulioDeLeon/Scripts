package GF::Help;

use strict;
use warnings;
use Exporter qw(import);

our @EXPORT_OK = qw(show_usage show_help show_version show_usage_and_exit);

# Version and author information
our $VERSION = "1.0.0";
our $VERSION_DATE = "2024";
our $AUTHOR = "Julio de Leon";

=head1 NAME

GF::Help - Help system for the gf file search tool

=head1 SYNOPSIS

    use GF::Help qw(show_usage show_help show_version);
    
    show_usage();
    show_help();
    show_version();

=head1 DESCRIPTION

This module provides the help system functionality for the gf tool, including
usage messages, comprehensive help output, and version information.

=cut

sub show_usage {
  print "gf - Friendly file search tool with flexible filtering\n";
  print "\n";
  print "Usage: gf -s PATTERN [OPTIONS]\n";
  print "\n";
  print "Common examples:\n";
  print "  gf -s 'function'              # Search for 'function' in all files\n";
  print "  gf -s 'TODO' -t '*.pl'        # Search for 'TODO' in Perl files only\n";
  print "  gf -s 'error' -i '*.log'      # Search for 'error', ignoring log files\n";
  print "\n";
  print "Try 'gf --help' for complete options and more examples.\n";
}

sub show_usage_and_exit {
  print STDERR "Error: Invalid command line options provided.\n";
  print STDERR "\n";
  show_usage();
  exit 1;
}

sub show_help {
  print "gf - Friendly file search tool with flexible filtering\n";
  print "\n";
  print "DESCRIPTION\n";
  print "    gf recursively searches for text patterns within files, offering flexible\n";
  print "    filtering options to target specific file types or ignore unwanted ones.\n";
  print "    Perfect for code searches, log analysis, and content discovery.\n";
  print "\n";
  print "USAGE\n";
  print "    gf -s PATTERN [OPTIONS]\n";
  print "\n";
  print "SEARCH OPTIONS\n";
  print "    -s, --search PATTERN    Search pattern to find in files (required)\n";
  print "\n";
  print "FILE FILTERING OPTIONS\n";
  print "    -t, --target PATTERN    Target file patterns to include (e.g., '*.pl,*.pm')\n";
  print "    -i, --ignore PATTERN    File patterns to ignore (e.g., '*.log,*.tmp')\n";
  print "\n";
  print "OUTPUT CONTROL OPTIONS\n";
  print "    -c, --context NUM       Show NUM lines of context around matches\n";
  print "    -m, --maxline NUM       Maximum lines to read from each file\n";
  print "\n";
  print "UTILITY OPTIONS\n";
  print "    -d, --debug             Enable debug output\n";
  print "    -h, --help              Show this help message\n";
  print "    -V, --version           Show version information\n";
  print "\n";
  print "EXAMPLES\n";
  print "    Basic searches:\n";
  print "        gf -s 'function'                    # Find 'function' in all files\n";
  print "        gf -s 'TODO|FIXME'                  # Find TODO or FIXME comments\n";
  print "\n";
  print "    File type filtering:\n";
  print "        gf -s 'error' -t '*.pl,*.pm'       # Search only Perl files\n";
  print "        gf -s 'config' -i '*.log,*.tmp'    # Ignore log and temp files\n";
  print "\n";
  print "    Context and limits:\n";
  print "        gf -s 'sub main' -c 3               # Show 3 lines around matches\n";
  print "        gf -s 'import' -m 100               # Read only first 100 lines\n";
  print "\n";
  print "    Combined options:\n";
  print "        gf -s 'error' -t '*.pl' -c 2 -i '*.bak'  # Complex filtering\n";
  print "\n";
  print "CONFIGURATION FILES\n";
  print "    gf supports configuration files to set default ignore and target patterns:\n";
  print "\n";
  print "    Locations (checked in order):\n";
  print "        ~/.gfconf                           # User-specific config\n";
  print "        /etc/gfconf                         # System-wide config\n";
  print "\n";
  print "    Configuration syntax:\n";
  print "        ignore *.log                        # Ignore log files\n";
  print "        ignore *.tmp                        # Ignore temp files\n";
  print "        target *.pl                         # Target Perl files\n";
  print "        target *.pm                         # Target Perl modules\n";
  print "        source /path/to/other/config        # Include another config\n";
  print "\n";
  print "    Example ~/.gfconf:\n";
  print "        # Common development ignores\n";
  print "        ignore *.log\n";
  print "        ignore *.tmp\n";
  print "        ignore .git/*\n";
  print "        ignore node_modules/*\n";
  print "\n";
  print "EXIT STATUS\n";
  print "    0    Successful completion\n";
  print "    1    Error occurred\n";
  print "\n";
  print "SEE ALSO\n";
  print "    grep(1), find(1), ag(1), rg(1)\n";
  print "    For detailed documentation: man gf\n";
}

sub show_version {
  print "gf (Friendly File Search) version $VERSION\n";
  print "Released $VERSION_DATE\n";
  print "Written by $AUTHOR\n";
  print "\n";
  print "A flexible file search tool with pattern matching and filtering capabilities.\n";
  print "For help and examples, run: gf --help\n";
}

1;

__END__

=head1 AUTHOR

Julio de Leon

=head1 COPYRIGHT AND LICENSE

This software is copyright (c) 2024 by Julio de Leon.

=cut