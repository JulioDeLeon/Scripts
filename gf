#!/usr/bin/env perl
use strict;
no warnings;
#use warnings;
#look at Getopt library
#also look at data dump
use File::stat;
use Term::ANSIColor;
use Cwd;
use Getopt::Long;
use feature 'switch';

my $DEBUG = 0;
die "Expected search term" if @ARGV < 1;
my($term, %ignores, %targets, %seen, $context);

sub printDEBUG {
  if ($DEBUG == 1) {
    my ($str) = @_;
    print color("magenta");
    print $str;
    print color("reset");
  }
}

sub printTars {
  printDEBUG "ign: " . join(" ", keys %ignores) . "\n";
  printDEBUG "tar: " . join(" ", keys %targets) . "\n";
}

=begin
  lookupConfFile()
  gf script can handle a general configuration file which can hold 
  a list of file extensions that can be ignored or targed.
=cut
sub lookupConfFile {
  my @conffile = ( $ENV{"HOME"}."/\.gfconf", "/etc/gfconf");
  foreach my $file (@conffile) {
    if( -e $file ) {
      printDEBUG "looking at file: $file\n";
      open(my $fh, "<", $file);
      foreach my $line (<$fh>){
        chomp($line);
        if( $line =~ /^source\s/ip){
          my $tar = ${^POSTMATCH};
          $tar =~ s/^\s+|\s+$//g;
          printDEBUG "Found source file [$tar]\n";
          push(@conffile, $tar);
        } elsif ($line =~ /^target\s/ip) {
          my $tar = ${^POSTMATCH};
          $tar =~ s/^\s+|\s+$//g;
          printDEBUG "Found target pattern [$tar]\n";
          $targets{$tar} = 1;
        } elsif ($line =~ /^ignore\s/ip) {
          my $tar = ${^POSTMATCH};
          $tar =~ s/^\s+|\s+$//g;
          printDEBUG "Found ignore pattern [$tar]\n";
          $ignores{$tar} = 1;
        }
      }
      close $fh;
    }
  }     
}

sub processArgs {
  my $retTerm = undef;
  my %retIgnores;
  my %retTargets;
  my @ignoresArr;
  my @targetsArr;
  $context = 0;
  GetOptions("search=s" => \$retTerm,
             "target=s" => \@targetsArr,
             "ignore=s" => \@ignoresArr,
             "context=i" => \$context,
             "debug" => \$DEBUG)
  or die "Error: Could not process command line arguements";
  printDEBUG("retTerm: " . $retTerm . "\n");
  printDEBUG "ignoreArr: " . join(" ", @ignoresArr) . "\n";
  printDEBUG "targetArr: " . join(" ", @targetsArr) . "\n";
  die "NO SEARCH TERM" if (! defined $retTerm);

  #allow commas in targets and ignores 
  @ignoresArr = split(/,/, join(',', @ignoresArr));
  @targetsArr = split(/,/, join(',', @targetsArr));

  foreach my $ign (@ignoresArr) {
    $ignores{$ign} = 1;
  }

  foreach my $tar (@targetsArr) {
    $targets{$tar} = 1;
  }
  $term =  $retTerm;
}

&processArgs;
&printTars;
&lookupConfFile;

=begin
  printStr( line )
    for output, function will be called if the search term is match, printing
    the matched term in red.
=cut
sub printStr {
  my ($str) = @_;

  if ( $str =~ /$term/p) {
    my $pre = ${^PREMATCH};
    my $mat = ${^MATCH};
    my $pos = ${^POSTMATCH};
    
    $str =~ s/^\s+|\s+$//g;
    print "$pre";
    print color("bold red");
    print "$mat";
    print color("reset");
    &printStr($pos);
  }else{
    print "$str";
  }
}

=begin
  checkFile( filepath ) 
  with filepath being an abs path to a file, checkFile will open the given file
  and search for the search term. if the term is found, it will print the file
  path and the lines which the search term was found
=cut
sub checkFile {
  printDEBUG "in checkFile\n";
  my($fn) = @_;
  my $ln = 1, my $tog = 0;
  my $max = 0;
  my @fileContext;

  open(my $fh, "<", $fn);
  foreach my $line (<$fh>){
    chomp($line);
    push(@fileContext, $line);
    $max = $max + 1;
  }
  
  foreach my $line (@fileContext){
    if($line =~ /$term/p){
      if($tog == 0){
        $tog = 1;
        print "$fn\n";
      }
#Print context before match
      if($context) {
        for(my $x = ($ln - $context); $x < $ln; $x++){
          if($x > 0){
            print "[$x]\t";
            print $fileContext[$x];
            print "\n";
          }
        }
      }

      $line =~ s/^\s+|\s+$//g;
      print "[$ln]\t";
      &printStr($line);
      print "\n";

#print context after match
      if($context){
         for(my $x = ($ln + 1); $x < $ln + $context + 1; $x++){
          if($x < $max){
            print "[$x]\t";
            print $fileContext[$x];
          }
          print "\n";
        }
        print "\n"
      }
    }
    $ln++;
  }
  print "\n" if $tog == 1;
  close $fh;
}

=begin
  shouldSkip( filename, [undesired terms] )
  determines if the file should be skipped based on a list of undesired terms.
  if the file matches with any one of the undersired terms, it returns true
=cut
sub shouldSkip {
  my ($entry) =  @_;
  my @ign = keys %ignores;
  my @tar = keys %targets;
  printDEBUG("should skip ". $entry ."?\n");
  my $skip = 0;
  foreach my $key (@ign) {
    printDEBUG("\tign pattern: " . $key . "\n");
    my $pattern = qr/$key/i;
    ($skip = 1) if ($entry =~ /$pattern/);
    if($skip){
      printDEBUG("\treturning " . $skip . "\n"); 
      return $skip;
    }
  }
  foreach my $key (@tar) {
    printDEBUG("\ttar pattern: " . $key . "\n");
    my $pattern = qr/$key/i;
    ($skip = 1) if not ($entry =~ /$pattern/);
    if($skip){
      printDEBUG("\treturning " . $skip . "\n"); 
      return $skip;
    }
  }
  printDEBUG("\treturning " . $skip . "\n"); 
  return $skip;
}

=begin
  checkLink( $directory, $filename )
  checks if the file under the given directory is a soft link. returns true if
  it is. This is done to help avoid self linking soft links. (This is a known
  issue in some Ubuntu distrubitions and will break if not checked)
=cut
sub checkLink {
  my ($dn, $entry) = @_;
  my $abs = $dn."/".$entry;
  $abs =~ s/^\/\//\//g;
  if(-l $entry){
    my $rlink = readlink $entry;
    if( !($rlink =~ /^\//)) {
      $rlink = $dn."/".$rlink;
    }
    if(not exists $seen{$rlink}){
      $seen{$rlink} = 1;
      $seen{$abs} = 1;
    }
  }
}

=begin
  handleDir( directory_path )
  Looks through each entry in a directory, if a file is found, checks the file
  for the desired term. If a directory is found, it recursively handleDir' that
  directory.  
=cut
sub handleDir {
  my($dn) = @_;
  opendir (my $dh, $dn) or die "Could not open $dn";
  chdir($dh);
  foreach my $entry (readdir $dh){
    my $abs = $dn."/".$entry;
    $abs =~ s/^\/\//\//g;
    next if $entry =~ /^\./;
    next if 1 == shouldSkip($entry);
    next if (exists $seen{$abs});
    if ( not -R $entry){
      print color("bold yellow");
      print "lack permissions to open $abs\n";
      print color("reset");
      next;
    }

    #check for recursive links
    next if (-l $entry);
    if(-l $entry){
      my $link = readlink $entry;
      next if $link eq ".";
    }
    #next if (-l $entry);
    if( 1 == (-d $entry)){
      &checkLink($dn, $entry);
      &handleDir ($abs);
    }elsif( -f $entry){
      &checkLink($dn, $entry);
      next if (-B $entry);
      &checkFile($abs);
    }
  }
  chdir("..");
  closedir $dh;
}

sub main {
  &handleDir(getcwd);
}

&main;
