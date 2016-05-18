#!/usr/bin/env perl
use strict;
no warnings;
#use warnings;
use File::stat;
use Term::ANSIColor;
use Cwd;
use feature 'switch';

die "Expected search term" if @ARGV < 1;
my($term, %ignores, $fsym);

sub printHelp {
  print "Help\n"
}

sub lookupIgnoreFile {
  my @igfile = ( $ENV{"HOME"}."/\.gfignore", "/etc/gfignore");
  foreach my $file (@igfile) {
    if( -e $file ) {
      open(my $fh, "<", $file);
      foreach my $line (<$fh>){
        chomp($line);
        if( $line =~ /^source/p){
          my $tar = ${^POSTMATCH};
          $tar =~ s/^\s+|\s+$//g;
          push(@igfile, $tar);
        } else {
          $ignores{$line} = 1;
        }
      }
      close $fh;
    }
  }     
}

sub processArgs {
  my @args = @_;
  my $argc = @_;
  my $retTerm = undef;
  my %ignores;
  my $follow = -1;

  for( my $i = 0; $i < $argc; $i++) {
    if( $args[$i] eq "-i" ){
      if( $i != ($argc - 1)){ 
        $ignores{$args[$i+1]} = 1;
        $i++;
      }else{
        die 'Invalid flag use';
      }
    } elsif( $args[$i] eq "-h" ){
      &printHelp;
      exit(0);
    } elsif( $args[$i] eq "-fs") {
      $follow = 1;
    } else {
      if( $retTerm eq undef){
        $retTerm = $args[$i];
      } else {
        die "Found multiple search terms\n";
      }
    }
  }
  return ($retTerm, %ignores, $follow);
}

($term, %ignores, $fsym) = processArgs(@ARGV);
&lookupIgnoreFile;

sub checkExt {
  my($fn) = @_;
  my @appr = [qr/\.c$/, qr/\.cpp$/, qr/\.pl$/, qr/\.txt$/, qr/\.h$/, qr/\.java$/, qr/\.hs$/, qr/\.hs$/, qr/\.py$/];
  my @match;
  given($fn){
    when(@appr){
      return 1;
    }
    default{
      return 0;
    }
  }
}

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

sub checkFile {
  my($fn) = @_;
  my $ln = 1, my $tog = 0;
  open(my $fh, "<", $fn);
  foreach my $line (<$fh>){
    chomp($line);
    if($line =~ /$term/p){
      if($tog == 0){
        $tog = 1;
        print "\n$fn\n";
      }
      $line =~ s/^\s+|\s+$//g;
      print "[$ln]\t";
      &printStr($line);
      print "\n";

    }
    $ln++;
  }
  close $fh;
}

sub shouldSkip {
  my ($entry, @list) =  @_;

    my $skip = -1;
    foreach my $key (@list) {
      my $pattern = qr/$key/i;
      ($skip = 1) if ($entry =~ /$pattern/);
    }
  
  return $skip;
}

sub handleDir {
  my($dn) = @_;
  opendir (my $dh, $dn) or die "Could not open $dn";
  chdir($dh);
  foreach my $entry (readdir $dh){
    next if $entry =~ /^\./;
    next if 1 == shouldSkip($entry, (keys %ignores));
    next if (exists $ignores{$entry});
    next if (-l $entry && $fsym == -1);
    if( 1 == (-d $entry)){
      &handleDir ($dn."/".$entry);
    }elsif( -f $entry){
      next if (-B $entry);
      &checkFile($dn."/".$entry);
    }
  }
  chdir("..");
  closedir $dh;
}

sub main {

  &handleDir(getcwd);
}

&main;
