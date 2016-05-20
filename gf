#!/usr/bin/env perl
use strict;
no warnings;
#use warnings;
use File::stat;
use Term::ANSIColor;
use Cwd;
use feature 'switch';

die "Expected search term" if @ARGV < 1;
my($term, %ignores, %targets, %seen);


sub lookupConfFile {
  my @igfile = ( $ENV{"HOME"}."/\.gfconf", "/etc/gfconf");
  foreach my $file (@igfile) {
    if( -e $file ) {
      open(my $fh, "<", $file);
      foreach my $line (<$fh>){
        chomp($line);
        if( $line =~ /^source/ip){
          my $tar = ${^POSTMATCH};
          $tar =~ s/^\s+|\s+$//g;
          push(@igfile, $tar);
        } elsif ($line =~ /^target/ip) {
          my $tar 
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
  my $targets;
  my $follow = -1;

  for( my $i = 0; $i < $argc; $i++) {
    if( $args[$i] eq "-t" ){
      if( $i != ($argc - 1)){ 
        $targets{$args[$i+1]} = 1;
        $i++;
      }else{
        die 'Invalid flag use';
      }
    } elsif( $args[$i] eq "-i" ){
      if( $i != ($argc - 1)){ 
        $ignores{$args[$i+1]} = 1;
        $i++;
      }else{
        die 'Invalid flag use';
      }
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
  return ($retTerm, %ignores, %targets);
}

($term, %ignores, %targets) = processArgs(@ARGV);
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
        print "$fn\n";
      }
      $line =~ s/^\s+|\s+$//g;
      print "[$ln]\t";
      &printStr($line);
      print "\n";

    }
    $ln++;
  }
  print "\n" if $tog == 1;
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

sub handleDir {
  my($dn) = @_;
  opendir (my $dh, $dn) or die "Could not open $dn";
  chdir($dh);
  foreach my $entry (readdir $dh){
    my $abs = $dn."/".$entry;
    $abs =~ s/^\/\//\//g;
    next if $entry =~ /^\./;
    next if 1 == shouldSkip($entry, (keys %ignores));
    next if (exists $ignores{$entry});
    next if (exists $seen{$abs});
    if(%targets){
      my $check = 0;
    }
    if ( not -R $entry){
      print color("bold yellow");
      print "lack permissions to open $abs\n";
      print color("reset");
      next;
    }

    #check for recursive links
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
