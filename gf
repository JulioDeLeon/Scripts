#!/usr/bin/perl
use strict;
no warnings;
#use warnings;
use File::stat;
use Term::ANSIColor;
use Cwd;
use feature 'switch';

die "Expected search term" if @ARGV < 1;

sub printHelp {
  print "Help\n"
}

sub processArgs {
  my @args = @_;
  my $argc = @_;
  my $retTerm = undef;
  my %ignores;
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
    } else {
      if( $retTerm eq undef){
        $retTerm = $args[$i];
      } else {
        die "Found multiple search terms\n";
      }
    }
  }
  return ($retTerm, %ignores);
}

my($term, %ignores) = processArgs(@ARGV);

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
        if( 1 == (-d $entry)){
      &handleDir ($dn."/".$entry);
    }elsif( -f $entry){
#if(&checkExt($entry)){
  next if (-B $entry);
  next if $entry =~ /\.a$/;
  next if $entry =~ /\.bundle$/; #VMWare's bundle breaks this, so skip for now. 
    next if $entry =~ /\.run$/; #nvidia's .run
    &checkFile($dn."/".$entry);
#}
    }
  }
  chdir("..");
  closedir $dh;
}

sub main {

  &handleDir(getcwd);
}

&main;
