#!/usr/bin/perl -w
use strict;
use warnings;
use File::stat;
use Cwd;
use feature 'switch';

die "Expected search term" if @ARGV < 1;
my($term) = @ARGV;

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

sub checkFile {
	my($fn) = @_;
	my $ln = 1, my $tog = 0;
	open(my $fh, "<", $fn);
	foreach my $line (<$fh>){
		chomp($line);
		if($line =~ /$term/){
			if($tog == 0){
				$tog = 1;
				print "\n$fn\n";
			}
			$line =~ s/^\s+|\s+$//g;
			print "[$ln]\t$line\n";
		}
		$ln++;
	}
	close $fh;
}

sub handleDir {
	my($dn) = @_;
	opendir (my $dh, $dn) or die "Could not open $dn";
	chdir($dh);
	foreach my $entry (readdir $dh){
		next if $entry =~ /^\./;

		if( 1 == (-d $entry)){
			&handleDir ($dn."/".$entry);
		}elsif( -f $entry){
			#if(&checkExt($entry)){
			next if (-B $entry);
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
