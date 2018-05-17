#!/usr/bin/perl -w


#USAGE: score.pl [-g granularity {coarse|mixed|fine}] [-m exclude instances with multiple tags] [-v verbose] answs.file key.file

# SCORER2 OPTIONS:
# This scorer requires two command-line arguments in the
# following order:
#
#    ANSWER FILE NAME (name of a file containing formatted answers)
#
#    KEY FILE NAME (name of an answer-key file)
#
# Optionally, the following may be appended:
#
#    SENSE FILE NAME (name of a file containing sense-map information)
#      - without this file, only fine-grained scoring is available,
#        and illformed sense tags will lower precision (rather than recall)
#
#    -g (specifies granularity: "coarse" or "mixed"; "fine" is default)
#
#    -m (causes exclusion of instances tagged with multiple tags in key)
#
#    -v (causes line-by-line scoring calculations to be printed)
#


use strict;
use IO::File;
use File::Basename;
use Getopt::Std;


my $baseDir = ".";
my $eval = "$baseDir/scorer2";

my $tmpDir = "/tmp";


## Get options
my %opts = ();
getopts("g:mv",\%opts);

my $opt = $opts{"g"} ? "-g ".$opts{'g'} : "";
$opt = $opts{"m"} ? $opt." -m" : "";
$opt = $opts{"v"} ? $opt." -v" : "";
$opt =~ s/^\s+//;


die "USAGE: score.pl [-g granularity {coarse|mixed|fine}] [-m exclude instances with multiple tags] [-v verbose] answs.file key.file\n" if not defined $ARGV[1];

## Sort files:
my $ansFile =  basename($ARGV[0]);
my $fh = IO::File->new($ARGV[0],"r"); die "Cannot open $ARGV[0]\n" unless defined $fh;
my @lines = <$fh>;
$fh->close();

$fh = IO::File->new("$tmpDir/$ansFile","w"); die "Cannot open $tmpDir/$ansFile\n" unless defined $fh;
foreach (sort{$a cmp $b} @lines){
    print $fh $_;
}
$fh->close();

my $keyFile =  basename($ARGV[1]);
$fh = IO::File->new($ARGV[1],"r"); die "Cannot open $ARGV[1]\n" unless $fh;
@lines = <$fh>;
$fh->close();

$fh = IO::File->new("$tmpDir/$keyFile","w"); die "Cannot open $tmpDir/$keyFile\n" unless $fh;
foreach (sort{$a cmp $b} @lines){
    print $fh $_;
}
$fh->close();



## Eval answers: 
system "$eval $tmpDir/$ansFile $tmpDir/$keyFile $opt";
