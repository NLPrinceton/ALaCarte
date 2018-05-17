			     SEMEVAL-2013
			    -------------

               TASK 12: Multilingual Word Sense Disambiguation
	    Organizers: Roberto Navigli and David Jurgens 

Many thanks for your interest in the SemEval-2013 task on multilingual WSD! The
test data package contains the following files:


  README			     this file
  data/				     directory with the test data
    multilingual-all-words.dtd	     dtd for input files
    multilingual-all-words.de.xml    test data (GERMAN)
    multilingual-all-words.en.xml    test data (ENGLISH)
    multilingual-all-words.es.xml    test data (SPANISH)
    multilingual-all-words.fr.xml    test data (FRENCH)
    multilingual-all-words.it.xml    test data (ITALIAN)


  docs/				    documentation for the task
    answer-format.txt		    SENSEVAL answer format doc
    documentation.txt 		    SENSEVAL scorer doc
    scorescheme.txt		    SENSEVAL scoring scheme

  keys/                             directory with the key files
  keys/gold/                        directory with all gold-standard keys
  keys/baselines/                   directory with keys for all baselines
  keys/submissions/                 directory with keys for all participants

  scorer/                           directory with the scoring program

--------------
Task languages 
--------------

The test data contains contexts in five languages: English, French, German,
Italian, and Spanish.

--------------
Answer format
--------------

The answer format is the same of the gold-standard format (as found in
the keys included in the trial data). See also the task's webpage for
additional information:

http://www.cs.york.ac.uk/semeval-2013/task12/index.php?id=data

We allow as sense labels any of the WordNet sense keys, Wikipedia page titles or
Babel synset offsets found within the sense inventory. Please look at the
gold-standard files within the keys/ directory of the trial data for examples of
system outputs for any of these sense label formats. Note that systems will be
evaluated separately depending on the type of sense label they output.

Note that the answer format also allows systems to return multiple
weighted senses. Please check the original SENSEVAL documentation (found
in the "doc" directory) for details.

-------------
Participating
-------------

In order to participate in the multilingual WSD task, you need to:

   1. Download the test data from the FTP site (i.e., the data included in this
      archive)

   2. (optional) If using the BabelNet sense inventory, download the latest
      version of BabelNet 1.1.1 from http://babelnet.org

   3. Upload up to three system results for your team to the FTP site by
      midnight (PST) on March 15th 2013, including all data specified below
  
   4. Write a paper describing your system and submit it by April 9 [TBC]

Participants may use any of the three sense inventories: WordNet 3.0, BabelNet
1.1.1, or Wikipedia.  For Wikipedia, senses are specified as Wikipedia page
titles.  WordNet senses are specified using sense keys (e.g., month%1:28:01::)
and BabelNet senses are specified using synset offsets (e.g., bn:00014710n).
For further examples, please look at the .key solution files in the trial data.
For those participants using BabelNet, we have now released a new gold-standard
version of the BabelNet sense inventory (version 1.1.1), which is currently
available at http://babelnet.org .  In version 1.1.1, the synsets of all EN target words in Task 12 have all been
manually verified and corrected.  Participants that use BabelNet should use the
1.1.1 inventory for obtaining their BabelNet synsets offsets.

Each team should submit each of their system's solutions in a separated archive
file (.zip or .tgz).  Archives should be named as task12-TEAM-APPROACH where
TEAM is your team's registered name and APPROACH is a unique identifier used by
your team for that solution.  Submissions should include the sense annotations
and a short README describing the system that produced the annotations.  We ask
that each submission's README include (1) which sense inventory is used, (2) a
description of the WSD system, (3) if the system is supervised, what training
data was used or if the system is knowledge-based, what knowledge resources were
used, and (4) what languages are being annotated.

Teams will upload each of their submissions to the FTP server at
semeval2013.ku.edu.tr .  Log in information should be provided when registering
the team on the SemEval website www.cs.york.ac.uk/semeval-2013 .

-------
Scoring
-------

Please note that the official scoring programs are case sensitive and will be
sensitive if a different default character encoding is used (due to the presence
of accented characters).  If comparing scores with existing systems, please
lower case the keys and use a utf-8 compatible encoding to ensure that the
systems' scores are directly comparable with those from the official results.
For any questions on scoring, please contact the organizers.

----------
Contact us
----------

Feel free to contact us for any problem by joining and posting in our
Google group: http://groups.google.com/group/semeval13-multilingual-wsd

----------------
Acknowledgements
----------------

This SemEval task has been developed in the context of the ERC Starting
Grant MultiJEDI No. 259234.
