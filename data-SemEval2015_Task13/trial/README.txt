SemEval-2015 task 13
Multilingual All-Words Sense Disambiguation and Entity Linking
http://alt.qcri.org/semeval2015/task13/

This package contains the trial data for the SemEval-2015 task 13.
It contains three files.

readme.txt: this file

trialInput.xml: an xml formatted input text with document/sentence/
word/lemma/pos/id annotations. We used a reduced set of pos tags
containing 5 types: noun (N), adverb (R), adjective(A), verb (V)
and none of the above (X).

trialOutput.tsv: a tab separated value file which contains three
fields, the starting and ending ids of the textual mention to be
disambiguated in the input text and the BabelNet or Wikipedia or
WordNet id with which the mention was disambiguated by the
considered system.

Organizers and Contact Information

Andrea Moro, Sapienza University of Rome (moro@di.uniroma1.it)
Roberto Navigli, Sapienza University of Rome (navigli@di.uniroma1.it)
