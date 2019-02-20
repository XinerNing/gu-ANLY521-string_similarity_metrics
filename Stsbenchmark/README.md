Semantic textual similarity using string similarity
---------------------------------------------------

This project examines string similarity metrics for semantic textual similarity.
Though semantics go beyond the surface representations seen in strings, some of these
metrics constitute a good benchmark system for detecting STS.


Data is from the [STS benchmark](http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark).

## Discription of each metrics used
* NIST: a method for evaulating the quality of text which has been translated using machine translation.
* BLEU: a algorithm for evaulating the quality of text which has been machine-translated from one natural language to another.
* Word Error Rate: is a measure of the performance of an automatic speech recognition, machine translation etc.
* Longest common substring: given two strings, find the lenght of their longest common substring.
* Levenshtein distance: is a string metric for measuring difference between two sequences. It is the minumum number of single character edits, including insertions, deletions and substitutions required to change one word into the other.



## Discription of sts_pearson.py
Calculate metrics described above, and compare the scores to the standard label to get the correlations. Results are written in a .txt file. 
## A usage example showing command line flags
Under the directory of sts_pearson.py, 'python sts_pearson.py' will gives you a test_output.txt


## Description of output
Output of running sts_pearson.py is sts-test.txt, which contains the correlations between metrics described above and the standard labels in the sts-test.csv under STS benchmark. 

