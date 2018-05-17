Data used for the experiments in: Multimodal word meaning induction from minimal exposure to natural text

* dataset.txt
Data used for Experiment 1 and Experiment 2.
Fields:
	- TRIAL: Id of trial
	- NONCE: The nonce word masking the real words
	- IMAGE_PROBE_URL: The image of the probe
	- PASSAGE: The contructed passage for the chimera. Sentences in the passage are separated with "@@"
	- RESPONSE: The average human responses
	- VARIANCE: The variance of the human responses
	- PASSAGE_LENGTH: Number of sentences in the passage  
	- IMAGE_QUALITY: The "quality" of the image probe. From A to F with decreasing relatedness between chimera-probe. 
	- CHIMERA: The concepts forming the chimera in the form of conceptA_conceptB
	- WORD_COUNT: Number of words in the passage
	- INFORMATIVENESS_CHIMERA: Informativeness of whole passage
	- INFORMATIVENESS_CHIMERA_A: Informativeness of sentences corresponding to conceptA
	- INFORMATIVENESS_CHIMERA_B: Informativeness of sentences corresponding to conceptB

* ratings.txt
Ratings collected for the control experiment of Section 2.1.3

