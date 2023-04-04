# EMNLP-DABERTA-2022

Code repository for the paper titled: 

**Empowering the Fact-checkers! Automatic Identification of Claim Spans on Twitter. Megha Sundriyal, Atharva Kulkarni, Vaibhav Pulastya, Md Shad Akhtar, Tanmoy Chakraborty**

Accepted at the 2022 Conference on Empirical Methods in Natural Language Processing (EMNLP'22), Abu Dhabi, December 7–11, 2022. 

Directions to implement the code: 
- Create and activate a virtual environment.
- Install all the dependencies from the requirements.txt file: pip install -r requirements.txt 
- Add all datasets in the dataset folder (train, test, and validation). The dataset folder already contains the claim descriptions embeddings from RoBERTa.
- Run python daberta.py to run the code: python daberta.py
- The model for each epoch will get saved in the models folder.
