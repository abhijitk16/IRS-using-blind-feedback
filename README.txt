*In order to run the query, open main.py using command "python main.py" on terminal and enter the query.
 Click on search to get the ranked results. The dataset "d,q and r " folders should be present in the same folder as the main.py*

*In order to run the intermediate files, take them out of the folder, i.e. in a folder containing "d,q and r" folders*

Required libraries to run the code-
numpy 
nltk 
nimfa 
Tkinter
s klearn 
scipy
 itertools (only for rocchio)

Overview of the files-

"d", "q" and "r" folders contains documents, queries and relevent results respectively. 
This is the dataset used by us for the experiment.

main.py - Contains the code for GUI as well as retrieved ranked results as per the query after performing matrix factorixation.

retrieved documents ranked.txt - Shows the top 15 documents retrieved in ranked order  for each query present in the "q" folder.

Intermediate results folder has files corresponding to different heuristics and methids i.e.-
	
	noprf - A simple ranked retrieval along with average precision, p@10, p@20 for each query and then overall MAP, p@10, p@20 	
		calculated by taking average of the data in the csv file.
	
	rocchio - Ranked retrieval using rocchio's algorithm along with average precision, p@10, p@20 for each query and then overall MAP, 			  p@10, p@20 calculated by taking average of the data in the csv file.
	
	tf - Ranked retrieval after applying matrix factorization and considering term frequency as the weight function along with average 		     precision, p@10, p@20 for each query and then overall MAP, p@10, p@20 calculated by taking average of the data in the csvfile.
	
	tfidf - Ranked retrieval after applying matrix factorization and considering term frequency as well as inverse document frequency 			as the weight function along with average precision, p@10, p@20 for each query and then overall MAP, p@10, p@20 calculated 			by taking average of the data in the csv file.
	

 
