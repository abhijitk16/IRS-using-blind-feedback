from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import average_precision_score
import numpy as np
import nltk
from nltk.stem import PorterStemmer
import nimfa
from sklearn import preprocessing
from scipy import sparse
import Tkinter as tkr

english_stemmer = PorterStemmer()
class StemmedTfidfVectorizer(TfidfVectorizer):
	def build_analyzer(self):
		analyzer = super(TfidfVectorizer, self).build_analyzer()
		return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))
corpus = []
# prepare corpus and relevance
corpus.append(" ")
for d in range(0,1400):
	f = open("./d/"+str(d+1)+".txt")
	corpus.append((f.read()))
def printtext():
	global e
	string = e.get() 
	corpus[0]=string
	tfidf_vectorizer = StemmedTfidfVectorizer(min_df=1, stop_words='english', analyzer='word', ngram_range=(1,1))
	tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
	# Cosine similarity measure
	csim = np.array(cosine_similarity(tfidf_matrix[0], tfidf_matrix[1:len(corpus)])[0])
	#using 10 feedback documents
	x=10
	arr=np.array([0])
	arr1=np.append(arr,csim.argsort()[-x:][::-1])
	arr2=(tfidf_matrix.todense())[arr1]
	#Applying matrix factorization using nimfa library
	nmf = nimfa.Nmf(arr2, seed="nndsvd",W=None,H=None, rank=x+1, max_iter=100, update='euclidean',objective='div')
	xq=nmf()
	#F is the estimated matrix
	F = xq.fitted()

	#using 50 feedback terms
	p=50
	arr3=np.zeros(F[0].size)
	top_terms= np.squeeze(np.asarray(F[0].argsort()[-p:][::-1]))
	query_vector= np.squeeze(np.asarray(tfidf_matrix.todense()[0]))
	estimated_query_vector=np.squeeze(np.asarray(F[0]))
	for k in top_terms:
		arr3[k]=estimated_query_vector[k]

	#normalization
	sum1=0.0
	for j in range(0,arr3.size):
		sum1=sum1+arr3[j]*arr3[j]

	root=np.sqrt(sum1)
	F_normalized=np.zeros(F[0].size)
	for j in range(0,arr3.size):
		F_normalized[j]=arr3[j]/root


	#taking 0.4 as feedback coefficient
	r=0.4
	for i in range(0,F[0].size):
		estimated_query_vector[i]=(1-r)*query_vector[i] + r*F_normalized[i]

	#normalization
	sum2=0.0
	for j in range(0,estimated_query_vector.size):
		sum2=sum2+estimated_query_vector[j]*estimated_query_vector[j]
	root1=np.sqrt(sum2)

	for j in range(0,estimated_query_vector.size):
		estimated_query_vector[j]=estimated_query_vector[j]/root1

	tfidf_matrix[0]=sparse.csr_matrix(estimated_query_vector)

	#cosine similarity of documents with queue
	csim1 = np.array(cosine_similarity(tfidf_matrix[0], tfidf_matrix[1:len(corpus)])[0])
	top10=csim.argsort()[-10:][::-1]+1
	docs=""
	for d in top10:
		f = open("./d/"+str(d)+".txt")
		docs=docs+"Doc: "+str(d)+"\n"+f.read()+"\n\n\n"
	return docs
def lab():
	top10=printtext()
	
	T.delete(1.0,tkr.END)  # if you want to remove the old data
	T.insert(1.0,top10)

root = tkr.Tk()
root.attributes("-zoomed", True)
root.title('Name')
e = tkr.Entry(root,width=70)
e.pack()
e.focus_set()
b = tkr.Button(root,text='search',command=lab)
b.pack(side='top')
T = tkr.Text(root, height=400, width=150)
T.pack()
root.mainloop()
