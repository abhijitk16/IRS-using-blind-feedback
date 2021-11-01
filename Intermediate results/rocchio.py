from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import average_precision_score
import numpy as np
import nltk
from nltk.stem import PorterStemmer
import nimfa
from sklearn import preprocessing
from scipy import sparse
import itertools

english_stemmer = PorterStemmer()
class StemmedTfidfVectorizer(TfidfVectorizer):
	def build_analyzer(self):
		analyzer = super(TfidfVectorizer, self).build_analyzer()
		return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))
p10=0.0
p20=0.0
corpus = []
mapscore=0.0
# prepare corpus and relevance
f = open("./q/"+str(1)+".txt")
corpus.append(f.read)
for d in range(0,1400):
	f = open("./d/"+str(d+1)+".txt")
	corpus.append((f.read()))

for q in range(1, 226):
	
	relevance = ""
	
	# add query to corpus
	f = open("./q/"+str(q)+".txt")
	corpus[0]=(f.read())
	# relevance
	f = open("./r/"+str(q)+".txt")
	relevance = np.array(list(map(int, filter(None, f.read().split("\n")))))

	#creating tfidf_matrix
	tfidf_vectorizer = StemmedTfidfVectorizer(min_df=1, stop_words='english', analyzer='word', ngram_range=(1,1))
	tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
	
	#binary array of relevant documents
	truearr=[0]*1400
	sizenow=len(relevance)
	for temper in range(0,sizenow):
		truearr[relevance[temper]-1]=1

	# Cosine similarity measure
	csim = np.array(cosine_similarity(tfidf_matrix[0], tfidf_matrix[1:len(corpus)])[0])
	#using 10 feedback documents
	x=10

	#setting rocchio parmeters
	alpha=0.4
	beta=0.7
	gamma=1.0

	#Rocchio algorithm
	arr1=np.array(csim.argsort()[-x:][::-1])+1
	siz=tfidf_matrix[0].toarray().size
	arrRel=[0.0]*siz
	arrQue=[0.0]*siz
	arrnotRel=[0.0]*siz
	n = tfidf_matrix.tocoo()
	sum1=0.0
	for i,j,v in itertools.izip(n.row, n.col, n.data):
		if i==0:
			arrQue[j]=arrQue[j]+v
		if i in arr1:
			arrRel[j]=arrRel[j]+v
		else:
			arrnotRel[j]=arrnotRel[j]+v
	for j in range(0,siz):
		arrRel[j]=(arrRel[j]*alpha)/arr1.size
		arrnotRel[j]=(arrnotRel[j]*beta)/(csim.size-arr1.size)
		arrQue[j]=arrQue[j]*gamma + arrRel[j] + arrnotRel[j]
		sum1=sum1+arrQue[j]*arrQue[j]

	root=np.sqrt(sum1)

	#normalization
	for j in range(0,siz):
		arrQue[j]=arrQue[j]/root
	tfidf_matrix[0]=sparse.csr_matrix(arrQue)

	#cosine similarity of documents with queue
	csim1 = np.array(cosine_similarity(tfidf_matrix[0], tfidf_matrix[1:len(corpus)])[0])

	#calculating p@10 and p@20
	count=0.0
	count1=0.0
	top10=csim.argsort()[-10:][::-1]+1
	top20=csim.argsort()[-20:][::-1]+1
	for x in range(0,10):
		for y in range(0,relevance.size):
			if top10[x]==relevance[y]:
				count=count+1
	for x in range(0,20):
		for y in range(0,relevance.size):
			if top20[x]==relevance[y]:
				count1=count1+1

	count=count/10.0
	count1=count1/20.0
	p10=p10+count
	p20=p20+count1

	#average precision
	cscore = average_precision_score(truearr, csim1, average='micro')
	mapscore=mapscore+cscore

#average p@10 for all the queries
p10=p10/q

#average p@10 for all the queries
p20=p20/q

#mean average score
mapscore=mapscore/q

#print ("MAP = " + str(mapscore) + " p@10 = " + str(p10) + " p@20 = " + str(p20) + "\n")