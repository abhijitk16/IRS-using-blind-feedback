from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import average_precision_score
import numpy as np
import nltk
from nltk.stem import PorterStemmer
import nimfa
from sklearn import preprocessing
from scipy import sparse

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
    cscore = average_precision_score(truearr, csim, average='micro')
    mapscore=mapscore+cscore


#average p@10 for all the queries
p10=p10/q

#average p@10 for all the queries
p20=p20/q

#mean average score
mapscore=mapscore/q

#print ("MAP = " + str(mapscore) + " p@10 = " + str(p10) + " p@20 = " + str(p20) + "\n")