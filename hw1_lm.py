########################################
## CS447 Natural Language Processing  ##
##           Homework 1               ##
##       Julia Hockenmaier            ##
##       juliahmr@illnois.edu         ##
########################################
##
## Part 1:
## Develop a smoothed n-gram language model and evaluate it on a corpus
##
from __future__ import division
import os.path
import sys
import random
import math
from operator import itemgetter
from collections import defaultdict

#----------------------------------------
#  Data input 
#----------------------------------------

# Read a text file into a corpus (list of sentences (which in turn are lists of words))
# (taken from nested section of HW0)
def readFileToCorpus(f):
    """ Reads in the text file f which contains one sentence per line.
    """
    if os.path.isfile(f):
        file = open(f, "r") # open the input file in read-only mode
        i = 0 # this is just a counter to keep track of the sentence numbers
        corpus = [] # this will become a list of sentences
        print "Reading file ", f
        for line in file:
            i += 1
            sentence = line.split() # split the line into a list of words
            corpus.append(sentence) # append this list as an element to the list of sentences
            if i % 1000 == 0:
                sys.stderr.write("Reading sentence " + str(i) + "\n") # just a status message: str(i) turns the integer i into a string, so that we can concatenate it
        return corpus
    else:
        print "Error: corpus file ", f, " does not exist"  # We should really be throwing an exception here, but for simplicity's sake, this will suffice.
        sys.exit() # exit the script

# Preprocess the corpus to help avoid sparsity issues
def preprocess(corpus):
    print "Task 0: edit the preprocess function to replace rare words with UNK and add sentence markers"
    counts = {}
    unkWords = {}
    for i in range(len(corpus)):
        corpus[i].append(end)
        corpus[i].insert(0,start)
        for j in range(1,len(corpus[i])-1):
            word = corpus[i][j]
            if(word in counts):
                counts[word] += 1
                unkWords.pop(word,None)
            else:
                counts[word] = 1
                unkWords[word] = (i,j)
	
    for word,position in unkWords.iteritems():
        corpus[position[0]][position[1]] = UNK

    return corpus
	
def getVocab(corpus):
    vocabulary = []
    for sentenceList in corpus:
        vocabulary.extend(sentenceList)
    sortedVocab = list(set(vocabulary))
    sortedVocab.remove(start)
    return sortedVocab
    
def preprocess_testcorpus(corpus, vocab):
    for i in range(len(corpus)):
        corpus[i].append(end)
        corpus[i].insert(0,start)
        for j in range(1,len(corpus[i])-1):
            if(corpus[i][j] not in vocab):
                corpus[i][j] = UNK

    return corpus



   	
# Constants 
UNK = "UNK"     # Unknown word token
start = "<s>"   # Start-of-sentence token
end = "</s>"    # End-of-sentence-token


#--------------------------------------------------------------
# Language models and data structures
#--------------------------------------------------------------

# Parent class for the three language models you need to implement
class LanguageModel:
    # Initialize and train the model (ie, estimate the model's underlying probability
    # distribution from the training corpus)
    def __init__(self, corpus):
        print """Your task is to implement three kinds of n-gram language models:  
      a) an (unsmoothed) unigram model (UnigramModel)
      b) an unsmoothed bigram model (BigramModel)
      c) a bigram model smoothed using absolute discounting (SmoothedBigramModel)"""

    # Generate a sentence by drawing words according to the model's probability distribution
    # Note: think about how to set the length of the sentence in a principled way
    def generateSentence(self):
        print "Implement the generateSentence method in each subclass"
        return "mary had a little lamb ."

    # Given a sentence (sen), return the probability of that sentence under the model
    def getSentenceProbability(self, sen):
        print "Implement the getSentenceProbability method in each subclass"
        return 0.0

    # Given a corpus, calculate and return its perplexity (normalized inverse log probability)
    def getCorpusPerplexity(self, corpus):
        print "Implement the getCorpusPerplexity method"
        return 0.0

    # Given a file (filename) and the number of sentences, generate a list
    # of sentences and write each to file along with its model probability.
    # Note: you shouldn't need to change this method
    def generateSentencesToFile(self, numberOfSentences, filename):
        file=open(filename, 'w+')
        for i in range(0,numberOfSentences):
            sen = self.generateSentence()
            while(len(sen.split()) == 2):
                sen = self.generateSentence()
            prob = self.getSentenceProbability(sen)
            print >>file, prob, " ", sen

# Unigram language model
class UnigramModel(LanguageModel):
    def __init__(self, corpus):
        self.unigramDist = UnigramDist(corpus)
		
    def generateSentence(self):
        sentence = [start]
        sentenceEnd = False
        while(not sentenceEnd):
            word = self.unigramDist.draw()
            sentence.append(word)
            if word == end:
                sentenceEnd = True
        return " ".join(sentence)

    def getCorpusPerplexity(self, corpus):
        perp = 0.0
        N = 0.0
        for sent in corpus:
            for word in sent:
                if word == start:
                    continue
                perp += self.unigramDist.prob(word)
                N += 1
        perp =  -perp / N
        perp = math.exp(perp)
        return perp
            
	# Given a sentence (sen), return the probability of that sentence under the model
    def getSentenceProbability(self, sen):
        prob = 0.0
        sen = sen.split()
        for i in range(1,len(sen)):
            prob += self.unigramDist.prob(sen[i])     
        return prob

# Unsmoothed bigram language model
class BigramModel(LanguageModel):
    def __init__(self, corpus):
        self.bigramDist = BigramDist(corpus)
    
    def generateSentence(self):
        sentence = [start]
        sentenceEnd = False
        givenword = start
        while(not sentenceEnd):
            word = self.bigramDist.draw(givenword)
            sentence.append(word)
            givenword = word
            if word == end:
                sentenceEnd = True
        return " ".join(sentence)

    def getSentenceProbability(self, sen):
        sen = sen.split()
        prob = 0.0
        for i in range(1,len(sen)):
            prob += self.bigramDist.prob(sen[i],sen[i-1]) 
        return prob

# Smoothed bigram language model (use absolute discounting for smoothing)
class SmoothedBigramModel(LanguageModel):
    def __init__(self, corpus):
        self.smoothedDist = SmoothedBigramDist(corpus)
    
    def generateSentence(self):
        sentence = [start]
        sentenceEnd = False
        givenword = start
        while(not sentenceEnd):
            word = self.smoothedDist.draw(givenword)
            sentence.append(word)
            givenword = word
            if word == end:
                sentenceEnd = True
        return " ".join(sentence)

    def getSentenceProbability(self, sen):
        sen = sen.split()
        prob = 0.0
        for i in range(1,len(sen)):
            prob += self.smoothedDist.prob(sen[i],sen[i-1]) 
        return prob  
        
    def getCorpusPerplexity(self, corpus):
        perp = 0.0
        N = 0.0
        for i in range(len(corpus)):
            for j in range(1,len(corpus[i])):
                perp += self.smoothedDist.prob(corpus[i][j],corpus[i][j-1])
                N += 1
        perp =  -perp / N
        perp = math.exp(perp)
        return perp

# Sample class for a unsmoothed unigram probability distribution
# Note: 
#       Feel free to use/re-use/modify this class as necessary for your 
#       own code (e.g. converting to log probabilities after training). 
#       This class is intended to help you get started
#       with your implementation of the language models above.
class UnigramDist:
    def __init__(self, corpus):
        self.counts = defaultdict(float)
        self.total = 0.0
        self.train(corpus)

    # Add observed counts from corpus to the distribution
    def train(self, corpus):
        for sen in corpus:
            for word in sen:
                if word == start:
                    continue
                self.counts[word] += 1.0
                self.total += 1.0

    # Returns the probability of word in the distribution
    def prob(self, word):
        return math.log(self.counts[word]/self.total)

    # Generate a single random word according to the distribution
    def draw(self):
        rand = self.total*random.random()
        for word in self.counts:
            rand -= self.counts[word]
            if rand <= 0.0:
                return word
# End sample unigram dist code

class BigramDist:
    def __init__(self, corpus):
        self.bicounts = defaultdict(float)
        self.unicounts = defaultdict(float)
        self.total = 0.0
        self.train(corpus)

    def train(self, corpus):
        for i in range(len(corpus)):
            for j in range(len(corpus[i])):
                if j!=0:
                    self.bicounts[(corpus[i][j],corpus[i][j-1])] += 1.0
                self.unicounts[corpus[i][j]] += 1.0
                self.total += 1.0

    def prob(self, word, givenword):
        return math.log(self.bicounts[word,givenword]/self.unicounts[givenword])

    def draw(self,givenword):
        rand = self.unicounts[givenword]*random.random()
        for word1,word2 in self.bicounts:
            if word2 == givenword:
                rand -= self.bicounts[word1,word2]
                if rand <= 0.0:
                    return word1
        #
                    
class SmoothedBigramDist:
    def __init__(self, corpus):
        self.bicounts = defaultdict(float)
        self.unicounts = defaultdict(float)
        self.total = 0.0
        self.D = 0.0
        self.Sw = defaultdict(float)
        self.unigramDist = UnigramDist(corpus)
        self.train(corpus)
        self.setD(self.bicounts)
        self.setSw(self.bicounts)

    def train(self, corpus):
        for i in range(len(corpus)):
            for j in range(len(corpus[i])):
                if j!=0:
                    self.bicounts[(corpus[i][j],corpus[i][j-1])] += 1.0
                self.unicounts[corpus[i][j]] += 1.0
                self.total +=1 
                
    def setD(self, bicounts):
        n1 = 0.0
        n2 = 0.0
        for key,val in bicounts.iteritems():
            if val == 1:
                n1 += 1
            elif val == 2:
                n2 += 1
        self.D = n1/(n1+2*n2)
            
    def setSw(self, bicounts):
        for word,prevWord in bicounts:
            self.Sw[prevWord] += 1
                
    
    def prob(self, word, givenword):
        prob = max((self.bicounts[word,givenword] - self.D), 0)/self.unicounts[givenword]
        temp = math.exp(self.unigramDist.prob(word))
        prob += self.D * self.Sw[givenword] * temp / self.unicounts[givenword]
        return math.log(prob)
        
    def getSmoothedCounts(self, word, givenword):
        return max((self.bicounts[word,givenword] - self.D), 0) + self.D * self.Sw[givenword] * math.exp(self.unigramDist.prob(word))

    def draw(self,givenword):
        rand = self.unicounts[givenword]*random.random()
        for word in self.unicounts:
            if word == start:
                continue
            rand -= self.getSmoothedCounts(word,givenword)
            if rand <= 0.0:
                return word
        
        

#-------------------------------------------
# The main routine
#-------------------------------------------
if __name__ == "__main__":
    trainCorpus = readFileToCorpus('train.txt')
    trainCorpus = preprocess(trainCorpus)
    
    # Run sample unigram dist code
    unigramDist = UnigramDist(trainCorpus)
    print "Sample UnigramDist output:"
    print "Probability of \"vader\": ", unigramDist.prob("vader")
    print "Probability of \""+UNK+"\": ", unigramDist.prob(UNK)
    print "\"Random\" draw: ", unigramDist.draw()
    # Sample test run for unigram model
    unigram = UnigramModel(trainCorpus)
    # Task 1   (*** remember to generate 20 sentences for final output ***)
    unigram.generateSentencesToFile(20, "unigram_output.txt")
    # Task 2
    posTestCorpus = readFileToCorpus('pos_test.txt')
    negTestCorpus = readFileToCorpus('neg_test.txt')
    vocab = getVocab(trainCorpus)
    posTestCorpus = preprocess_testcorpus(posTestCorpus, vocab)
    negTestCorpus = preprocess_testcorpus(negTestCorpus, vocab)
    trainPerp = unigram.getCorpusPerplexity(trainCorpus)
    posPerp = unigram.getCorpusPerplexity(posTestCorpus)
    negPerp = unigram.getCorpusPerplexity(negTestCorpus)   
    print "Perplexity of positive training corpus:    "+ str(trainPerp) 
    print "Perplexity of positive review test corpus: "+ str(posPerp)
    print "Perplexity of negative review test corpus: "+ str(negPerp)
     
    ## Run sample bigram dist code
    bigramDist = BigramDist(trainCorpus)
    bigram = BigramModel(trainCorpus)
    print "Sample BigramDist output:"
    print "Probability of \"darth vader\": ", bigramDist.prob("vader","darth")
    print "\"Random\" draw for a word appearing after 'and': ", bigramDist.draw("and")
    # Task 1   (*** remember to generate 20 sentences for final output ***)
    bigram.generateSentencesToFile(20, "bigram_output.txt")
    
    ## Run sample smoothed bigram dist code
    smoothedBigramDist = SmoothedBigramDist(trainCorpus)
    smoothedBigram = SmoothedBigramModel(trainCorpus)
    print "Sample BigramDist output:"
    print "Probability of \"darth vader\": ", smoothedBigramDist.prob("vader","darth")
    print "\"Random\" draw for a word appearing after 'and': ", smoothedBigramDist.draw("and")
    
    # Task 2
    posTestCorpus = readFileToCorpus('pos_test.txt')
    negTestCorpus = readFileToCorpus('neg_test.txt')
    vocab = getVocab(trainCorpus)
    posTestCorpus = preprocess_testcorpus(posTestCorpus, vocab)
    negTestCorpus = preprocess_testcorpus(negTestCorpus, vocab)
    trainPerp = smoothedBigram.getCorpusPerplexity(trainCorpus)
    posPerp = smoothedBigram.getCorpusPerplexity(posTestCorpus)
    negPerp = smoothedBigram.getCorpusPerplexity(negTestCorpus)   
    print "Perplexity of positive training corpus:    "+ str(trainPerp) 
    print "Perplexity of positive review test corpus: "+ str(posPerp)
    print "Perplexity of negative review test corpus: "+ str(negPerp)

