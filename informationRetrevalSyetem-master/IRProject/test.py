import math
import pandas as pd



# defining term frequency dictionary
term_frequency = dict()
# computing  weighted term frequency Wt,f (which is Log(TF)
weighted_term_frequency = dict()
# compute Inverse document frequency
idf = dict()
# compute document frequency
df = dict()
# compute TF-IDF
tf_idf = dict()
# tf idf document score
documentScore = dict()
numberOfFiles = 10

positional_index = {
    'antony': [3, {1: [0], 2: [0], 6: [0]}],
    'brutus': [3, {1: [1], 2: [1], 4: [0]}],
    'caeser': [5, {1: [2], 2: [2], 4: [1], 5: [0], 6: [1]}],
    'cleopatra': [1, {1: [3]}],
    'mercy': [5, {1: [4], 3: [0], 4: [2], 5: [1], 6: [2]}],
    'worser': [4, {1: [5], 3: [1], 4: [3], 5: [2]}],
    'calpurnia': [1, {2: [3]}],
    'angels': [3, {7: [0], 8: [0], 9: [0]}],
    'fools': [4, {7: [1], 8: [1], 9: [1], 10: [0]}],
    'fear': [3, {7: [2], 8: [2], 10: [1]}],
    'in': [4, {7: [3], 8: [3], 9: [2], 10: [2]}],
    'rush': [4, {7: [4], 8: [4], 9: [3], 10: [3]}],
    'to': [4, {7: [5], 8: [5], 9: [4], 10: [4]}],
    'tread': [4, {7: [6], 8: [6], 9: [5], 10: [5]}],
    'where': [4, {7: [7], 8: [7], 9: [6], 10: [6]}]
}


def termFrequency():  # Term Frequency TF : It's the number of times that a term occurs in a document
    # loop in each term in positional index
    for term in positional_index:
        # create a list contain 0's and it's size = to number of files
        term_frequency[term] = [0 for i in range(numberOfFiles)]
        # foreach term, loop on it's document id which is the key
        for documentId in positional_index[term][1].keys():
            # count the number of occurence of the term in one document
            term_frequency[term][documentId - 1] = len(positional_index[term][1][documentId])

    # for term in term_frequency:
    # print(term, "           ", term_frequency[term])


def logFrequencyWeightForTF():  # log the TF to damp it's effect
    # loop on every term in TF
    for term in term_frequency:
        # for every term as a key create a list of zero's of size = number of file's
        weighted_term_frequency[term] = [0] * numberOfFiles
        i = 0  # will help in iterating over the TF
        # we use this to loop over the list of frequencies in TF
        for frequency in term_frequency[term]:
            # if frequency = 0 or 1 do nothing because log 0 is undefined and (log 1 + 1 = 1)
            if frequency == 0 or frequency == 1:
                weighted_term_frequency[term][i] = frequency
            else:
                # else we add one to the log of frequency to the base 10
                weighted_term_frequency[term][i] = 1 + math.log10(frequency)
            i += 1
    # print(weighted_term_frequency)


def computeDF():
    # use to know how many documents that 1 term is mentioned on it
    for term in positional_index:
        df[term] = positional_index[term][0]


def computeIDF():
    for term in positional_index:
        idf[term] = math.log10(numberOfFiles / df[term])
        # print('idf for the term : ',term,'is ' ,idf[term])


def computeTFxIDF():
    for term in weighted_term_frequency:
        tf_idf[term] = [0] * numberOfFiles
        i = 0
        for value in weighted_term_frequency[term]:
            tf_idf[term][i] = value * idf[term]
            i += 1
        # print(term, tf_idf[term])


termFrequency()
logFrequencyWeightForTF()
computeDF()
computeIDF()
computeTFxIDF()
pd.set_option('display.expand_frame_repr', False)
df = pd.DataFrame(tf_idf,[i for i in range(1,numberOfFiles+1)])
print(tf_idf)
