# import nltk
#
# nltk.download('stopwords')
# nltk.download('punkt')
import pandas as pd
from nltk.corpus import stopwords

import math

# defining positional index dictionary
positional_index = dict()

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
# get Normalized document Length = sqrt(tf_idf ^ 2 ) for all terms in a document
documentLength = dict()
# Normalized TFxIDF = TFxIDF / Document Length
normalized_tf_idf = dict()

# fileID used for detecting which file we work on
fileID = 1
numberOfFiles = 10


# this functions read all txt data from specific folder and assing it to a data list
def reading_files():
    # for i in range(numberOfFolders):
    with open('D:\Python\ir-workingProject\\files/' + str(fileID) + '.txt', 'r') as file2:
        # read file then convert it to lower case then split the string to list
        data = file2.read().lower().split()
    return data


# function that remove stop words except in and to ### work properly ###
def remove_stop_words(data):
    my_stopwords = stopwords.words('english')
    my_lst = ['in', 'to', 'where']
    new_data = list()
    my_stopwords = [el for el in my_stopwords if el not in my_lst]
    for term in data:
        if term not in my_stopwords and term != ' ':
            new_data.append(term)
    return new_data


# trying to make a postion index
def PositionalIndexing(token):
    for pos, term in enumerate(token):  # position value
        # if term isn't exist in  pos_list
        if term not in positional_index:
            # initializing a list
            positional_index[term] = []  # in key we put term name # we assign an empty list in value
            # the frequency of document is one because we have just created it once we find it
            positional_index[term].append(1)
            # initializing the doc_id-positions dictionary
            positional_index[term].append({})
            # make the key = to the file id and the value equal position of the keyword in file
            positional_index[term][1][fileID] = [pos]
        # if the term exist in the index
        else:
            # if the file id exist
            if fileID in positional_index[term][1]:
                positional_index[term][1][fileID].append(pos)
            # if the file id doesn't exist
            else:
                # increase document frequency by 1
                positional_index[term][0] = positional_index[term][0] + 1
                # then append document frequency
                positional_index[term][1][fileID] = [pos]


def PhraseQuery(word):  # phrase => documentID's , positions
    # make empty list of lists in the size of number of files
    semiFinalList = [[] for i in range(numberOfFiles + 1)]  # [[4 , 5], [7, 8] ]
    # for every term in the query
    for w in word:
        # if the term in the keys of positional index do the following
        if w in list(positional_index.keys()):
            # loop in every document id
            for key in list(positional_index[w][1].keys()):  # keys is the fileID'ss
                # if final list in the position of document id is not empty do the following
                if semiFinalList[key] != []:
                    # if the postion last element in final list in a certain file == the position of currnt word in the same file  - 1 then it will match
                    if semiFinalList[key][-1] == positional_index[w][1][key][0] - 1:
                        semiFinalList[key].append(
                            positional_index[w][1][key][0])  # append the position to the final list

                else:  # if the final list is empty then append the position of the keyword in the file
                    semiFinalList[key].append(positional_index[w][1][key][0])
        else:
            continue
    FinalList = dict()
    for document, positions in enumerate(semiFinalList):
        if len(positions) == len(word):
            FinalList[document] = positions[0]
    return FinalList


def termFrequency():  # Term Frequency TF : It's the number of times that a term occurs in a document
    # loop in each term in positional index
    for term in positional_index:
        # create a list contain 0's and it's size = to number of files
        term_frequency[term] = [0 for i in range(numberOfFiles)]  # term_frequency['ahemd'] = [0,0,0,0,0,0,0,0,0]
        # foreach term, loop on its document id which is the key
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
        # print('idf for the term : ', term, 'is ', idf[term])


def computeTFxIDF():
    # for each term in weighted term frequency
    for term in weighted_term_frequency:
        # make a value for each term = to number of files to make sure we have a list with indcies = number of files
        tf_idf[term] = [0] * numberOfFiles
        i = 0  # reset the counter to zero // i explain its use below
        for value in weighted_term_frequency[term]:
            # accessing each item in a list for each term
            tf_idf[term][i] = value * idf[term]
            i += 1  # i used in accessing each item in the list


def DisplayTFxIDF():
    # to show all the table details
    pd.set_option('display.expand_frame_repr', False)
    # to name the table rows before transpose
    df = pd.DataFrame(tf_idf, [('doc ' + str(i)) for i in range(1, numberOfFiles + 1)])
    print(df.transpose())


def computeDocumentLength():
    for i in range(numberOfFiles):
        sum = 0.0
        for term in tf_idf:
            sum += pow(tf_idf[term][i], 2.0)
        documentLength[i + 1] = math.sqrt(sum)
        # print(i+1 ,documentLength[i+1])


def displayDocumentLength():
    for term in documentLength:
        print(term, documentLength[term])


def NormalizedTFxIDF():
    for term in tf_idf:
        normalized_tf_idf[term] = [0] * numberOfFiles
        i = 0
        for value in tf_idf[term]:
            normalized_tf_idf[term][i] = value / documentLength[(i + 1)]
            i += 1
        # print(normalized_tf_idf[term])


def computeCosineSimmilarity(query, matched_docs_id):
    table_data = dict()
    # TF of a query
    query_tf = dict()
    for term in query:
        if term in query_tf.keys():
            query_tf[term] += 1
        else:
            query_tf[term] = 1
    # computer Wtf for a query
    query_wtf = dict()
    query_idf = dict()
    for term in query:
        if query_tf[term] == 1 or query_tf[term] == 0:
            query_wtf[term] = query_tf[term]
        else:
            query_wtf[term] = 1 + math.log10(query_tf[term])

        # get idf for a terms
        query_idf[term] = idf[term]

    # calculate query length
    queryLength = 0.0
    for term in query_wtf:

        queryLength += pow((query_wtf[term] * query_idf[term]), 2)
    queryLength = math.sqrt(queryLength)
    # calculate Normalized Query
    NormalizedQuery = dict()
    for term in query_tf:
        NormalizedQuery[term] = query_wtf[term] * query_idf[term] / queryLength

    # compute simmilarity
    score = dict()
    for docID in matched_docs_id:
        score[docID]=0
        for term in query:
            table_data[term] = {'tf-raw': query_tf[term],'Normalied tf' : query_wtf[term] ,'idf':idf[term] ,'tf-idf':(query_wtf[term] * query_idf[term]) ,'Normalized tf-idf':(query_wtf[term] * query_idf[term] / queryLength)  }
            score[docID] += NormalizedQuery[term] * normalized_tf_idf[term][docID-1]
    table = pd.DataFrame(table_data)
    print(table.transpose())

    df = pd.DataFrame(score,['Simmilarity'])
    print(df.transpose())
    sortedDocs = sorted(score,key=score.get,reverse=True)
    print('Simmilarity of  returned documents in descending order :')
    for doc in sortedDocs:
        print('Doc ' ,doc)




# compute all values needed
def computer():
    termFrequency()
    logFrequencyWeightForTF()
    computeDF()
    computeIDF()
    computeTFxIDF()
    computeDocumentLength()
    NormalizedTFxIDF()


# ================================

for i in range(numberOfFiles):
    data = reading_files()
    data = remove_stop_words(data)
    PositionalIndexing(data)
    fileID += 1
# print(positional_index)

computer()




query =  input('please enter a searching word : ')

searchWord = remove_stop_words(query.lower().split())
matched_docs = PhraseQuery(searchWord)
computeCosineSimmilarity(query.lower().split(), matched_docs)


