import os
import re
import nltk
import gensim
import numpy
import pdfplumber
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.parsing.preprocessing import remove_stopwords
from gensim.models import Word2Vec
from gensim import corpora
from sklearn.metrics.pairwise import cosine_similarity
import gensim.downloader as api

nltk.download('stopwords')
nltk.download('punkt')

def pdf_extract(file_name, directory="docs"):
    pdf_txt = ""
    if not os.path.exists(directory):
        raise FileNotFoundError(f"The directory {directory} does not exist.")
    file_path = os.path.join(directory, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_name} does not exist in the directory {directory}.")
    
    with pdfplumber.open(file_path) as pdf:
        for pdf_page in pdf.pages:
            single_page_text = pdf_page.extract_text()
            pdf_txt += single_page_text
    return pdf_txt

def clean_sentence(sentence, stopwords=False):
    sentence = sentence.lower().strip()
    sentence = re.sub(r'[^a-z0-9\s]', '', sentence)
    if stopwords:
        sentence = remove_stopwords(sentence)
    return sentence

def get_cleaned_sentences(tokens, stopwords=False):
    cleaned_sentences = []
    for row in tokens:
        cleaned = clean_sentence(row, stopwords)
        cleaned_sentences.append(cleaned)
    return cleaned_sentences

def retrieveAndPrintFAQAnswer(question_embedding, sentence_embeddings, sentences):
    max_sim = -1
    index_sim = -1
    for index, embedding in enumerate(sentence_embeddings):
        sim = cosine_similarity(embedding, question_embedding)[0][0]
        if sim > max_sim:
            max_sim = sim
            index_sim = index
    return index_sim

def naive_drive(file_name, question, directory="docs"):
    pdf_txt = pdf_extract(file_name, directory)
    tokens = nltk.sent_tokenize(pdf_txt)
    cleaned_sentences = get_cleaned_sentences(tokens, stopwords=True)
    cleaned_sentences_with_stopwords = get_cleaned_sentences(tokens, stopwords=False)
    sentences = cleaned_sentences_with_stopwords
    sentence_words = [[word for word in document.split()] for document in sentences]

    dictionary = corpora.Dictionary(sentence_words)
    bow_corpus = [dictionary.doc2bow(text) for text in sentence_words]

    question = clean_sentence(question, stopwords=False)
    question_embedding = dictionary.doc2bow(question.split())

    index = retrieveAndPrintFAQAnswer(question_embedding, bow_corpus, sentences)
    return sentences[index]

def getWordVec(word, model):
    try:
        vec = model[word]
    except KeyError:
        vec = [0] * len(model['pc'])
    return vec

def getPhraseEmbedding(phrase, embeddingmodel):
    vec = numpy.array([0] * len(embeddingmodel['pc']))
    den = 0
    for word in phrase.split():
        den += 1
        vec = vec + numpy.array(getWordVec(word, embeddingmodel))
    return vec.reshape(1, -1)

def word2vec_drive(file_name, question, directory="docs"):
    pdf_txt = pdf_extract(file_name, directory)
    tokens = nltk.sent_tokenize(pdf_txt)
    cleaned_sentences = get_cleaned_sentences(tokens, stopwords=True)
    cleaned_sentences_with_stopwords = get_cleaned_sentences(tokens, stopwords=False)
    sentences = cleaned_sentences_with_stopwords

    sent_embeddings = []
    for sent in sentences:
        sent_embeddings.append(getPhraseEmbedding(sent, v2w_model))

    question_embedding = getPhraseEmbedding(question, v2w_model)
    index = retrieveAndPrintFAQAnswer(question_embedding, sent_embeddings, cleaned_sentences_with_stopwords)
    return cleaned_sentences_with_stopwords[index]

def glove_drive(file_name, question, directory="docs"):
    pdf_txt = pdf_extract(file_name, directory)
    tokens = nltk.sent_tokenize(pdf_txt)
    cleaned_sentences = get_cleaned_sentences(tokens, stopwords=True)
    cleaned_sentences_with_stopwords = get_cleaned_sentences(tokens, stopwords=False)
    sentences = cleaned_sentences_with_stopwords

    sent_embeddings = []
    for sent in cleaned_sentences:
        sent_embeddings.append(getPhraseEmbedding(sent, glove_model))

    question_embedding = getPhraseEmbedding(question, glove_model)
    index = retrieveAndPrintFAQAnswer(question_embedding, sent_embeddings, cleaned_sentences_with_stopwords)
    return cleaned_sentences_with_stopwords[index]

# Load models
v2w_model = None
try:
    v2w_model = gensim.models.KeyedVectors.load('./w2vecmodel.mod')
    print("w2v Model Successfully loaded")
except:
    v2w_model = api.load('word2vec-google-news-300')
    v2w_model.save("./w2vecmodel.mod")
    print("w2v Model Saved")

glove_model = None
try:
    glove_model = gensim.models.KeyedVectors.load('./glovemodel.mod')
    print("Glove Model Successfully loaded")
except:
    glove_model = api.load('glove-twitter-25')
    glove_model.save("./glovemodel.mod")
    print("Glove Model Saved")

# Example usage
if __name__ == "__main__":
    question = "Explain the free cash flow situation?"
    try:
        answer = glove_drive('AMZN-Q1-2020-Earnings-Release.pdf', question)
        print(answer)
    except FileNotFoundError as e:
        print(e)
