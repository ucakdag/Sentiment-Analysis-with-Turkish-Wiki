###UMUT CAN AKDAG
from wiki_dump_reader import Cleaner, iterate
import time 
from gensim.models import Word2Vec
from tabulate import tabulate
from gensim.models import FastText
import gensim
from nltk import tokenize
import string
import sklearn.ensemble import RandomForestClassifier
cleaner = Cleaner()
w = open("CleanWiki_.txt","w",encoding="utf8")
start = time.time()
for title, text in iterate('tr-wikipedia.xml'):
    text = cleaner.clean_text(text)
    text, links = cleaner.build_links(text)
    cleaned_text = text
    try:
        w.writelines(text+"\n")
    except: pass 

w.close()
def clean(sentence):
    table = str.maketrans("ABCÇDEFGĞHİIJKLMNOÖPRSŞTUÜVYZWXQ","abcçdefgğhiıjklmnoöprsştuüvyzwxq")
    # Check special char
    if "==" not in sentence and  "|" not in sentence and "!" in sentence and "ISBN" not in sentence and  ". Bölüm" not in sentence and "≈" not in sentence and "=" not in sentence and "http" not in sentence:
        sentence_ = sentence.strip().split()
        if sentence_[0] != "InDesign" and sentence_[0] != "Dosya" and sentence_[0] != "Image" and sentence_[0] != "YÖNLENDİRME" and sentence_[0] != "bar:" and sentence_[0] != "TextData=" and      sentence_[0] != "fontsize:" and sentence_[0] != "id" and sentence_[0] != "ImageSize" and sentence_[0] != "PlotArea" and sentence_[0] != "DateFormat" and sentence_[0] != "Period" and sentence_[0] != "TimeAxis" and sentence_[0] != "AlignBars" and sentence_[0] != "ScaleMajor" and sentence_[0] != "ScaleMinor" and sentence_[0] != "BackgroundColors" and sentence_[0] != "BarData" and sentence_[0] != "REDIRECT" and sentence_[0] != "@":
            if len(sentence.split()) > 4:
                # Removing special characters
                sentence = sentence.replace("\n"," ").replace("\t"," ").replace("Hicrî","Hicrî ").replace("Rumi","Rumi ")
                # Lowercase
                sentence = sentence.translate(table)
                return sentence
with open("CleanWiki_.txt","r",encoding="utf8") as f:
    corpus = f.readlines()
corpus_cleaned = []  

for seq in corpus:
    seq = seq.replace("\n", "")
    seq = seq.translate(str.maketrans('', '', string.punctuation))
    corpus_cleaned.append(seq)
word_model = Word2Vec([i.split() for i in corpus_cleaned], vector_size=100, min_count=7, window=5, epochs=3,workers=12)
word_model.save("w2v_.model")
model = Word2Vec.load("w2v_.model")
fasttext_model = FastText([i.split() for i in corpus_cleaned], vector_size=100)
fasttext_model.save("Fasttext.model")
model2 = Word2Vec.load("Fasttext.model")
tfidf_model = gensim.models.TfidfModel([i.split() for i in corpus_cleaned])
tfidf_model.save("TFIDF.model")
from sklearn.feature_extraction.text import CountVectorizer
BoW_Vector = CountVectorizer(min_df = 0., max_df = 1.)
BoW_Matrix = BoW_Vector.fit_transform(corpus_cleaned)
turkStem = TurkishStemmer()
all_stem_lists = []
for word_group in corpus_cleaned:
    output_stems = []
    for word in word_group:
        stem = turkStem.stemWord(word)
        output_stems.append(stem)
    all_stem_lists.append(output_stems)
word_model = Word2Vec([i.split() for i in all_stem_lists], vector_size=100, min_count=7, window=5, epochs=3,workers=12)
word_model.save("w2v_Stemming.model")
RFC=RandomForestClassifier(n_estimators=100)
num_features=100
Vecs=get_avg_feature_vecs(corpus_cleaned, model, num_features)
RFC=RFC.fit(Vecs, corpus_cleaned)
"""
q = "barış"
print(f"\n{q.capitalize()} kelimesine en yakın 10 kelime:\n")
# print(tabulate(model.wv.most_similar(positive=["geliyor","gitmek"],negative=["gelmek"]), headers=["Kelime", "Benzerlik Skoru"]))
print(tabulate(model.wv.most_similar(q), headers=["Kelime", "Benzerlik Skoru"]))
print()
q = "umut"
print(f"\n{q.capitalize()} kelimesine en yakın 10 kelime:\n")
# print(tabulate(model.wv.most_similar(positive=["geliyor","gitmek"],negative=["gelmek"]), headers=["Kelime", "Benzerlik Skoru"]))
print(tabulate(model.wv.most_similar(q), headers=["Kelime", "Benzerlik Skoru"]))
print()
q = "düşman"
print(f"\n{q.capitalize()} kelimesine en yakın 10 kelime:\n")
# print(tabulate(model.wv.most_similar(positive=["geliyor","gitmek"],negative=["gelmek"]), headers=["Kelime", "Benzerlik Skoru"]))
print(tabulate(model.wv.most_similar(q), headers=["Kelime", "Benzerlik Skoru"]))
print()
q = "sevda"
print(f"\n{q.capitalize()} kelimesine en yakın 10 kelime:\n")
# print(tabulate(model.wv.most_similar(positive=["geliyor","gitmek"],negative=["gelmek"]), headers=["Kelime", "Benzerlik Skoru"]))
print(tabulate(model.wv.most_similar(q), headers=["Kelime", "Benzerlik Skoru"]))
print()
q = "açık"
print(f"\n{q.capitalize()} kelimesine en yakın 10 kelime:\n")
# print(tabulate(model.wv.most_similar(positive=["geliyor","gitmek"],negative=["gelmek"]), headers=["Kelime", "Benzerlik Skoru"]))
print(tabulate(model.wv.most_similar(q), headers=["Kelime", "Benzerlik Skoru"]))
print()
print(time.time()- start) 
"""