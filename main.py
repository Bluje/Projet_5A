import json
import pandas as pd
import numpy as np
import plotly.express as px
import re
import sklearn
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA # Réduction de la dimensionnalité à 2 dimensions avec PCA
stemmer = PorterStemmer()
nltk.download('stopwords')



# Opening JSON file
with open("dataset/test_english2.json", encoding="utf8") as f:
    my_json = json.load(f)
print(my_json)
# Iterating through the json
# list
for i in my_json['items']:
    print(i['title'])


### PRE-PROCESSING

    #function to create a field title + container title (if it exists) + abstract in one variable
def get_full_strings(json_object):
    string_items = []
    items = json_object["items"]
    for item in items:
        title = item.get("title", "")
        container_title = item.get("container-title", "")
        abstract = item.get("abstract", "")


        full_string = title + " " + container_title + " " + abstract
        full_string = full_string.strip()

        print(full_string)
        print("\n\n")

        string_items.append(full_string)

    return string_items

    # function to clean text
def review_to_words(raw_review):
    # 1. Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", raw_review)
    
    # 2. Convert to lower case, split into individual words
    words = letters_only.lower().split()
    
    # 3. Remove Stopwords. In Python, searching a set is much faster than searching
    # a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))
    
    # 4. Remove stop words
    meaningful_words = [w for w in words if not w in stops]  # returns a list
    
    # 5. Stem words. Need to define porter stemmer above
    #singles = [stemmer.stem(word) for word in meaningful_words]
    
    # 6. Join the words back into one string separated by space, and return the result.
    return (" ".join(meaningful_words))

### ALGORITHMS

def calcul_tfidf(text):
    # Creating object TfidfVectorizer
    vectorizer = TfidfVectorizer()

    # Calcul of the matrix TF-IDF
    tfidf_matrix = vectorizer.fit_transform(text)

    # Print terms found in the documents
    print("Terms:", vectorizer.get_feature_names_out())

    # Print matrix TF-IDF
    print("\nMatrix TF-IDF:\n", tfidf_matrix.toarray())
    print("\n")
    
    # Stock of the results in a DataFrame of pandas
    results = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

    # Loop on the documents
    for i in range(len(text)):
        # Select important terms of each document
        top_words = results.iloc[i].sort_values(ascending=False).head(5)
        print("Top 5 words for document {}".format(i + 1))
        print(top_words)
        print("\n")
        
    
    # Number of clusters
    num_clusters = 4
    
    # Initialisation of k-means model
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    
    # Train model k-means with TF-IDF scores 
    kmeans.fit(tfidf_matrix)
    
    # Print centers of clusters
    print("Centers of clusters:\n", kmeans.cluster_centers_)
    print("\n")
    
    # Affecting documents to clusters
    clusters = kmeans.predict(tfidf_matrix)
    
    # Loop on the documents
    for i in range(len(text)):
        print("Document {} belongs to the cluster {}".format(i + 1, clusters[i]))
    
    
    pca = PCA(n_components=2)
    X = pca.fit_transform(tfidf_matrix.toarray())
    
    # Visualisation of clusters
    plt.scatter(X[:,0], X[:,1], c=kmeans.labels_)
    plt.show()
    
    # Visualisation with Plotly
    fig = px.scatter(X, x=0, y=1, color=1, title='Text clustering with K-means and TF-IDF')
    fig.show()

   
def calcul_word2vec(text):
    
    # Creating dictionary to stock documents' informations
    doc_dict = {}
    for i in range(len(text)):
        doc_dict[i] = text[i]
        
    # Training of word2vec model
    model = Word2Vec(text)
    
    # Get vector for each word
    word_vectors = model.wv.vectors
    
    # Clustering with K-means
    num_clusters = 4
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(word_vectors)
    
    # Reducting dimensionality
    pca = PCA(n_components=2)
    pca.fit(word_vectors)
    reduced_vectors = pca.transform(word_vectors)
    
    # Add a column for the group of the document
    group_labels = ['Group ' + str(label) for label in kmeans.labels_]
    
    data_with_groups = [(*reduced_vectors[i], group_labels[i]) for i in range(len(reduced_vectors))]
    
    
    # Visualisation with Matplotlib
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for i in range(len(reduced_vectors)):
        cluster_id = kmeans.labels_[i]
        plt.scatter(reduced_vectors[i, 0], reduced_vectors[i, 1], color=colors[cluster_id])
        plt.annotate("Doc " + str(i) , xy=(reduced_vectors[i, 0], reduced_vectors[i, 1])) #add [+ ": " + " ".join(doc_dict[i])] to print also words

    plt.show()
    
    
    # Visualisation with Plotly
    fig = px.scatter(data_with_groups, x=0, y=1, color=2, title='Text clustering with K-means and Word2Vec')
    fig.show()

   # fig.on_click(on_point_clicked)

    
def on_point_clicked(trace, points, state):
    if points.point_inds:
        point_index = points.point_inds[0]
        group_label = data_with_groups[point_index][2]
        print(f"Le point {point_index} appartient au groupe {group_label}.")



    
    

    # apply it to our text data
    # dataset is named concatened_json and the text are in the column "wmn"
concatened_json = get_full_strings(my_json)
processed_json = [review_to_words(text) for text in concatened_json]
calcul_tfidf(processed_json)
calcul_word2vec(processed_json)




#TODO pouvoir réassocier les variables avec l'article auquel cela correspond



