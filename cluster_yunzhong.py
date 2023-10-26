import json
import pickle
import re
import string
from collections import defaultdict
import matplotlib.pyplot as plt
import nltk
import numpy as np
from googletrans import Translator
from langdetect import detect
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from wordcloud import WordCloud
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs
from sklearn.cluster import SpectralClustering
from sklearn.cluster import Birch
from joblib import Parallel, delayed
"""# Preprocess"""


class Article:
    def __init__(self, index, _id, abstract, abstract_ori, title, title_ori, title_abstract, tokens, label=0):
        self.index = index
        self._id = _id
        self.abstract = abstract
        self.title = title
        self.abstract_ori = abstract_ori
        self.title_ori = title_ori
        self.title_abstract = title_abstract
        self.tokens = tokens


class Cluster:
    def __init__(self, keywords, keywords_counts):
        self.keywords = keywords
        self.keywords_counts = keywords_counts


nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')


def remove_something(sentence):
    text = sentence.replace("<jats:title content-type=\"abstract-subheading\">", ' ')
    text = text.replace("&amp;lt;p&amp;gt;", " ")
    text = text.replace("<jats:sec id=\"sec001\">", " ")
    text = text.replace("<jats:sec>", ' ')
    text = text.replace("</jats:sec>", ' ')
    text = text.replace("<jats:title>", " ")
    text = text.replace("</jats:title>", " ")
    text = text.replace("&#x0D", " ")
    text = text.replace("-", " ")
    text = text.replace("<jats:p>", ' ')
    text = text.replace("</jats:p>", ' ')
    text = text.replace("<title>", ' ')
    text = text.replace("</title>", ' ')
    text = text.replace("<sec>", ' ')
    text = text.replace("</sec>", ' ')
    # reomove digits
    text = re.sub(r'\d+', '', text)
    return text


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def clean_text(text):
    # Initialize the WordNet lemmatizer
    lemmatizer = WordNetLemmatizer()
    text = remove_something(text)
    lemmatized_set = set()
    lemmatized_words = []
    terms_dict = defaultdict(int)
    text = text.lower()
    # remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    # Tokenize the input text into words
    words = nltk.word_tokenize(text)
    tagged_words = nltk.pos_tag(words)
    # Lemmatize each word in the text
    for word, tag in tagged_words:
        if word.isdigit(): continue
        if len(word) <= 3: continue
        pos = get_wordnet_pos(tag)
        word = lemmatizer.lemmatize(word, pos)
        if word in stop_words: continue
        lemmatized_words.append(word)
        terms_dict[word] += 1

    # Join the lemmatized words back into a single text
    lemmatized_set = set(lemmatized_words)
    # cleaned_text = ' '.join(lemmatized_words)

    return lemmatized_words, lemmatized_set, terms_dict


def translate_to_english(text):
    translator = Translator()
    translated_text = translator.translate(text, src='auto', dest='en')
    return translated_text.text

def generate_wordcloud(text):
    # WordCloud object
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)

    # plot word cloud
    plt.figure(figsize=(5, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show(block=False)

def compute_silhouette_score(n_clusters, X):
    birch = Birch(threshold=0.5, n_clusters=n_clusters)
    birch.fit(X)
    labels = birch.labels_
    silhouette_avg = silhouette_score(X, labels)
    return n_clusters, silhouette_avg

if __name__ == '__main__':
    json_file = '/Users/yun/Desktop/Learn/ANU/comp8830/cluster_intern/data/artificial_intelligence.json'
    first_time = False
    select_N = False
    if first_time:
        stop_words_ = set(nltk.corpus.stopwords.words('english'))
        # remove stop words

        stop_words = stop_words_.union(
            {'jats', 'ai', 'artificial', 'intelligence', 'research', 'technology', 'article', 'pattern', 'compared',
             'development', 'paper', 'application', 'method', 'deep', 'algorithm', 'problem', 'model', 'iso',
             'function', 'noise', 'sample', 'layer', 'optimization', 'solution', 'term', 'including', 'factor', 'tool',
             'main', 'experimental', 'optimal', 'parameter', 'predict', 'forecast', 'cluster', 'classifier', 'label',
             'class', 'input', 'world', 'test', 'training', 'case', 'level', 'give', 'focus', 'change', 'current',
             'point', 'important', 'number', 'strategy', 'real', 'however', 'review', 'need', 'various', 'many',
             'solve', 'analyze', 'human', 'significant', 'source', 'include', 'datasets', 'dataset', 'image',
             'approach', 'accuracy', 'information', 'component', 'potential', 'also', 'time', 'show', 'provide', 'task',
             'make', 'existing', 'multiple', 'making', 'activity', 'search', 'performance', 'knowledge', 'user',
             'feature', 'proposed', 'propose', 'year', 'related', 'efficient', 'rate', 'process', 'system', 'well',
             'study', 'form', 'effective', 'example', 'better', 'characteristic', 'support', 'different', 'high',
             'improve', 'design', 'recognition', 'specific', 'finally', 'novel', 'order', 'purpose', 'train', 'compare',
             'field', 'apply', 'develop', 'based', 'content', 'impact', 'possible', 'error', 'presented', 'technique',
             'analysis', 'present', 'work', 'condition', 'implementation', 'evaluation', 'increase', 'target', 'learn',
             'machine', 'language', 'large', 'issue', 'value', 'given', 'higher', 'context', 'scale', 'complex',
             'intelligent', 'base', 'neural', 'network', 'framework', 'learning', 'developed', 'best', 'multi', 'fuzzy',
             'result', 'use', 'supervised', 'representation', 'domain', 'architecture', 'provides', 'obtained', 'like',
             'prediction', 'classification', 'natural', 'conference', 'data', 'identify', 'concept', 'available',
             'world', 'conclusion', 'objective', 'abstract', 'background', 'author', 'disclosure', 'title', 'computer',
             'agent', 'challenge', 'processing', 'future', 'recent', 'transfer', 'state', 'yang', 'evolutionary',
             'program', 'combination', 'vector', 'chapter', 'theory', 'aim', 'addition', 'step', 'cost', 'type', 'part',
             'efficiency', 'complexity', 'comparison', 'set', 'experiment', 'measure', 'quality', 'operation',
             'difference', 'direction', 'practice', 'platform', 'range', 'researcher', 'area', 'stage', 'object',
             'structure', 'decision', 'mean', 'finding', 'response', 'effect', 'group', 'effectiveness', 'size',
             'people', 'way', 'science', 'question', 'role', 'requirement', 'goal', 'implement', 'nan', 'mechanism',
             'modeling', 'second', 'considered', 'space', 'simulation', 'internet', 'standard', 'variable', 'physical',
             'interaction', 'local', 'methodology', 'spatial', 'smart', 'significantly', 'a', 'an', 'the', 'in', 'on',
             'at', 'from', 'to', 'with', 'over', 'under', 'and', 'but', 'or', 'so', 'nor', 'for', 'yet', 'i', 'you',
             'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'is', 'am', 'are', 'was', 'were', 'be',
             'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'shall', 'can', 'could', 'would',
             'should', 'may', 'might', 'must', 'ought', 'that', 'this', 'these', 'those', 'how', 'what', 'why', 'by',
             'their', 'of', 'as', 'there', 'within', 'which'})


        print('stop words created')
        with open(json_file, 'r') as file:
            data = json.load(file)

            # preprocessing
            training_texts = []
            training_texts_title = []
            lemmatized_words_all = []
            article_ls = []
            training_texts_title_abstract = []
            words = set()
            terms_dict_all = defaultdict(int)
            # print(data[0])
            k = 0

            # for item in data:
            for article in data:
                if "abstract" in article.keys() and "title" in article.keys():
                    text_title_abstract = article['abstract'] + " " + article['title'][0]
                    # text_title_abstract_ori = text_title_abstract
                    # language = detect(text_title_abstract)
                    # if language != "en":
                    #     continue
                    #     # print("Before translate: " + text_title_abstract)
                    #     if len(text_title_abstract) >= 5000:
                    #         text_title_abstract = text_title_abstract[:5000]
                    #
                    #     text_title_abstract = translate_to_english(text_title_abstract)
                    #     # print("After translate: " + text_title_abstract)
                    lemmatized_words, lemmatized_set, terms_dict = clean_text(text_title_abstract)
                    words.update(lemmatized_set)
                    terms_dict_all.update(terms_dict)
                    lemmatized_words_all.append(lemmatized_words)


                    data_article = Article(index=k, _id=article["_id"], abstract='',
                                           abstract_ori=article['abstract'], title='',
                                           title_ori=article['title'][0], title_abstract='', tokens=[])
                    article_ls.append(data_article)

                k += 1
            print(len(article_ls))
            avoiding_words = set(sorted(terms_dict, key=lambda x: terms_dict[x], reverse=True)[:50])  # high frequent words
            low_frequency_terms = set([term for term in terms_dict if terms_dict[term] == 1])  # low frequent words
            avoiding_words |= low_frequency_terms
            training_texts_title_abstract = [' '.join(term for term in terms_list if term not in avoiding_words) for terms_list
                                             in lemmatized_words_all]
            for i, article in enumerate(article_ls):
                tokens = nltk.word_tokenize(training_texts_title_abstract[i])
                article.tokens = tokens
                article.title_abstract = training_texts_title_abstract[i]

            print('preprocessing finished')
        """# Text Embedding"""


        # Load the pre-trained model

        model_name = 'all-MiniLM-L6-v2'
        model = SentenceTransformer(model_name)

        final_sentence_embedding = model.encode(training_texts_title_abstract, show_progress_bar=True)
        print('embedding finished')
        """# Save & load"""
        # save article_ls
        with open('../data/list_article.pkl', 'wb') as f:
            pickle.dump(article_ls, f)
        # save embedding
        np.save('../data/embedding.npy', final_sentence_embedding)
        # save training_texts_title_abstract
        with open('../data/list_title_abstract.pkl', 'wb') as f_:
            pickle.dump(training_texts_title_abstract, f_)
    else:
        # load article_ls
        with open('/Users/yun/Desktop/Learn/ANU/comp8830/codes/list_article.pkl', 'rb') as f:
            article_ls = pickle.load(f)

        # load embedding
        final_sentence_embedding = np.load('/Users/yun/Desktop/Learn/ANU/comp8830/codes/embedding.npy')

        # load training_texts_title_abstract
        with open('/Users/yun/Desktop/Learn/ANU/comp8830/codes/list_title_abstract.pkl', 'rb') as f_:
            training_texts_title_abstract = pickle.load(f_)

        print('load finished')
    """# dimensional reduction"""

    from sklearn.decomposition import PCA
    print(len(final_sentence_embedding))
    num_components = 100
    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=num_components)
    X = pca.fit_transform(final_sentence_embedding)
    # X = final_sentence_embedding
    del final_sentence_embedding
    print('dimensional reduction finished')
    """# GMM"""

    from sklearn.metrics import silhouette_score
    if select_N:
        silhouette_scores = []
        max_components = 24
        for n_clusters in range(10, max_components, 2):
            birch = Birch(threshold=0.5, n_clusters=n_clusters)
            # Fit the model
            birch.fit(X)
            # Get cluster labels
            labels = birch.labels_
            silhouette_avg = silhouette_score(X, labels)
            silhouette_scores.append(silhouette_avg)
        # Plot the silhouette scores
        cluster_range = list(range(10, max_components, 2))
        plt.plot(cluster_range, silhouette_scores, marker='o')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Analysis for GMM Clustering')
        plt.show(block=True)

       # Find the best K based on the highest silhouette score
        best_k = cluster_range[np.argmax(silhouette_scores)]
        print("Best number of clusters (K):", best_k)


    else:
        # trainging
        n_components = 22
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        labels = gmm.fit_predict(X)

        # Create a BIRCH clustering model
        # birch = Birch(threshold=0.5, n_clusters=12)
        #
        # # Fit the model
        # birch.fit(X)
        #
        # # Get cluster labels
        # labels = birch.labels_

        print('training finished')
        my_dict = defaultdict(list)
        for index, label_ in enumerate(labels):
            article_ls[index].label = label_
            my_dict[label_].append(article_ls[index])

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(training_texts_title_abstract)
        # 打印每个簇的文本和关键词
        for cluster_id in my_dict.keys():
            cluster_tfidf_matrix = tfidf_matrix[np.array(labels) == cluster_id]
            cluster_tfidf_scores = cluster_tfidf_matrix.mean(axis=0).A1
            top_tfidf_indices = cluster_tfidf_scores.argsort()[-10:][::-1]
            top_keywords = [vectorizer.get_feature_names_out()[i] for i in top_tfidf_indices]
            print(f"Cluster {cluster_id + 1} Keywords: {', '.join(top_keywords)}\n")
        print('visualize keywords finished')
        # visualize clustering result

        """# Output clustering result"""

        dict_label_id = defaultdict(list)
        for key in my_dict.keys():
            print(key)
            for item in my_dict[key]:
                dict_label_id["Cluster " + str(key)].append(item._id)
        file_path = "./result.json"
        with open(file_path, 'w') as file:
            json.dump(dict_label_id, file)
        print('output finished')

        """# Visualization"""

        dict_label_title_abstract = defaultdict(list)
        dict_label_tokens = defaultdict(list)
        for key in my_dict.keys():
            for article in my_dict[key]:
                dict_label_title_abstract[key].append(article.title_abstract)
                dict_label_tokens[key].extend(article.tokens)

        for key in my_dict.keys():
            print(len(dict_label_tokens[key]))

        for key in my_dict.keys():
            print("cluster " + str(key + 1) + " with " + str(len(my_dict[key])) + "articles")
        print('visualization paper distribution finished')


        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(training_texts_title_abstract)
        dict_label_keywords_count = defaultdict(lambda: defaultdict(int))
        for cluster_id in my_dict.keys():
            cluster_tfidf_matrix = tfidf_matrix[np.array(labels) == cluster_id]
            cluster_tfidf_scores = cluster_tfidf_matrix.mean(axis=0).A1
            top_tfidf_indices = cluster_tfidf_scores.argsort()[-10:][::-1]
            top_keywords = [vectorizer.get_feature_names_out()[i] for i in top_tfidf_indices]
            # print(top_keywords)
            for keyword in top_keywords:
                for word in dict_label_tokens[cluster_id]:
                    if word == keyword:
                        dict_label_keywords_count[cluster_id][keyword] += 1
            print(f"Cluster {cluster_id + 1} Keywords: {', '.join(top_keywords)}\n")

        print("visualize keywords finished")

        for label in dict_label_keywords_count.keys():
            keywords = []
            keyword_counts = []
            for keyword in dict_label_keywords_count[label].keys():
                keywords.append(keyword)
                keyword_counts.append(dict_label_keywords_count[label][keyword])
            plt.clf()
            plt.barh(keywords, keyword_counts)
            plt.xlabel('Number of Occurrences')
            plt.ylabel('Keywords')
            plt.title(f'Keyword Occurrences - Cluster {label + 1}')
            plt.tight_layout()
            plt.savefig('plot'+str(label+1)+'.png')
            plt.show(block=False)
            plt.close()

        print('visualization keywords occurences finished')

        # Adjust layout and display the plots


