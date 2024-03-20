from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
import pandas as pd
from nltk import ne_chunk, pos_tag
from nltk.tokenize import sent_tokenize
from nltk.stem import SnowballStemmer
from nltk.metrics import jaccard_distance
from sklearn.decomposition import TruncatedSVD
from nltk.translate.meteor_score import meteor_score
from nltk.corpus import stopwords


stemmer = SnowballStemmer("english")

# Make sure to download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('maxent_ne_chunker')
nltk.download('averaged_perceptron_tagger')
nltk.download('words')
nltk.download('stopwords')

def read_data_from_csv(csv_file_path):
    """
    Read data from a CSV file and return article and summary columns as lists.
    Args:
        csv_file_path (str): Path to the CSV file.
    Returns:
        Tuple: (article_values, summary_values)
    """
    data = pd.read_csv(csv_file_path, sep=',')  # Read the CSV file

    # Assume the column names are arbitrary and not known in advance
    # You can rename them to 'article' and 'highlights' if they exist, or use default names
    if 'inputs' in data.columns and 'inferences' in data.columns:
        data.rename(columns={'inputs': 'article', 'inferences': 'highlights'}, inplace=True)
    else:
        # Use default column names
        data.columns = ['article', 'highlights']

    article_values = data['article'].tolist()
    summary_values = data['highlights'].tolist()
    return article_values, summary_values
    

def clean_text(text):
    # Tokenize the text into words
    words = word_tokenize(text)
    
    # Remove stop words
    words = [word for word in words if word.lower() not in stopwords.words("english")]
    
    # Apply stemming
    words = [stemmer.stem(word) for word in words]
    
    # Rejoin the cleaned words into a single string
    cleaned_text = " ".join(words)
    cleaned_text = cleaned_text.replace('\n', ' ').replace('\r', ' ')
    
    return cleaned_text    
    
def calculate_score(original_content, summary):
    # Apply data cleaning to the original content and summary
    cleaned_original_content = clean_text(original_content)
    cleaned_summary = clean_text(summary)

    # Factor weights (can be adjusted based on importance)
    factor_weights = {
        'Semantic Similarity': 0.25,
        'Relevance': 0.25,
        'Redundancy': 0.25,
        'Bias Avoidance': 0.25
    }
    # Factor scores (on a scale from 0 to 100)
    factor_scores = {}

    # Factor: Semantic Similarity (LSA + CS)
    lsa_score = semantic_similarity_score(cleaned_original_content, cleaned_summary)
    factor_scores['Semantic Similarity'] = 100 * lsa_score

    # Factor: Relevance (METEOR)
    relevance_score = relevance_similarity(cleaned_original_content, cleaned_summary)
    factor_scores['Relevance'] = 100 * relevance_score

    # Factor: Redundancy (Cosine Similarity)
    redundancy_score = redundancy_analysis(cleaned_summary)
    factor_scores['Redundancy'] = 100 * redundancy_score

    # Factor: Bias Avoidance (NER + Jaccard Similarity)
    bias_avoidance_score = bias_avoidance_analysis(cleaned_original_content, cleaned_summary)
    factor_scores['Bias Avoidance'] = 100 * bias_avoidance_score

    # Calculate overall score as weighted sum of factor scores
    overall_score = sum(factor_scores[factor] * factor_weights[factor] for factor in factor_weights) / 100
    return overall_score

def calculate_average_score(original_contents, summaries):
    total_score = 0
    for original_content, summary in zip(original_contents, summaries):
        score = calculate_score(original_content, summary)
        total_score += score
    average_score = total_score / len(original_contents)
    return average_score

def cosine_similarity_score_for_redundancy(vector1, vector2):
    similarity = cosine_similarity(vector1, vector2)[0][0]
    return similarity

def calculate_cosine_similarity(vector1, vector2):
    similarity = cosine_similarity(vector1.reshape(1, -1), vector2.reshape(1, -1))
    return similarity[0][0]

def semantic_similarity_score(original_content, summary):
    documents = [original_content, summary]

    # Create a document-term matrix using TfidfVectorizer
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(documents)

    # Apply LSA for dimensionality reduction
    num_topics = 2  # Number of topics (latent dimensions)
    lsa = TruncatedSVD(n_components=num_topics)
    X_lsa = lsa.fit_transform(X)

    # Calculate cosine similarity between the original text and its summary
    cosine_similarity_score = calculate_cosine_similarity(X_lsa[0], X_lsa[1])
    lsa_sim = cosine_similarity_score
    return lsa_sim

def relevance_similarity(original_content, summary):
    main_text =  word_tokenize(original_content)
    summary =  word_tokenize(summary)

    # Calculate METEOR score
    meteor = meteor_score([main_text], summary)
    return meteor

def redundancy_analysis(summary):
    sentences = sent_tokenize(summary)

    if len(sentences) == 1:
        return 1.0

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(sentences)
    
    similarity_threshold = 0.5
    redundant_count = 0
    
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            similarity_score = cosine_similarity_score_for_redundancy(vectors[i], vectors[j])
            if similarity_score < similarity_threshold:
                redundant_count += 1
    
    total_pairs = (len(sentences) * (len(sentences) - 1)) / 2
    redundancy_score = (redundant_count / total_pairs)
    return redundancy_score
    
def bias_avoidance_analysis(original_content, summary):
    # Use Named Entity Recognition (NER) to identify named entities in the summary
    ### higher score is better ###
    original_entities = extract_named_entities(original_content)
    ##print('-------------------------------------------------------')
    ##print('original_entities: ',original_entities)
    summary_entities = extract_named_entities(summary)
    ##print('-------------------------------------------------------')
    ##print('summary_entities: ',summary_entities)
    
    # Calculate the Jaccard similarity of named entities to measure bias avoidance
    jaccard_score = calculate_jaccard_similarity(original_entities, summary_entities)
    ###print("bias_avoidance_analysis (NER)", jaccard_score)
    return jaccard_score    

def extract_named_entities(text):
    entities = []
    sentences = nltk.sent_tokenize(text)
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        pos_tags = nltk.pos_tag(words)
        chunked = ne_chunk(pos_tags)
        for subtree in chunked:
            if isinstance(subtree, nltk.Tree):
                entity = " ".join([word for word, tag in subtree.leaves()])
                entities.append(entity)
    return set(entities)

def calculate_jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1) + len(set2) - intersection
    jaccard_similarity = intersection / union if union > 0 else 0
    return jaccard_similarity

average_score = calculate_average_score(article_values, summary_values)
score = calculate_score(original_content, summary)
print(f"Average Score: {average_score:.4f}")    
print(f"Score: {score:.4f}")