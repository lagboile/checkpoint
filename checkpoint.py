import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('french'))


def preprocess(text):

    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])

    words = word_tokenize(text)

    words = [word for word in words if word not in stop_words]

    if not words:
        return ""

    words = [lemmatizer.lemmatize(word) for word in words]

    return ' '.join(words)


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize


def get_most_relevant_sentence(query, text_data):
    sentences = sent_tokenize(text_data, language='french')
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

    if not sentences:
        return "le texte foruni est vide ou mal formaté."
    if not query.strip():
        return "Votre requete est vide apres prétraitement."

    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform([query] + sentences)
        similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])

        most_similar_index = similarity_matrix.argmax()
        return sentences[most_similar_index]

    except ValueError:
        return "Erreur: impossible de vectoriser les données(texte vide ou mal formaté)."


def chatbot(user_query, text_data):

    user_query = preprocess(user_query)
    print("Requête prétraitée :", user_query)

    if not user_query.strip():
        return "Votre question ne contient pas assez d'informations."

    relevant_sentence = get_most_relevant_sentence(user_query, text_data)

    return relevant_sentence

import json
import streamlit as st

def main():
    st.title("Chatbot Bancaire")

    try:
        with open('C:\\Users\\LENOVO\\checkpoint\\file_client_js.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        st.write("erreur lors du chargement du fichier JSON :",e)
        return

    text_data = data.get("key_with_sentences","")

    if isinstance(text_data, list):
        text_data = ' '.join(json.dumps(item) if isinstance(item, dict)
        else str(item) for item in text_data)

    if not text_data.strip():
        st.write("le texte extrait du Json est vide ou inexistant.")
        return

    user_query = st.text_input("Posez votre question sur les services bancaires :")

    if user_query:
        response = chatbot(user_query, text_data)
        st.write("Réponse du chatbot :")
        st.write(response)

if __name__ == "__main__":
    main()

