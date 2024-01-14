import string
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from collections import Counter

def perform_sentiment_analysis(text):
    lower_case = text.lower()
    cleaned_text = lower_case.translate(str.maketrans('', '', string.punctuation))

    tokenized_words = word_tokenize(cleaned_text, "english")

    final_words = [word for word in tokenized_words if word not in stopwords.words('english')]

    # Lemmatization - From plural to single + Base form of a word (example better-> good)
    lemma_words = [WordNetLemmatizer().lemmatize(word) for word in final_words]

    emotion_list = []
    with open('emotions.txt', 'r') as file:
        for line in file:
            clear_line = line.replace("\n", '').replace(",", '').replace("'", '').strip()
            word, emotion = clear_line.split(':')

            if word in lemma_words:
                emotion_list.append(emotion)

    w = Counter(emotion_list)

    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(cleaned_text)
    compound_score = score['compound']

    return w, compound_score

if __name__ == "__main__":
    text_content = open('read.txt', encoding='utf-8').read()
    result, compound_score = perform_sentiment_analysis(text_content)
    print(result)
    print(compound_score)