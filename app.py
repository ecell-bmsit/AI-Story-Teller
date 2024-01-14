from flask import Flask, render_template, request
import openai
from clean import generate_images 
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
from my_nltk import SentimentIntensityAnalyzer

app = Flask(__name__)

openai.api_key = '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'

def generate_story(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    return response['choices'][0]['message']['content'].strip()

def genimg(prompt):
    response = openai.Image.create(
        prompt=prompt,
        n=1,
        size="1024x1024"
    )
    image_url = response['data'][0]['url']
    return image_url.strip()

def sentiment_analyse(text):
    sentiment_analyzer = SentimentIntensityAnalyzer()
    sentiment = sentiment_analyzer.polarity_scores(text)
    return sentiment

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_story', methods=['POST'])
def generate_story_route():
    user_input = request.form['user_input']
    input_text = open('read.txt', 'r').read()
    combined_text = f"{input_text}\n\nUser: {user_input}\nChatbot:"
    generated_story = generate_story(combined_text)
    generated_image_url = genimg(combined_text)
    sentiment_result = sentiment_analyse(combined_text)
    return render_template('result.html', generated_story=generated_story, sentiment=sentiment_result, generated_image_url=generated_image_url)

if __name__ == '__main__':
    app.run(debug=True)