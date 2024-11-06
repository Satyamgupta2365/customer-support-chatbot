import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from fuzzywuzzy import process

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_qa(file_path):
    """Load question-answer pairs from a JSON file."""
    try:
      with open('chattxt.json', 'r') as file:
        data = json.load(file)
        print("JSON loaded successfully:", data)
    except json.JSONDecodeError as e:
      print("JSON parsing error:", e)
    except FileNotFoundError:
      print("File not found.")

def preprocess(text):
    """Preprocess text by tokenizing, removing stopwords, and lemmatizing."""
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text.lower())
    processed = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(processed)

def get_best_match(user_input, qa_pairs, threshold=70):
    """Find the best matching question from QA pairs based on user input."""
    user_input_processed = preprocess(user_input)
    questions = [pair['question'] for pair in qa_pairs]
    best_match, score = process.extractOne(user_input_processed, questions)

    if score > threshold:  # Adjust threshold for accuracy
        for pair in qa_pairs:
            if pair['question'] == best_match:
                return pair['answer']
    return "Sorry, I don't understand that."

def chatbot():
    qa_pairs = load_qa(r'chattxt.json')  # Load the question-answer pairs from the specified JSON file.

    if not qa_pairs:
        print("Simple Boy Chatbot: I currently have no information to assist with.")
        return

    print("Simple Boy Chatbot: Hello! How can I help you today?")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Simple Boy Chatbot: Goodbye!")
            break
        response = get_best_match(user_input, qa_pairs)
        print(f"Simple Boy Chatbot: {response}")

if __name__ == "__main__":
    chatbot()
