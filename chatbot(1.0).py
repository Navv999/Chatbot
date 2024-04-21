# import pandas as pd
# from sklearn.model_selection import train_test_split
# import nltk
# from nltk.tokenize import word_tokenize,sent_tokenize

# nltk.download("punkt")

# contraction_dict = {"ain't": "are not","'s":" is","aren't": "are not"}

# path="/Users/pawankumar/Desktop/chatbot/train.csv"
# df=pd.read_csv(path)
# print(df.dtypes)
# context=df['Context']
# respose=df['Response']
# print(context)

# # X_nopunct=[''.join(char for char in )]sclera

# Importing necessary libraries
from nltk.stem import WordNetLemmatizer
import nltk
import numpy as np
import random
from keras import Sequential
from keras.layers import Dense, Dropout
import string
import json
import tensorflow as tf

# Downloading NLTK resources
nltk.download("punkt")

# Sample dataset representing intents, patterns, and responses
ourData = {
  "ourIntents": [
    {
      "tag": "age",
      "patterns": ["how old are you?", "what is your age", "When were you created?", "Birthdate of the bot"],
      "responses": ["I am just born", "I don't have an age, but I was created recently", "I'm a virtual assistant, so no age!"]
    },
    {
      "tag": "greeting",
      "patterns": ["Hi", "Hello", "Hey", "Hola", "sup?", "Greetings", "Good day", "Howdy"],
      "responses": ["Hey There", "Hello", "Hi", "Namaskaar", "Greetings!", "Howdy!"]
    },
    {
      "tag": "goodbye",
      "patterns": ["bye", "seeyou", "later", "See you later", "Goodbye", "Farewell", "Take care"],
      "responses": ["Bye", "take care", "Farewell", "See you later!", "Goodbye!", "Have a great day!", "Until next time!"]
    },
    {
      "tag": "name",
      "patterns": ["Whats your name?", "Who are you?", "Are you a bot?", "Tell me your name"],
      "responses": ["You can give me one", "I'm often called ChatBot", "I'm your friendly virtual assistant"]
    },
    {
      "tag": "conversation",
      "patterns": ["Tell me a joke", "What's your favorite color?", "How is the weather today?", "What's the capital of France?", "Any interesting facts?", "Share something fun", "Entertain me", "What's up?"],
      "responses": ["Why don't scientists trust atoms? Because they make up everything!", "I don't have a favorite color, but I like all the colors!", "I'm not sure about the weather, but I can chat with you!", "The capital of France is Paris.", "Sure, here's a fact: Honey never spoils!", "Sure, let me share a fun fact: The world's oldest known joke dates back to 1900 BC in Sumeria!"]
    },
    {
      "tag": "thanks",
      "patterns": ["Thank you", "Thanks a lot", "Appreciate it", "Gratitude", "Thanks a bunch", "Thanks so much"],
      "responses": ["You're welcome!", "Anytime!", "No problem!", "Glad I could help!", "You're appreciated!", "Happy to assist!"]
    },
    {
      "tag": "polite",
      "patterns": ["Please", "Could you help me?", "May I ask a question?", "Would you mind helping?", "A little assistance, please?", "Can you assist me?"],
      "responses": ["Certainly!", "Of course!", "I'd be happy to help!", "Absolutely!", "Sure, how can I assist you?", "Ask away!"]
    },
    {
      "tag": "favorites",
      "patterns": ["What's your favorite food?", "Do you have any hobbies?", "Tell me about your interests", "Any preferences?", "What do you like?", "Favorite things"],
      "responses": ["I don't eat, but I think pizza sounds interesting!", "I enjoy chatting and helping users!", "I'm always interested in learning and assisting.", "I don't have personal preferences, but I can suggest some popular genres like pop or rock!", "I like the sound of a well-constructed algorithm!", "How about trying a classic like 'Bohemian Rhapsody' by Queen?", "I'm a fan of positive conversations!"]
    },
    {
      "tag": "music",
      "patterns": ["What music do you like?", "Any favorite songs?", "Tell me a music recommendation", "Music preferences", "Favorite genre", "Recommend me some music"],
      "responses": ["I don't have personal preferences, but I can suggest some popular genres like pop or rock!", "I like the sound of a well-constructed algorithm!", "How about trying a classic like 'Bohemian Rhapsody' by Queen?", "Music is so diverse; there's something for every mood!", "Exploring new music is always a great idea!", "I'm not a DJ, but I can recommend some tunes!"]
    },
    {
      "tag": "technology",
      "patterns": ["Tell me about the latest technology news", "What's your opinion on AI?", "Recommend a tech book", "Favorite tech trends", "Latest innovations", "Tech updates", "What's hot in tech?"],
      "responses": ["I'm always up-to-date with the latest tech news!", "As an AI, I find the advancements in artificial intelligence fascinating!", "If you're interested in tech, you might like 'The Innovators' by Walter Isaacson.", "The tech world is always buzzing with exciting developments!", "AI is transforming how we live and work.", "Stay tuned for the latest tech trends!", "Let's dive into the world of technology!"]
    },
    {
      "tag": "movies",
      "patterns": ["Favorite movie?", "Any good movies lately?", "Recommend a movie", "Movie preferences", "Top movie picks", "Tell me about movies"],
      "responses": ["I don't watch movies, but I've heard 'The Shawshank Redemption' is a classic!", "Movies are a great way to relax and unwind!", "If you like action, 'The Dark Knight' is a must-watch.", "Have you seen 'Inception'? It's mind-bending!", "There are so many genres to explore in the world of movies!", "Grab some popcorn and enjoy a movie night!"]
    },
    {
      "tag": "books",
      "patterns": ["Favorite book?", "Any good books lately?", "Recommend a book", "Book preferences", "Top book picks", "Tell me about books"],
      "responses": ["I don't read, but I've heard 'To Kill a Mockingbird' is a literary classic!", "Books are a fantastic way to broaden your horizons!", "If you enjoy fantasy, 'The Hobbit' is a timeless tale.", "Have you explored the world of 'Harry Potter'? It's magical!", "There are so many genres to explore in the world of books!", "Get lost in a good book and let your imagination soar!"]
    }
  ]
}
# Initializing WordNet lemmatizer
lm = WordNetLemmatizer()

# Initializing lists for words, classes, documents, and training data
ourClasses = []
newWords = []
documentX = []
documentY = []

# Extracting tokens from patterns and building lists for training
for intent in ourData["ourIntents"]:
    for pattern in intent["patterns"]:
        ournewTkns = nltk.word_tokenize(pattern)
        newWords.extend(ournewTkns)
        documentX.append(pattern)
        documentY.append(intent["tag"])

    # Adding unique tags to the list of classes
    if intent["tag"] not in ourClasses:
        ourClasses.append(intent["tag"])

# Lowercasing, lemmatizing, and removing punctuation from words
newWords = [lm.lemmatize(word.lower()) for word in newWords if word not in string.punctuation]
newWords = sorted(set(newWords))
ourClasses = sorted(set(ourClasses))

# Initializing training data and creating a bag of words representation
trainingData = []
outEmpty = [0] * len(ourClasses)

for idx, doc in enumerate(documentX):
    bagOfwords = []
    text = lm.lemmatize(doc.lower())
    for word in newWords:
        bagOfwords.append(1) if word in text else bagOfwords.append(0)

    # Creating one-hot encoded output rows
    outputRow = list(outEmpty)
    outputRow[ourClasses.index(documentY[idx])] = 1
    trainingData.append([bagOfwords, outputRow])

# Shuffling and converting training data to numpy array
random.shuffle(trainingData)
trainingData = np.array(trainingData, dtype=object)
x = np.array(list(trainingData[:, 0]))
y = np.array(list(trainingData[:, 1]))

# Defining input and output shapes
iShape = (len(x[0]),)
oShape = len(y[0])

# Building a Sequential model
ourNewModel = Sequential()
ourNewModel.add(Dense(128, input_shape=iShape, activation="relu"))
ourNewModel.add(Dropout(0.5))
ourNewModel.add(Dense(64, activation="relu"))
ourNewModel.add(Dense(oShape, activation="softmax"))

# Compiling the model
md = tf.keras.optimizers.Adam(learning_rate=0.01)
ourNewModel.compile(loss='categorical_crossentropy', optimizer=md, metrics=["accuracy"])

# Displaying the model summary
print(ourNewModel.summary())

# Training the model
ourNewModel.fit(x, y, epochs=200, verbose=1)

# Function to tokenize text
def ourText(text):
    newtkns = nltk.word_tokenize(text)
    newtkns = [lm.lemmatize(word) for word in newtkns]
    return newtkns

# Function to create a bag of words for a given text
def wordbag(text, vocab):
    newtkns = ourText(text)
    bagOfwords = [0] * len(vocab)
    for w in newtkns:
        for idx, word in enumerate(vocab):
            if word == w:
                bagOfwords[idx] = 1
    return np.array(bagOfwords)

# Function to predict intent based on user input
def Pclass(text, vocab, labels):
    bagOwords = wordbag(text, vocab)
    ourResult = ourNewModel.predict(np.array([bagOwords]))[0]
    newThresh = 0.2
    yp = [[idx, res] for idx, res in enumerate(ourResult) if res > newThresh]

    yp.sort(key=lambda x: x[1], reverse=True)
    newList = []
    for r in yp:
        newList.append(labels[r[0]])
    return newList

# Function to get a random response based on the predicted intent
def getRes(firstlist, fJson):
    tag = firstlist[0]
    listOfIntents = fJson["ourIntents"]
    for i in listOfIntents:
        if i["tag"] == tag:
            ourResult = random.choice(i["responses"])
            break
    return ourResult

# Main loop for user interaction
while True:
    newMessage = input("")
    intent = Pclass(newMessage, newWords, ourClasses)
    ourResult = getRes(intent, ourData)
    print(ourResult)
