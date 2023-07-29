# Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
import streamlit as st

# Set the page title and header
st.title('News Classification')
st.header('About the News Prediction Application')

# Provide information about the application using Markdown
st.markdown("""
    <div class="explanation-container">
        The news prediction application has been developed using data from the Nepali news website, setopati.net.
        If you wish to test the application, you can visit <a href="https://en.setopati.com/", style="text-decoration: none;">Setopati Webiste</a>, copy a piece of news, 
        and then predict the category of that news using the application.
    </div>
    """, unsafe_allow_html=True)

# Function to load the data and train the model
def load_data_and_train_model():
    # Load the news data from a CSV file
    df = pd.read_csv('news.csv', encoding='latin1')
    df = df.sample(frac=1)  # Shuffle the data

    # Create a TF-IDF vectorizer and apply it to the news text
    vectorizer = TfidfVectorizer(stop_words="english")
    X = df['news']
    Y = df['category']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15)  # Split the data into training and testing sets

    # Build a pipeline that includes feature selection using SelectKBest and logistic regression as the classifier
    pipeline = Pipeline([('vect', vectorizer),
                         ('chi',  SelectKBest(chi2, k=1450)),
                         ('clf', LogisticRegression(random_state=0))])

    # Train the model using the pipeline
    model = pipeline.fit(X_train, y_train)

    return model

# Function to predict the category of news
def predict_category(model, txt):
    # Prepare the input news data as a DataFrame
    news_data = {'predict_news': [txt]}
    news_data_df = pd.DataFrame(news_data)

    # Make predictions using the trained model
    predict_news_cat = model.predict(news_data_df['predict_news'])[0]
    return predict_news_cat

# Load the model
model = load_data_and_train_model()

# Custom CSS to enhance the design
st.markdown("""
<style>
    .stButton button {
        background-color: #0059b3;
        color: white;
        font-family: "Helvetica Neue", Helvetica, sans-serif;
        font-size: 16px;
        font-weight: bold;
        padding: 10px 20px;
        border-radius: 5px;
        border: none;
        cursor: pointer;
    }
            
    .stButton button:hover {
        background-color: black;
        color:white;
    }
    
    .stTextInput textarea {
        font-family: "Helvetica Neue", Helvetica, sans-serif;
        font-size: 18px;
        line-height: 1.6;
        padding: 12px;
        border-radius: 5px;
        border: 2px solid #0059b3;
        resize: none; /* Disable textarea resizing */
    }
    
    .stSuccess {
        color: #008000;
        font-size: 18px;
        font-weight: bold;
    }
    
    .stWarning {
        color: #ff0000;
        font-size: 18px;
        font-weight: bold;
    }
   
           
    .explanation-container {
        background-color: #262730; /* Custom background color */
        padding: 15px;
        border-radius: 10px;
        font-family: "Roboto", sans-serif; /* Custom font */
        font-style: italic; /* Italic font */
        font-size: 16px;
        line-height:40px;
        color:#F0F2F6;
    }
            
</style>
          
</style>
""", unsafe_allow_html=True)

# Page title and heading with custom CSS
st.header('Enter your news below and click "Submit" to predict its category.')

# Adding a container to wrap the app content with the custom background
with st.container():
    # Text area for user input with custom CSS
    txt = st.text_area('Enter news', height=300)

    # Submit news for classification
    if st.button('Submit'):
        if txt.strip():  # Check if the input is not empty
            # Predict the category of the input news and display the result
            predicted_category = predict_category(model, txt)
            st.markdown(f'<p class="stSuccess">Predicted news category: {predicted_category}</p>', unsafe_allow_html=True)
        else:
            # Display a warning message if the input is empty
            st.markdown('<p class="stWarning">Please enter some news to predict its category.</p>', unsafe_allow_html=True)

# Caption to credit the developer
st.caption('Developed by :blue[Miraj Deep Bhandari]:sunglasses:')
