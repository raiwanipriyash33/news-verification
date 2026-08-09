# News Reliability Verification Terminal

This web application utilizes machine learning to analyze the linguistic structure and source markers of news articles to predict their authenticity. It is built using Python, Scikit-Learn, and Streamlit.

## 🚀 Live Demo
You can access the live web application here: [News Verification Terminal](https://news-verification-mpmff2eba74mjuy63gnnup.streamlit.app/)

## 🧠 Model Architecture

The terminal processes raw text using a custom `wordopt` function (which removes punctuation, URLs, HTML tags, and numerical artifacts) followed by a **TF-IDF Vectorizer** to convert the text into a numerical format.

The text is then evaluated simultaneously by four distinct machine learning classifiers:

1.  **Logistic Regression (LR):** A foundational linear model that predicts the probability of a binary outcome.
2.  **Decision Tree (DT):** A non-linear model that splits data into branches based on feature values.
3.  **Gradient Boosting (GB):** An ensemble technique that builds trees sequentially, with each new tree correcting the errors of the previous ones.
4.  **Random Forest (RF):** A powerful ensemble method that builds multiple decision trees simultaneously and averages their predictions. 
    * *Note on RF Architecture:* To ensure the model remains lightweight and performant for web deployment, the Random Forest classifier has been optimized (`n_estimators=50`, `max_depth=15`). This pruning reduces file size by over 80% while maintaining highly accurate predictive capabilities.

