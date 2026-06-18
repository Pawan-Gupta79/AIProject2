"""
Aaida Hossain 40281796
Khogulan Thambu 40029149
Pawan Kumar Gupta 40254781

Project 2: AI Email Spam Filtering System
"""
import sys                              
import pandas as pd                     
import matplotlib.pyplot as plt                                 # draws bar chart
import seaborn as sns                                           # draws the confusion matrix heatmap
from sklearn.model_selection import train_test_split            # splits data into "for training" / "for testing"
from sklearn.feature_extraction.text import TfidfVectorizer     # converts words into numbers
from sklearn.naive_bayes import MultinomialNB                   # learning algo Naive Bayes
from sklearn.metrics import accuracy_score, confusion_matrix    # evaluates the performance of the model


# Spam Detection Class
class SpamDetector:
 
    def __init__(self, csv_path):
        self.csv_path = csv_path        
        self.df = None
        self.vectorizer = TfidfVectorizer(stop_words='english')     # ignore common workds like the, is, and, etc.
        self.model = MultinomialNB()
        self.is_trained = False                                     # flag to track if model is trained yet
 
    def load_data(self):
        try:
            # Original dataset is in latin-1 encoding, but we want UTF-8. encoding='utf-8-sig' can handle both.
            self.df = pd.read_csv(self.csv_path, encoding='utf-8-sig')
 
        # In case file error, catch and print freidnly message instead of crashing.
        except FileNotFoundError:
            print(f"Error: could not find dataset file at '{self.csv_path}'")
            sys.exit(1)
 
        except UnicodeDecodeError:
            # This runs ONLY if the file exists but isn't valid UTF-8 text.
            print("Error: file is not valid UTF-8. Check the file's encoding.")
            sys.exit(1)
 
        # The dataset has 5 col, we only care about the first 2 and remove the rest.
        self.df = self.df.iloc[:, :2]
 
        # There was an error in naming the col, it said lable instead of label so we force rename the col to label and message to prevent errors.
        self.df.columns = ['label', 'message']
 
        # Remove any row with missing data. Prevent future crash when vectorizing.
        before = len(self.df)
        self.df.dropna(inplace=True)
        dropped = before - len(self.df)
        if dropped:
            print(f"Dropped {dropped} row(s) with missing data.")
 
        # Print a quick summary so we can see what we loaded.
        print(f"Loaded {len(self.df)} messages.")
        print(self.df['label'].value_counts())              # show number of spam vs ham
        return self.df
 

    """Visuzalize the distribution of spam vs ham messages in the dataset, and save it as an image file.""" 
    def visualize_distribution(self, save_path='spam_ham_distribution.png'):
        counts = self.df['label'].value_counts()
 
        plt.figure(figsize=(6, 4))                  # blank chart 6x4 
        bars = plt.bar(counts.index, counts.values, # draw one bar per label
                        color=["#0059FF", "#D40000"])
        plt.title('Number of Spam vs Ham Messages')
        plt.xlabel('Label')
        plt.ylabel('Count')
 
        # Write the exact number on top of each bar.
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height + 20,
                      str(height), ha='center')
 
        plt.tight_layout()         # avoid labels getting cut off at the edges
        plt.savefig(save_path)     # save the chart as an image file
        plt.close()              
        print(f"Saved distribution chart to '{save_path}'")
 

    """Split the data into training/testing groups, then convert the message text into numbers (TF-IDF)."""
    def prepare_features(self, test_size=0.2):
        X = self.df['message']   # X = the "inputs": the raw text of every message
        y = self.df['label']     # y = the "answers": spam or ham for every message
 
        # Randomly divide messages into a training group (80%) and testing group (20%). random_state=42 just makes the random
        # split repeatable - rerun the script, get the same split. stratify=y keeps the same spam/ham ratio in both groups.

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
 
        # fit_transform on the TRAINING text: this is the step where the vectorizer actually "learns" the vocabulary (which words exist,
        # how rare each one is) AND converts that text into numbers, in one move.
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
 
        # transform (NOT fit_transform) on the TEST text: we reuse the exact vocabulary already learned from training. If we let the
        # test data influence the vocabulary too, we'd be "cheating" - the model would get a sneak peek at test data during training.
        X_test_tfidf = self.vectorizer.transform(X_test)
 
        return X_train_tfidf, X_test_tfidf, y_train, y_test
    
 
    """Show the model the training examples and let it learn the pattern."""
    def train(self, X_train, y_train):

        self.model.fit(X_train, y_train)   # learning happens here
        self.is_trained = True             # flip the flag now that it's trained

 
    """Check how good the trained model is, using ONLY the test examples it has never seen before."""
    def evaluate(self, X_test, y_test, save_path='confusion_matrix.png'):
  
        # Ask model to guess the label for every test message.
        y_pred = self.model.predict(X_test)
 
        # Compare the model's guesses (y_pred) to the real answers (y_test)
        accuracy = accuracy_score(y_test, y_pred)
 
        # Build a 2x2 table breaking into 4 specific counts
        # correctly-caught spam, correctly-identified ham, spam that slipped through as ham, ham wrongly flagged as spam.
        cm = confusion_matrix(y_test, y_pred, labels=['spam', 'ham'])
 
        print(f"\nAccuracy: {accuracy * 100:.2f}%")
        print("\nConfusion Matrix (rows = actual, columns = predicted):")
        print(f"{'':10}{'Spam':>8}{'Ham':>8}")
        print(f"{'Spam':10}{cm[0][0]:>8}{cm[0][1]:>8}")
        print(f"{'Ham':10}{cm[1][0]:>8}{cm[1][1]:>8}")
 
        # Draw the same 2x2 table as a color-coded heatmap image 
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Spam', 'Ham'], yticklabels=['Spam', 'Ham'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Saved confusion matrix heatmap to '{save_path}'")
 
        return accuracy, cm
 

    """Take new message types by user and return guess + confidence."""
    def predict(self, message):

        if not self.is_trained:                 # Safety check, only predict if trained.
            raise RuntimeError("Model has not been trained yet.")
 
        features = self.vectorizer.transform([message])
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        confidence = max(probabilities) * 100
        return prediction, confidence
    
 
    """Keep asking the user for messages until quit."""
    def interactive_loop(self):

        print("\nType 'quit' to exit.\n")
        while True: 
            message = input("Enter message: ").strip()
            if message.lower() == 'quit':
                print("Goodbye!")
                break
 
            if not message:
                print("Please enter a non-empty message.\n")
                continue 

            try:
                prediction, confidence = self.predict(message)
                print(f"Prediction: {prediction.upper()}")
                print(f"Confidence: {confidence:.1f}%\n")
            except Exception as error:
                # Catch-all safetyL if anything unexpected goes wrong while predicting, show the error instead of crashing.
                print(f"Error processing message: {error}\n")
 

def main():

    print("Welcome to AI Email Spam Detector\n")
    detector = SpamDetector('spam.csv')
 
    detector.load_data()                # Step 1: read and clean the data
    detector.visualize_distribution()   # Step 2: save the spam/ham bar chart
 
    # Step 3: split into train/test groups AND convert text to numbers
    X_train, X_test, y_train, y_test = detector.prepare_features()
 
    print("\nTraining model...")
    detector.train(X_train, y_train)    # Step 4: train the classifier
 
    detector.evaluate(X_test, y_test)   # Step 5: report accuracy + confusion matrix
 
    detector.interactive_loop()         # Step 6: let the user try their own messages

if __name__ == '__main__':
    main()


