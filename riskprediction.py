# Import Dependencies
import yaml
from joblib import dump, load
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# Naive Bayes Approach
from sklearn.naive_bayes import MultinomialNB
# Trees Approach
from sklearn.tree import DecisionTreeClassifier
# Ensemble Approach
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import seaborn as sn
import matplotlib.pyplot as plt
class DiseasePrediction:
    # Initialize and Load the Config File
    def _init_(self, model_name=None):
        # Load Config File
        
        try:
    # Load Configuration
            with open("C:\\Users\\Gandhana\\Desktop\\Varcons\\config.yaml", 'r') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError as e:
            raise ConfigurationError("Configuration file not found. Please make sure 'config.yaml' is in the correct location.") from e
        except yaml.YAMLError as e:
            raise ConfigurationError("Error reading the configuration file. Please check the file format and content.") from e


        # Verbose
        self.verbose = self.config['verbose']
        # Load Training Data
        self.train_features, self.train_labels, self.train_df = self._load_train_dataset()
        # Load Test Data
        self.test_features, self.test_labels, self.test_df = self._load_test_dataset()
        # Feature Correlation in Training Data
        self._feature_correlation(data_frame=self.train_df, show_fig=False)
        # Model Definition
        self.model_name = model_name
        # Model Save Path
        self.model_save_path = self.config['model_save_path']
        self.clf = None

    # Function to Load Train Dataset
    def _load_train_dataset(self):
        df_train = pd.read_csv(self.config['dataset']['training_data_path'])

        cols = df_train.columns
        cols = cols[:-2]
        train_features = df_train[cols]
        train_labels = df_train['prognosis']

        # Check for data sanity
        assert (len(train_features.iloc[0]) == 132)
        assert (len(train_labels) == train_features.shape[0])

        if self.verbose:
            print("Length of Training Data: ", df_train.shape)
            print("Training Features: ", train_features.shape)
            print("Training Labels: ", train_labels.shape)
        return train_features, train_labels, df_train

    # Function to Load Test Dataset
    def _load_test_dataset(self):
        df_test = pd.read_csv(self.config['dataset']['test_data_path'])

        cols = df_test.columns
        cols = cols[:-1]
        test_features = df_test[cols]
        test_labels = df_test['prognosis']

        # Check for data sanity
        assert (len(test_features.iloc[0]) == 132)
        assert (len(test_labels) == test_features.shape[0])

        if self.verbose:
            print("Length of Test Data: ", df_test.shape)
            print("Test Features: ", test_features.shape)
            print("Test Labels: ", test_labels.shape)
        return test_features, test_labels, df_test

    # Features Correlation
    def _feature_correlation(self, data_frame=None, show_fig=False):
        # Get Feature Correlation
        numeric_cols = data_frame.select_dtypes(include=['float64', 'int64'])
        corr = numeric_cols.corr()
        sn.heatmap(corr, square=True, annot=False, cmap="YlGnBu")
        plt.title("Feature Correlation")
        plt.tight_layout()
        if show_fig:
            plt.show()
        plt.savefig('feature_correlation.png')

    # Dataset Train Validation Split
    def _train_val_split(self):
        X_train, X_val, y_train, y_val = train_test_split(self.train_features, self.train_labels,
                                                          test_size=self.config['dataset']['validation_size'],
                                                          random_state=self.config['random_state'])

        if self.verbose:
            print("Number of Training Features: {0}\tNumber of Training Labels: {1}".format(len(X_train), len(y_train)))
            print("Number of Validation Features: {0}\tNumber of Validation Labels: {1}".format(len(X_val), len(y_val)))
        return X_train, y_train, X_val, y_val

    # Model Selection
    def select_model(self):
        if self.model_name == 'mnb':
            self.clf = MultinomialNB()
        elif self.model_name == 'decision_tree':
            self.clf = DecisionTreeClassifier(criterion=self.config['model']['decision_tree']['criterion'])
        elif self.model_name == 'random_forest':
            self.clf = RandomForestClassifier(n_estimators=self.config['model']['random_forest']['n_estimators'])
        elif self.model_name == 'gradient_boost':
            self.clf = GradientBoostingClassifier(n_estimators=self.config['model']['gradient_boost']['n_estimators'],
                                                  criterion=self.config['model']['gradient_boost']['criterion'])
        return self.clf

    # ML Model
    def train_model(self):
        # Get the Data
        X_train, y_train, X_val, y_val = self._train_val_split()
        classifier = self.select_model()
        # Training the Model
        classifier = classifier.fit(X_train, y_train)
        self.clf = classifier
        # Trained Model Evaluation on Validation Dataset
        confidence = classifier.score(X_val, y_val)
        # Validation Data Prediction
        y_pred = classifier.predict(X_val)
        # Model Validation Accuracy
        accuracy = accuracy_score(y_val, y_pred)
        # Model Confusion Matrix
        conf_mat = confusion_matrix(y_val, y_pred)
        # Model Classification Report
        clf_report = classification_report(y_val, y_pred)
        # Model Cross Validation Score
        score = cross_val_score(classifier, X_val, y_val, cv=3)

        if self.verbose:
            print('\nTraining Accuracy: ', confidence*100)
            print('\nValidation Prediction: ', y_pred)
            print('\nValidation Accuracy: ', accuracy*100)
            print('\nValidation Confusion Matrix: \n', conf_mat)
            print('\nCross Validation Score: \n', score)
            print('\nClassification Report: \n', clf_report)

        # Save Trained Model
        dump(classifier, str(self.model_save_path + self.model_name + ".joblib"))
        self.clf = classifier

    # Function to Make Predictions on Test Data
    def make_prediction(self, saved_model_name=None, test_data=None):
        try:
            # Load Trained Model
            if self.clf is None:
                print(f"Model {saved_model_name} not found...")
                self.clf = load(str(self.model_save_path + saved_model_name + ".joblib"))
                return None
            if test_data is not None:
                preprocessed_data = self.preprocess_text_input(test_data)
                result = self.clf.predict(preprocessed_data)
                return result
            else:
                print("Test data not provided. Please provide test data.")
                return None
        except Exception as e:
            print("Model not found...")
            return None
    def preprocess_text_input(self, text_input):
        # Tokenize the text (split into words)
        words = text_input.split(',')

        # Convert words into a numerical format (e.g., using word embeddings or TF-IDF)
        # Here, we'll just use a simple one-hot encoding as an example
        unique_words = list(set(words))
        word_indices = {word: i for i, word in enumerate(unique_words)}
        num_features = len(unique_words)

        # Create a one-hot encoded vector
        one_hot_vector = np.zeros(num_features)
        for word in words:
            if word in word_indices:
                index = word_indices[word]
                one_hot_vector[index] = 1

        # Return the preprocessed data as a DataFrame or a NumPy array
        return pd.DataFrame([one_hot_vector], columns=unique_words)
      if _name_ == "_main_":
    # Model Currently Training
    current_model_name = 'decision_tree'
    # Instantiate the Class
    dp = DiseasePrediction(model_name=current_model_name)
    # Train the Model
    dp.train_model()
    # Get Model Performance on Test Data
    test_accuracy, classification_report = dp.make_prediction(saved_model_name=current_model_name)
    print("Model Test Accuracy: ", test_accuracy)
    print("Test Data Classification Report: \n", classification_report)
symptoms = input() 
dp = DiseasePrediction(model_name='decision_tree')  # Specify the model name without the '.joblib' extension

# Make a prediction
predicted_disease = dp.make_prediction(saved_model_name='decision_tree', test_data=symptoms)

print("Predicted Disease:", predicted_disease)
