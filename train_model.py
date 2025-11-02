import pandas as pd
import numpy as np
import os
import re
import string
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, Layer
from tensorflow.keras import backend as K # Used for custom layer operations
from tensorflow.keras.callbacks import EarlyStopping

# --- Configuration ---
VOCAB_SIZE = 10000
EMBEDDING_DIM = 100
MAX_LEN = 300 
PAD_TYPE = 'post'
TRUNC_TYPE = 'post'
EPOCHS = 10 
BATCH_SIZE = 64
SEED = 42

# Define paths for saving artifacts
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'fake_news_model.h5')
TOKENIZER_PATH = os.path.join(BASE_DIR, 'tokenizer.pkl')

# --- Custom Attention Layer (Crucial for Real-World Performance) ---
class AttentionLayer(Layer):
    """
    Implements a simple Attention Mechanism. It learns to weight the importance 
    of each word's hidden state from the LSTM output.
    """
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Context vector weights (W)
        self.W = self.add_weight(name="att_weight", 
                                 shape=(input_shape[-1], 1), # Shape (units, 1)
                                 initializer="normal", 
                                 trainable=True)
        # Context vector bias (b)
        self.b = self.add_weight(name="att_bias", 
                                 shape=(input_shape[-1],), # Shape (units,)
                                 initializer="zeros", 
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # 1. Alignment Score (u = tanh(inputs @ W + b))
        # K.dot(inputs, self.W) transforms (batch, seq_len, units) -> (batch, seq_len, 1)
        u = K.tanh(K.dot(inputs, self.W) + self.b)
        
        # 2. Attention Weights (alpha = softmax(u))
        # Softmax is applied along the sequence dimension (axis=1)
        alpha = K.softmax(u, axis=1)
        
        # 3. Context Vector (weighted sum of inputs)
        # output = inputs * alpha (element-wise multiplication)
        output = inputs * alpha
        
        # 4. Sum across the sequence length dimension to get the final context vector
        # Returns (batch_size, units)
        return K.sum(output, axis=1)

    def get_config(self):
        return super(AttentionLayer, self).get_config()
# --- End Custom Layer Definition ---


# --- Data Loading and Preprocessing (Unchanged) ---
def load_data():
    """Loads and merges the fake and true news datasets."""
    print("1. Loading and preparing data...")
    try:
        fake_df = pd.read_csv('Fake.csv')
        true_df = pd.read_csv('True.csv')
        fake_df['label'] = 1  # 1 for Fake
        true_df['label'] = 0  # 0 for Real
        df = pd.concat([fake_df, true_df]).sample(frac=1, random_state=SEED).reset_index(drop=True)
        print(f"Dataset loaded. Total samples: {len(df)}")
        return df
    except FileNotFoundError:
        print("🚨 ERROR: Please download the 'Fake.csv' and 'True.csv' files and place them in the project folder.")
        return None

def clean_text(text):
    """Removes special characters, links, and converts text to lowercase."""
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

def preprocess(df):
    """Applies text cleaning and splits data for training."""
    df['combined_text'] = df['title'] + " " + df['text']
    df['combined_text'] = df['combined_text'].apply(clean_text)
    X = df['combined_text'].values
    y = df['label'].values
    return train_test_split(X, y, test_size=0.2, random_state=SEED)

def tokenize_and_pad(X_train, X_test):
    """Tokenizes text and pads sequences."""
    print("2. Tokenizing and padding sequences...")
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)
    train_sequences = tokenizer.texts_to_sequences(X_train)
    test_sequences = tokenizer.texts_to_sequences(X_test)
    X_train_padded = pad_sequences(train_sequences, maxlen=MAX_LEN, padding=PAD_TYPE, truncating=TRUNC_TYPE)
    X_test_padded = pad_sequences(test_sequences, maxlen=MAX_LEN, padding=PAD_TYPE, truncating=TRUNC_TYPE)
    
    with open(TOKENIZER_PATH, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Tokenizer saved to {TOKENIZER_PATH}")

    return X_train_padded, X_test_padded, tokenizer


# --- Model Definition and Training ---

def create_model():
    """Defines and compiles the Deep Bidirectional LSTM model with Attention."""
    print("3. Defining Deep Bidirectional LSTM Model with ATTENTION...")
    model = Sequential([
        Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LEN),
        
        # 1st Bi-LSTM: returns sequences for the next layer
        Bidirectional(LSTM(64, return_sequences=True)), 
        Dropout(0.3),
        
        # 2nd Bi-LSTM: MUST return sequences for the Attention layer
        Bidirectional(LSTM(32, return_sequences=True)),
        
        # Attention Layer: processes the sequence output and weights importance
        AttentionLayer(), 
        
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid') # Sigmoid for binary classification
    ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()
    return model

def train_model(model, X_train, y_train, X_test, y_test):
    """Trains the model and saves it."""
    print(f"4. Training model for {EPOCHS} epochs...")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    ]
    
    history = model.fit(X_train, y_train, 
                        epochs=EPOCHS, 
                        batch_size=BATCH_SIZE, 
                        validation_data=(X_test, y_test),
                        callbacks=callbacks,
                        verbose=1)
    
    # Save the model, requiring custom_objects to save the AttentionLayer structure
    # Option 1 (Recommended): Save in the native Keras format (to a directory)
    model.save(MODEL_PATH) 
    
    # Option 2 (If you must use H5, though deprecated):
    # model.save(MODEL_PATH, save_format='h5')
    print(f"\n✅ Training complete. Model saved to {MODEL_PATH}")
    
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Final Model Evaluation: Loss={loss:.4f}, Accuracy={accuracy*100:.2f}%")


# --- Main Execution Flow ---

if __name__ == '__main__':
    df = load_data()
    if df is not None:
        X_train, X_test, y_train, y_test = preprocess(df)
        X_train_padded, X_test_padded, tokenizer = tokenize_and_pad(X_train, X_test)
        model = create_model()
        train_model(model, X_train_padded, y_train, X_test_padded, y_test)
        
        print("\nAll steps completed. Re-run 'python app.py' now.")
