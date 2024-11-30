import os
import re
import pdfplumber
import pandas as pd
import logging
import numpy as np
from keras.models import load_model
from keras.layers import Layer, Dense, Dropout, LayerNormalization
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf


logging.basicConfig(level=logging.INFO)


MODEL_DIR = 'models'  
MODEL_PATH = os.path.join(MODEL_DIR, 'model.h5')  


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'pdf'}

def extract_data_from_pdf(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            all_text = ""
            for page in pdf.pages:
                all_text += page.extract_text()

        lines = re.split(r'\r?\n|\r', all_text)
        lines = [line.strip() for line in lines if line.strip()]

        column_names = ['Year', 'Company', 'Market_Cap_in_B_USD', 'Revenue', 'Gross_Profit', 
                        'Net_Income', 'Cash_Flow_from_Operating', 'Cash_Flow_from_Investing', 
                        'Cash_Flow_from_Financial_Activities', 'Debt_Equity_Ratio']
        
        data_lines = [line.split(',') for line in lines]
        final_data = []
        
        for line in data_lines:
            cleaned_line = [element.strip() for element in line]
            if len(cleaned_line) < len(column_names):
                cleaned_line += [''] * (len(column_names) - len(cleaned_line))
            final_data.append(cleaned_line)

        df = pd.DataFrame(final_data, columns=column_names)
        for col in column_names[0:1] + column_names[2:]: 
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(thresh=len(df.columns) // 2)
        return df

    except Exception as e:
        logging.error(f"PDF processing error: {str(e)}")
        raise ValueError(f"An error occurred during PDF processing: {str(e)}")

def preprocess_data(df, feature_columns):
    df_numeric = df[feature_columns].copy()
    df_numeric = df_numeric.apply(pd.to_numeric, errors='coerce')
    df_numeric.fillna(df_numeric.median(), inplace=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_numeric)
    return X_scaled, scaler


class MultiHeadAttention(Layer):
    def __init__(self, num_heads, d_model):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0, 

        self.depth = d_model // num_heads
        self.wq = Dense(d_model)
        self.wk = Dense(d_model)
        self.wv = Dense(d_model)

        self.dense = Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention_logits = tf.matmul(q, k, transpose_b=True) / tf.sqrt(tf.cast(self.depth, tf.float32))
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)

        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.d_model))
        return self.dense(output)

class FeedForwardNetwork(Layer):
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super(FeedForwardNetwork, self).__init__()
        self.dense1 = Dense(dff, activation='relu')
        self.dense2 = Dense(d_model)
        self.dropout = Dropout(dropout_rate)

    def call(self, x):
        return self.dropout(self.dense2(self.dense1(x)))

    def get_config(self):
        config = super(FeedForwardNetwork, self).get_config()
        config.update({
            "d_model": self.d_model,
            "dff": self.dff,
            "dropout_rate": self.dropout_rate
        })
        return config

class TransformerLayer(Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(TransformerLayer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate
        
        self.attention = MultiHeadAttention(num_heads, d_model)
        self.ffn = FeedForwardNetwork(d_model, dff, dropout_rate)
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, x, training):
        attn_output = self.attention(x, x, x)
        out1 = self.layernorm1(x + self.dropout1(attn_output, training=training))
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + self.dropout2(ffn_output, training=training))

    def get_config(self):
        config = super(TransformerLayer, self).get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "dropout_rate": self.dropout_rate
        })
        return config


def build_transformer_model(input_shape, num_heads, dff, dropout_rate=0.1):
    inputs = tf.keras.Input(shape=(input_shape,)) 
    x = Dense(dff, activation='relu')(inputs) 
    x = Dropout(dropout_rate)(x)
    x = Dense(1, activation='sigmoid')(x) 
    model = tf.keras.Model(inputs=inputs, outputs=x)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def train_and_save_model(pdf_path, feature_columns):
    try:
        df = extract_data_from_pdf(pdf_path)
        feature_columns = [col for col in feature_columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
        if not feature_columns:
            raise ValueError("No valid numeric feature columns found in the DataFrame.")
        
        X_scaled, scaler = preprocess_data(df, feature_columns)
        d_model = X_scaled.shape[1]
        num_heads = min(d_model, 5) 
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads}).")
        
        model = build_transformer_model(input_shape=d_model, num_heads=num_heads, dff=128)

        y = (X_scaled.sum(axis=1) > 3).astype(int)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        logging.info(f"Shape of X_train: {X_train.shape}")
        logging.info(f"Shape of y_train: {y_train.shape}")
        
        model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=1) 
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        
        model.save(MODEL_PATH)

    except Exception as e:
        logging.error(f"Training or saving model failed: {str(e)}")
        raise ValueError(f"An error occurred during model training: {str(e)}")


def detect_anomalies(df, columns, model, scaler):
    X = scaler.transform(df[columns])
    predictions = model.predict(X)
    df['Anomaly'] = (predictions.flatten() > 0.5).astype(int)
    return df

if __name__ == "__main__":

    pdf_path = 'financialstatement.pdf'
    feature_columns = ['Year', 'Revenue', 'Net_Income', 'Cash_Flow_from_Operating', 'Debt_Equity_Ratio']

    train_and_save_model(pdf_path, feature_columns)