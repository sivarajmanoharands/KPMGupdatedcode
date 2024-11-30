import os
import re
from flask import Flask, request, render_template, jsonify
import numpy as np
from keras.models import load_model
import logging
import pdfplumber
import pandas as pd
from sklearn.preprocessing import StandardScaler
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from flask import send_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = Flask(__name__)

MODEL_PATH = 'models/model.h5'

model = load_model(MODEL_PATH)

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

        for col in column_names[0:1] + column_names[2:]:  # Skip 'Company'
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(thresh=len(df.columns) // 2)
        return df

    except Exception as e:
        logging.error(f"PDF processing error: {str(e)}")
        raise ValueError(f"An error occurred during PDF processing: {str(e)}")

def preprocess_data(df, feature_columns):
    try:
        df_numeric = df[feature_columns].copy()
        df_numeric = df_numeric.apply(pd.to_numeric, errors='coerce')
        df_numeric.fillna(df_numeric.median(), inplace=True)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_numeric) 
        return np.array(X_scaled), scaler
    except Exception as e:
        logging.error(f"Data preprocessing error: {str(e)}")
        raise ValueError(f"An error occurred during data preprocessing: {str(e)}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid or no selected file'}), 400

    pdf_path = 'temp.pdf'
    file.save(pdf_path)
    logging.info("File uploaded and saved to %s", pdf_path)

    feature_columns = ['Year', 'Revenue', 'Net_Income', 'Cash_Flow_from_Operating', 'Debt_Equity_Ratio']

    try:
        df = extract_data_from_pdf(pdf_path)
        logging.info("PDF data extracted successfully")
        X_scaled, _ = preprocess_data(df, feature_columns)

        if len(X_scaled.shape) == 1:
            X_scaled = X_scaled[np.newaxis, :]

        predictions = model.predict(X_scaled)
        anomalies = [
            df.iloc[i].to_dict()
            for i in range(len(predictions))
            if predictions[i] > 0.5 
        ]

        os.remove(pdf_path)
        return jsonify({'metrics': df.to_dict(orient='records'), 'anomalies': anomalies})

    except Exception as e:
        logging.error("Error during processing: %s", str(e))
        os.remove(pdf_path)
        return jsonify({'error': 'Error processing the file: ' + str(e)}), 500
@app.route('/generate_report', methods=['POST'])
@app.route('/generate_report', methods=['POST'])
def generate_report():
    try:

        data = request.json
        metrics = data.get('metrics', [])
        anomalies = data.get('anomalies', [])

        if not metrics and not anomalies:
            raise ValueError("No metrics or anomalies provided to generate the report.")

        pdf_path = 'financial_report.pdf'
        c = canvas.Canvas(pdf_path, pagesize=letter)
        c.setFont('Helvetica-Bold', 16)
        width, height = letter
        y = height - 50

        c.drawString(50, y, "Financial Analysis Report")
        y -= 40

        c.setFont('Helvetica-Bold', 14)
        c.drawString(50, y, "Metrics:")
        y -= 20
        c.setFont('Helvetica', 12)
        if metrics:
            for metric in metrics:
                line = ', '.join([f"{key}: {value}" for key, value in metric.items()])
                c.drawString(50, y, line)
                y -= 20
                if y < 50:
                    c.showPage()
                    y = height - 50
        else:
            c.drawString(50, y, "No metrics available.")
            y -= 20

        c.setFont('Helvetica-Bold', 14)
        c.drawString(50, y, "Anomalies:")
        y -= 20
        c.setFont('Helvetica', 12)
        if anomalies:
            for anomaly in anomalies:
                line = ', '.join([f"{key}: {value}" for key, value in anomaly.items()])
                c.drawString(50, y, line)
                y -= 20
                if y < 50:
                    c.showPage()
                    y = height - 50
        else:
            c.drawString(50, y, "No anomalies detected.")
            y -= 20

        c.save()

        return send_file(pdf_path, as_attachment=True)

    except Exception as e:
        logging.error(f"Error generating report: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
