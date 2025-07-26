
from flask import Flask, render_template, request
from summarizer import summarize_bert
import os
import PyPDF2
from PIL import Image
import pytesseract

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def extract_text(file):
    if file.filename.endswith('.txt'):
        return file.read().decode('utf-8')
    elif file.filename.endswith('.pdf'):
        pdf = PyPDF2.PdfReader(file)
        text = ''
        for page in pdf.pages:
            text += page.extract_text()
        return text
    elif file.filename.endswith(('.jpg', '.jpeg', '.png')):
        image = Image.open(file)
        return pytesseract.image_to_string(image)
    return ""

@app.route("/", methods=["GET", "POST"])
def index():
    summary = ""
    if request.method == "POST":
        input_text = request.form.get("text", "")
        file = request.files.get("file")

        if file and file.filename != "":
            input_text = extract_text(file)

        if input_text.strip() != "":
            summary = summarize_bert(input_text)

    return render_template("index.html", summary=summary)

if __name__ == "__main__":
    app.run(debug=True)
