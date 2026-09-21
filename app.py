"""
app.py
Flask web application for the Automated Text Summarization System.
"""
import os
from flask import Flask, render_template, request
from summarizer import summarize_tfidf, get_stats

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    summary = None
    stats = None
    original_text = ""
    num_sentences = 3

    if request.method == "POST":
        original_text = request.form.get("original_text", "").strip()
        num_sentences = int(request.form.get("num_sentences", 3))
        if original_text:
            summary, _ = summarize_tfidf(original_text, num_sentences=num_sentences)
            stats = get_stats(original_text, summary)

    return render_template(
        "index.html",
        summary=summary,
        stats=stats,
        original_text=original_text,
        num_sentences=num_sentences,
    )


@app.route("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)
