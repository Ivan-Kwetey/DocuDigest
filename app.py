from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import os
from docudigest_model import DocuDigestModel

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = DocuDigestModel()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/results")
def results():
    return render_template("results.html")

@app.route("/summarize", methods=["POST"])
def summarize():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    filename = secure_filename(file.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(path)

    try:
        text = model.read_pdf(path)
        cleaned = model.clean_placeholders(text)
        summary = model.summarize_text(cleaned)
        return jsonify({"summary": summary, "full_text": cleaned})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question")
    full_text = data.get("full_text")
    summary = data.get("summary")

    if not question or not full_text:
        return jsonify({"error": "Missing question or document"}), 400

    answer_obj = model.smart_answer_dynamic(question, full_text, summary)
    return jsonify(answer_obj)


@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.json
    input_text = data.get("text", "")

    print("📨 Received input for recommendation. Length:", len(input_text))

    if not input_text.strip():
        return jsonify({"error": "No input text provided"}), 400

    try:
        similar_docs = model.recommend_similar(input_text)
        print("📦 Returning", len(similar_docs), "similar documents")
        return jsonify({"results": similar_docs})
    except Exception as e:
        print("🔥 Error in recommend():", str(e))
        return jsonify({"error": str(e)}), 500


@app.route("/compare-texts", methods=["POST"])
def compare_texts():
    if 'file' not in request.files or 'full_text' not in request.form:
        return jsonify({"error": "Full text and second file required"}), 400

    file = request.files['file']
    full_text = request.form['full_text']

    filename = secure_filename(file.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(path)

    try:
        text2 = model.read_pdf(path)
        cleaned2 = model.clean_placeholders(text2)
        missing1, missing2 = model.compare_documents(full_text, cleaned2)
        merged = model.merge_documents(full_text, cleaned2)
        return jsonify({
            "missing_from_doc1": missing1,
            "missing_from_doc2": missing2,
            "merged_content": merged
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
