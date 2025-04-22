import os
import torch
import re
import joblib
import fitz
from fpdf import FPDF
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForQuestionAnswering
)
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.util import cos_sim


class DocuDigestModel:
    def __init__(self):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["TRANSFORMERS_NO_TF"] = "1"

        self.sum_tokenizer = AutoTokenizer.from_pretrained("t5-small")
        self.sum_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

        self.qa_tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")

        self.gen_qa_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        self.gen_qa_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        print("✅ All models loaded and ready.")

    def read_pdf(self, file_path):
        doc = fitz.open(file_path)
        return " ".join([page.get_text() for page in doc])

    def clean_placeholders(self, text):
        return re.sub(r"\[.*?\]", "", text)

    def summarize_text(self, text, max_len=100, min_len=30):
        input_text = "summarize: " + text
        input_ids = self.sum_tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
        summary_ids = self.sum_model.generate(
            input_ids, max_length=max_len, min_length=min_len,
            length_penalty=2.0, num_beams=4, early_stopping=True
        )
        return self.sum_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    def get_heading_candidates(self, text):
        lines = text.splitlines()
        return list({line.strip() for line in lines if line.strip() and (
                line.strip().isupper() or ':' in line or len(line.strip()) <= 6
        )})

    def extract_section(self, text, section_name, max_lines=40):
        lines = text.splitlines()
        headings = self.get_heading_candidates(text)
        section_text = []
        capture = False
        count = 0
        for line in lines:
            if section_name.lower() in line.lower():
                capture = True
                continue
            if capture:
                if line.strip() in headings and line.strip().lower() != section_name.lower():
                    break
                section_text.append(line)
                count += 1
                if count >= max_lines:
                    break
        return "\n".join(section_text).strip()

    def answer_question(self, question, context, threshold=0.1):
        inputs = self.qa_tokenizer(question, context, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.qa_model(**inputs)
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits
        start_idx = torch.argmax(start_scores)
        end_idx = torch.argmax(end_scores) + 1
        start_conf = torch.softmax(start_scores, dim=1)[0][start_idx]
        end_conf = torch.softmax(end_scores, dim=1)[0][end_idx - 1]
        avg_conf = (start_conf + end_conf) / 2
        if avg_conf < threshold:
            return "(No confident answer found)"
        return self.qa_tokenizer.decode(inputs["input_ids"][0][start_idx:end_idx], skip_special_tokens=True)

    def generate_answer(self, question, context, max_len=128):
        prompt = f"question: {question} context: {context}"
        inputs = self.gen_qa_tokenizer(prompt, return_tensors="pt", truncation=True)
        outputs = self.gen_qa_model.generate(inputs.input_ids, max_length=max_len)
        return self.gen_qa_tokenizer.decode(outputs[0], skip_special_tokens=True)

    def smart_answer(self, question, context, summary=None, confidence_threshold=0.1):
        if "how many" in question.lower():
            count = len(re.findall(r'^\s*\d+\.\s', context, re.MULTILINE))
            return f"{count} item(s) found."
        extractive = self.answer_question(question, context, threshold=confidence_threshold)
        if extractive.strip() not in ["", "(No confident answer found)"]:
            return extractive
        return self.generate_answer(question, summary or context)

    def smart_answer_dynamic(self, question, full_text, summary=None, threshold=0.25, top_k=5):
        from sentence_transformers.util import cos_sim

        # Split full text into sentences
        sentences = re.split(r'(?<=[.!?])\s+', full_text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        # Get embeddings for question and sentences
        question_emb = self.embedding_model.encode([question], convert_to_tensor=True)
        sent_embs = self.embedding_model.encode(sentences, convert_to_tensor=True)

        # Compute cosine similarity between question and each sentence
        scores = cos_sim(question_emb, sent_embs)[0]
        top_indices = torch.topk(scores, k=min(top_k, len(sentences))).indices

        # Pick top-k most relevant sentences
        selected_context = " ".join([sentences[i] for i in top_indices])

        # Try extractive QA
        extractive_answer = self.answer_question(question, selected_context)
        if extractive_answer and extractive_answer.strip() not in ["", "(No confident answer found)"]:
            # Find which sentence it most likely came from
            best_sent = max(
                [(s, cos_sim(self.embedding_model.encode([extractive_answer], convert_to_tensor=True),
                             self.embedding_model.encode([s], convert_to_tensor=True)).item())
                 for s in sentences],
                key=lambda x: x[1],
                default=(None, 0)
            )[0]

            return {
                "answer": extractive_answer,
                "source": best_sent or selected_context  # Fallback if source sentence can't be confidently matched
            }

        # Fall back to generative answer
        generative = self.generate_answer(question, summary or selected_context)
        return {
            "answer": generative,
            "source": selected_context
        }

    def compare_documents(self, doc1_text, doc2_text, threshold=0.75):
        normalize = lambda text: re.sub(r'\s+', ' ', text).strip()
        sents1 = [normalize(s) for s in re.split(r'(?<=[.!?]) +', doc1_text) if s.strip()]
        sents2 = [normalize(s) for s in re.split(r'(?<=[.!?]) +', doc2_text) if s.strip()]
        emb1 = self.embedding_model.encode(sents1, convert_to_tensor=True)
        emb2 = self.embedding_model.encode(sents2, convert_to_tensor=True)
        unmatched_1 = [sents1[i] for i in range(len(sents1)) if torch.max(cos_sim(emb1[i], emb2)[0]) < threshold]
        unmatched_2 = [sents2[j] for j in range(len(sents2)) if torch.max(cos_sim(emb2[j], emb1)[0]) < threshold]
        return unmatched_1, unmatched_2

    def merge_documents(self, doc1_text, doc2_text, threshold=0.75):
        removed, added = self.compare_documents(doc1_text, doc2_text, threshold)
        combined = set(re.split(r'(?<=[.!?]) +', doc1_text))
        combined.update(added)
        combined.update(removed)
        return " ".join(sorted(combined))

    def save_to_pdf(self, text, filename="merged_output.pdf"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font("Arial", size=12)
        for line in text.split('\n'):
            pdf.multi_cell(0, 10, line)
        pdf.output(filename)

    def recommend_similar(self, input_text, k=5):
        import faiss
        import numpy as np
        import pandas as pd
        from urllib.parse import quote

        print("🔍 Input text length:", len(input_text))

        try:
            df = pd.read_csv("arxiv_cleaned.csv")
            print("📄 Dataset loaded:", df.shape)
        except Exception as e:
            print("❌ Error loading dataset:", e)
            return []

        try:
            index = faiss.read_index("arxiv_index.faiss")
            print("📦 FAISS index loaded. Total entries:", index.ntotal)
        except Exception as e:
            print("❌ Error loading FAISS index:", e)
            return []

        try:
            embedding = self.embedding_model.encode([input_text])
            faiss.normalize_L2(embedding)

            D, I = index.search(embedding.astype("float32"), k)
            print("🔗 Nearest neighbor indices:", I)
            print("📏 Distances:", D)
        except Exception as e:
            print("❌ Error during embedding + search:", e)
            return []

        results = []
        for i, idx in enumerate(I[0]):
            if idx < len(df):
                title = df.iloc[idx]['title']
                category = df.iloc[idx]['category']
                authors = df.iloc[idx]['authors']
                score = float(D[0][i])

                # Use title-based search link to avoid ID issues
                title_query = quote(title)
                link = f"https://arxiv.org/search/?query={title_query}&searchtype=title"

                results.append({
                    "title": title,
                    "category": category,
                    "authors": authors,
                    "score": score,
                    "link": link
                })

                print(f"✅ Match {i + 1}: {title} (Score: {score:.4f})")

        return results


# Save the model wrapper (not including large model weights)
digest_model = DocuDigestModel()
joblib.dump(digest_model, 'docudigest_model.pkl')
