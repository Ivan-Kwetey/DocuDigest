
import torch

import evaluate
from docudigest_model import DocuDigestModel


# For EM and F1 (can use 'exact_match' and 'f1' from `evaluate`)
# But we’ll implement F1 manually below


# Step 1: Load your model
digest_model = DocuDigestModel()

# Step 2: Read the PDF
pdf_path = "./notebooq/Declassified-Assessment-on-COVID-19-Origins.pdf"
full_text = digest_model.read_pdf(pdf_path)

# Optional: Clean text
cleaned_text = digest_model.clean_placeholders(full_text)

# Step 3: Generate summary
generated_summary = digest_model.summarize_text(cleaned_text)
print("\n📄 Generated Summary:\n", generated_summary)

# Step 4: Define reference summary (manually)
reference_summary = """
The document assesses the possible origins of COVID-19, highlighting uncertainties between zoonotic and laboratory-related explanations.
It notes that all agencies agree the virus was not developed as a biological weapon and presents intelligence consensus on natural and lab-based hypotheses.
"""

# Step 5: Compute ROUGE metrics
rouge = evaluate.load("rouge")

summary_scores = rouge.compute(predictions=[generated_summary], references=[reference_summary], use_stemmer=True)

print("\n📊 Summarization Metrics:")
for k, v in summary_scores.items():
    print(f"{k}: {v:.4f}")


# Step 6: QA Evaluation - Sample Question
question = "Was COVID-19 assessed to be a biological weapon?"
expected_answer = "No, it was not developed as a biological weapon."

# Get answer
qa_answer = digest_model.answer_question(question, cleaned_text)
print("\n🧠 QA Answer:", qa_answer)

# Evaluate EM & F1
def compute_f1(pred, true):
    pred_tokens = set(pred.lower().split())
    true_tokens = set(true.lower().split())
    common = pred_tokens & true_tokens
    if not common:
        return 0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(true_tokens)
    return 2 * (precision * recall) / (precision + recall)

print(f"\n📊 QA Metrics:\nExact Match: {int(qa_answer.strip().lower() == expected_answer.lower())}")
print(f"F1 Score: {compute_f1(qa_answer, expected_answer):.4f}")
