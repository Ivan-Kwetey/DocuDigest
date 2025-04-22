document.addEventListener("DOMContentLoaded", () => {
    const uploadBtn = document.getElementById("upload-btn");
    const fileInput = document.getElementById("file-input");
    const summaryBox = document.getElementById("summary-box");
    const questionBox = document.getElementById("question-box");
    const askBtn = document.getElementById("ask-btn");
    const answerBox = document.getElementById("qa-answer");
    const compareBtn = document.getElementById("compare-btn");
    const compareInput = document.getElementById("compare-input");
    const differenceBox = document.getElementById("difference-box");
    const mergeControls = document.getElementById("merge-controls");
    const mergeBtn = document.getElementById("merge-btn");
    const saveBtn = document.getElementById("save-btn");
    const playBtn = document.getElementById("play-btn");
    const stopBtn = document.getElementById("stop-btn");
    const startOverBtn = document.getElementById("start-over-btn");
  
    let fullText = "";
    let summary = "";
    let mergedContent = "";
    let utterance;
    let voices = [];
  
    // Load voices
    speechSynthesis.onvoiceschanged = () => {
      voices = speechSynthesis.getVoices();
    };
  
    // Upload handler
    if (uploadBtn && fileInput) {
      uploadBtn.addEventListener("click", (e) => {
        e.preventDefault();
        fileInput.click();
      });
  
      fileInput.addEventListener("change", async () => {
        const file = fileInput.files[0];
        if (!file) return;
  
        const formData = new FormData();
        formData.append("file", file);
  
        try {
          const response = await fetch("/summarize", {
            method: "POST",
            body: formData,
          });
  
          const result = await response.json();
          if (result.summary) {
            summary = result.summary;
            fullText = result.full_text;
            localStorage.setItem("summary", summary);
            localStorage.setItem("fullText", fullText);
            window.location.href = "/results";
          } else {
            alert("Error: " + result.error);
          }
        } catch (err) {
          console.error("Upload failed:", err);
          alert("Something went wrong.");
        }
      });
    }
  
    // Summary & Recommendations on Results Page
    if (summaryBox && localStorage.getItem("summary")) {
      summary = localStorage.getItem("summary");
      fullText = localStorage.getItem("fullText");
      summaryBox.textContent = summary;
  
      fetch("/recommend", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: fullText })
      })
      .then(res => res.json())
      .then(data => {
        const list = document.getElementById("similar-docs");
        if (list && data.results?.length > 0) {
          list.innerHTML = data.results.map(doc => `
            <li>
              <a href="${doc.link}" target="_blank"><strong>${doc.title}</strong></a> (${doc.category})<br>
              Score: ${doc.score.toFixed(4)}
            </li>
          `).join('');
        }
      });
    }
  
    // Q&A
    if (askBtn && questionBox) {
      askBtn.addEventListener("click", async () => {
        const question = questionBox.value.trim();
        if (!question) return;
  
        const response = await fetch("/ask", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question, full_text: fullText, summary }),
        });
  
        const result = await response.json();
        if (result.answer) {
          answerBox.innerHTML = `
            <strong>Answer:</strong><br>${result.answer}
            <hr><strong>Source:</strong><br><em>${result.source || "N/A"}</em>
          `;
        } else {
          answerBox.textContent = "No answer found.";
        }
      });
    }
  
    // Compare Docs
    if (compareBtn && compareInput) {
      compareBtn.addEventListener("click", async () => {
        const file2 = compareInput.files[0];
        if (!file2) return;
  
        const formData = new FormData();
        formData.append("file", file2);
        formData.append("full_text", fullText);
  
        const response = await fetch("/compare-texts", {
          method: "POST",
          body: formData
        });
  
        const result = await response.json();
        if (result && Array.isArray(result.missing_from_doc1)) {
          mergedContent = result.merged_content;
  
          let html = "";
          if (result.missing_from_doc1.length)
            html += `<strong>Missing in PDF 1:</strong><br>${result.missing_from_doc1.join("<br>")}<br><br>`;
          if (result.missing_from_doc2.length)
            html += `<strong>Missing in PDF 2:</strong><br>${result.missing_from_doc2.join("<br>")}`;
  
          if (differenceBox && mergeControls) {
            differenceBox.innerHTML = html;
            differenceBox.style.display = "block";
            mergeControls.style.display = "block";
          }
        }
      });
    }
  
    // Merge & Save
    if (mergeBtn) {
      mergeBtn.addEventListener("click", () => {
        summaryBox.textContent = mergedContent;
        differenceBox.style.display = "none";
        saveBtn.style.display = "inline-block";
      });
    }
  
    if (saveBtn) {
      saveBtn.addEventListener("click", () => {
        const blob = new Blob([summaryBox.textContent], { type: "text/plain" });
        const link = document.createElement("a");
        link.href = URL.createObjectURL(blob);
        link.download = "merged_summary.txt";
        link.click();
      });
    }
  
    // Speech
    if (playBtn) {
      playBtn.addEventListener("click", () => {
        if (summaryBox.textContent.trim()) {
          utterance = new SpeechSynthesisUtterance(summaryBox.textContent);
          utterance.voice = voices.find(v => v.lang.startsWith("en")) || voices[0];
          utterance.rate = 1;
          speechSynthesis.speak(utterance);
        }
      });
    }
  
    if (stopBtn) {
      stopBtn.addEventListener("click", () => {
        if (speechSynthesis.speaking) {
          speechSynthesis.cancel();
        }
      });
    }
  
    // Start Over
    if (startOverBtn) {
      startOverBtn.addEventListener("click", () => {
        localStorage.clear();
        fileInput.click();
      });
    }
  });
  