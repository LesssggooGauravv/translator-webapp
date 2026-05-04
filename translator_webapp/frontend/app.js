/**
 * app.js — Frontend logic for the Seq2Seq Translator
 * 
 * Handles:
 *   - Sending translation requests to the FastAPI backend via Fetch API
 *   - Showing a loading spinner while waiting for the response
 *   - Displaying results or error messages
 *   - Character count for the input textarea
 */

// ── Configuration ──
const API_BASE_URL = "http://localhost:8000";

// ── DOM Elements ──
const inputText = document.getElementById("input-text");
const outputText = document.getElementById("output-text");
const translateBtn = document.getElementById("translate-btn");
const btnText = document.getElementById("btn-text");
const spinner = document.getElementById("spinner");
const statusMsg = document.getElementById("status-msg");
const charCount = document.getElementById("char-count");
const srcLang = document.getElementById("src-lang");
const tgtLang = document.getElementById("tgt-lang");


// ── Character Counter ──
inputText.addEventListener("input", () => {
    charCount.textContent = inputText.value.length;
});


// ── Main Translation Function ──
async function translateText() {
    const text = inputText.value.trim();

    // Validate input
    if (!text) {
        showStatus("Please enter some text to translate.", "error");
        return;
    }

    // Show loading state
    setLoading(true);
    hideStatus();
    outputText.value = "";

    try {
        const response = await fetch(`${API_BASE_URL}/translate`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                text: text,
                src_lang: srcLang.value,
                tgt_lang: tgtLang.value,
            }),
        });

        // Parse response
        const data = await response.json();

        if (!response.ok) {
            // Server returned an error
            const errorMsg = data.detail || "Translation failed. Please try again.";
            showStatus(errorMsg, "error");
            return;
        }

        // Display the translated text
        outputText.value = data.translated_text;
        showStatus("Translation complete!", "success");

    } catch (error) {
        // Network error (backend not running, etc.)
        console.error("Translation error:", error);

        if (error.message.includes("Failed to fetch") || error.message.includes("NetworkError")) {
            showStatus(
                "Cannot connect to the backend server. Make sure the FastAPI server is running on port 8000.",
                "error"
            );
        } else {
            showStatus(`Error: ${error.message}`, "error");
        }
    } finally {
        setLoading(false);
    }
}


// ── Clear Everything ──
function clearAll() {
    inputText.value = "";
    outputText.value = "";
    charCount.textContent = "0";
    hideStatus();
}


// ── Loading State Helpers ──
function setLoading(isLoading) {
    translateBtn.disabled = isLoading;

    if (isLoading) {
        btnText.textContent = "Translating...";
        spinner.classList.remove("hidden");
    } else {
        btnText.textContent = "Translate";
        spinner.classList.add("hidden");
    }
}


// ── Status Message Helpers ──
function showStatus(message, type) {
    statusMsg.textContent = message;
    statusMsg.className = `status-msg ${type}`;
    statusMsg.classList.remove("hidden");
}

function hideStatus() {
    statusMsg.classList.add("hidden");
    statusMsg.className = "status-msg hidden";
}


// ── Allow Ctrl+Enter to translate ──
inputText.addEventListener("keydown", (event) => {
    if (event.ctrlKey && event.key === "Enter") {
        translateText();
    }
});
