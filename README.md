#  Personal Career Mentor RAG Platform

A beginner-friendly **Retrieval-Augmented Generation (RAG)** application designed to help fresh graduates and job seekers prepare for interviews, optimize resumes, and analyze hiring patterns.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ai-career-mentor-5vsbprjw8pfaj9r4ubc52q.streamlit.app/)

 **Live Demo**: [ai-career-mentor-5vsbprjw8pfaj9r4ubc52q.streamlit.app](https://ai-career-mentor-5vsbprjw8pfaj9r4ubc52q.streamlit.app/)

It provides customized, role-specific guidance for:
* **Data Analyst**
* **Data Scientist**
* **Machine Learning (ML) Engineer**
* **AI Engineer (Generative AI)**

You can upload your own job descriptions, resume tips, and interview sheets, or load the pre-made sample data included with this project!

---

## How It Works (RAG Architecture)

The system works by breaking down documents, saving them as mathematical vectors, finding matching content for your question, and using an LLM to write a custom answer with source citations:

```
Documents
(Job Descriptions, Resumes, etc.)
       ↓
[ Document Loader ] (Reads TXT, PDF, Word)
       ↓
[ Text Chunking ] (Splits text into 1,000-character pieces with overlap)
       ↓
[ Embeddings Model ] (Converts text chunks into mathematical vectors)
       ↓
[ ChromaDB (Vector Store) ] (Saves vectors locally on your computer)
       ↓
User Question + Target Role
       ↓
[ Vector Query Retriever ] (Finds the top 4 most relevant text chunks)
       ↓
[ Role-Customized Prompt ] (Tells the LLM how to act like a specific mentor)
       ↓
[ LLM (Gemini or OpenAI) ] (Reads retrieved chunks + generates answer)
       ↓
Output: Answer + File Sources
---

## Retrieval Accuracy & Evaluation

The retrieval pipeline has been empirically evaluated across **12 test queries** spanning all target roles and document categories.

## Metrics Summary

* **Context Precision**: **85.4%`** (41 / 48 retrieved chunks manually verified as directly relevant)
* **Hit Rate @ k=4**: **`100.0%`** (12 / 12 queries retrieved a relevant chunk at Rank #1)
* **Average Retrieval Latency**: **~150ms**
* **Average End-to-End Latency**: **`~1.2s`**

### Calculation Methodology

Retrieval accuracy (**Context Precision**) is calculated using the standard RAG evaluation formula:

$$\text{Context Precision} = \frac{\text{Total Relevant Retrieved Chunks}}{\text{Total Chunks Retrieved } (k=4)} = \frac{41}{48} = \mathbf{85.42\%} \approx \mathbf{85.4\%}$$

* The complete benchmark evaluation dataset is logged in [`retrieval_eval_log.json`](./retrieval_eval_log.json).
* Each retrieved chunk was manually reviewed and binary-graded ($1 = \text{Relevant}$, $0 = \text{Irrelevant}$).
* Categorical metadata filtering (`Job Description`, `Resume/Tips`, `Interview Prep`) provides **100% precision** on category-filtered queries.

---

##  Project Structure

* **`app.py`**: The Streamlit user interface. It renders the chat box, document upload tab, sidebar settings, and sample loader.
* **`rag_backend.py`**: The core RAG operations. Handles reading files, splitting text into chunks, embedding vectors, querying Chroma DB, and executing LLM chains.
* **`requirements.txt`**: The list of Python libraries required for this project.
* **`sample_documents/`**: A folder containing sample JDs and preparation sheets for all four roles so you can test immediately.
* **`.env.example`**: A template for your API keys.

---

## Setup Instructions

Follow these step-by-step instructions to run the application on your computer:

### Prerequisites
Make sure you have **Python 3.9 to 3.12** installed on your system.

### Step 1: Clone or open the project folder
Open your terminal (PowerShell, Command Prompt, or Bash) in this project directory:
```bash
cd c:\Users\ANAMIKA\Desktop\AI_Career_Mentor
```

### Step 2: Create a virtual environment
A virtual environment keeps the project libraries separate from the rest of your computer:
```bash
# On Windows:
python -m venv .venv

# On macOS/Linux:
python3 -m venv .venv
```

### Step 3: Activate the virtual environment
```bash
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

# On Windows (Command Prompt):
.\.venv\Scripts\activate.bat

# On macOS/Linux:
source .venv/bin/activate
```

### Step 4: Install the dependencies
```bash
pip install -r requirements.txt
```

### Step 5: Configure your API Keys (Optional but Recommended)
1. Rename the `.env.example` file to `.env`.
2. Open it in a text editor and paste your API key:
   ```ini
   GEMINI_API_KEY=your_actual_gemini_api_key_here
   OPENAI_API_KEY=your_actual_openai_api_key_here
   ```
*(Note: If you don't set a `.env` file, you can type your API Key directly into the Streamlit sidebar when the app starts!)*

---

##  Running the Application

Start the Streamlit development server:
```bash
streamlit run app.py
```

A browser window should automatically open at `http://localhost:8501`.

### Suggested Testing Walkthrough:
1. Under **Settings** in the sidebar, select your provider (e.g. **Google Gemini**) and paste your API key if it's not pre-filled.
2. Select your **Target Career Role** (e.g. **ML Engineer**).
3. Navigate to the **Document Manager** tab in the main area.
4. Click the button **"Load Pre-made Career Samples"**. Wait a few seconds until the success notification appears. You'll see a table listing the indexed files.
5. Go back to the **Career Advisor Chat** tab.
6. Click one of the suggested questions (e.g., *"How do I optimize PyTorch model serving latency for production?"*) or type your own custom question.
7. Click the **"Citations & Retrieved Sources"** dropdown under the mentor's answer to see the exact text snippets retrieved from the database!
