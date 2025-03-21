# LLM-for-Data-Analysis

## 📌 Project Overview

This project leverages **Large Language Models (LLMs)** to streamline data analysis tasks, specifically focusing on:

- **Automated SQL Query Generation**
- **Dynamic Data Visualization**

Built for **Scibids (DoubleVerify)**, our goal is to make complex advertising datasets more accessible and insightful for stakeholders.

---

## 🔧 Current Implementation

### 🗃️ SQL Query Generator (`SQL Generator Code`)

We've developed a **dual-LLM prompt-flow** solution:

- **Table Recognition LLM**: Interprets natural language queries to identify relevant database tables efficiently.
- **SQL Generation LLM**: Generates accurate SQL queries based on identified tables and their schemas.

This two-step method optimizes SQL query generation by avoiding the common context-length limitations of single-step prompting.

### 📊 Plot generator (`Plot_generator`)

Our Python visualization pipeline is designed to automatically:

- Generate SQL queries from user input.
- Execute queries and retrieve data from BigQuery.
- Generate initial data visualizations.
-give comments and analysis.
Currently serving as a proof-of-concept (our second implementation path), this component offers foundational visualization capabilities that can be significantly expanded in future iterations.

### 📊 Feedback Loop (`Feedback_loop`)
To expand the dataset for model training, a feedback loop was incorporated, allowing the user to evaluate whether the generated query matches their intended query. If it does, the query is added to the dataset.

### sql_query_and_plot_generator

This project combines all the explored work into a final product that is capable of:

- **Generating SQL queries**  
- **Creating plot visualizations**  
- **Commenting on the results**  
- **Incorporating a feedback loop**: Users can evaluate whether the generated query matches their intended query. If it does, the query is added to the dataset, helping expand the model's training data.

The project brings together multiple components into a single solution for automated SQL query generation, data visualization, and continuous improvement through user feedback.


## 🔮 Future Enhancements & Roadmap

### 🌟 LangChain Integration

We propose integrating **LangChain** to significantly enhance our pipeline through:

- **Context Management**: Improved session continuity and context-aware responses.
- **Prompt Chaining**: Advanced logical reasoning through chained AI prompts.
- **Modularity & Extensibility**: Easier integration of additional data sources and tools.

### 📈 Advanced Visualization Features

Potential visualization enhancements include:

- Integration of interactive visualizations using libraries such as Plotly, Dash, or advanced Streamlit components.
- Automatic insight extraction and annotation through advanced AI analytics.

---

## ⚙️ Setup and Usage Instructions

Follow these steps to set up and run the project locally using **VS Code** and **Google Cloud SDK**:

### **1️⃣ Install Google Cloud SDK**
- [Google Cloud SDK Installation Guide](https://cloud.google.com/sdk/docs/install)
- Log in with your Google account associated with the Capstone project.

---

### **2️⃣ Open VS Code**
- Launch **Visual Studio Code**.

---

### **3️⃣ Create & Activate a Virtual Environment (`venv`)**

**Create**:
```bash
python -m venv .venv
```

**Activate**:

**Windows (PowerShell)**:
```powershell
.venv\Scripts\activate
```

**Mac/Linux**:
```bash
source .venv/bin/activate
```

### **4️⃣ Create a Jupyter Notebook in VS Code**
- Create a new `.ipynb` file in VS Code.
- Click on **"Select Kernel"** in the top right corner.
- Choose **Python 3.12 (installed with Google Cloud SDK)**.

### **5️⃣ Select the Correct Python Interpreter (if needed)**
- If Python 3.12 is not listed, manually add its path.
- **Potential Path**:  
  ```
  C:\Users\``user``\AppData\Local\Google\Cloud SDK\google-cloud-sdk\platform\bundledpython
  ```

### **6️⃣ Authenticate GCloud in Terminal**
- Run:
```bash
gcloud auth application-default login
```
- Log in with your Google account.

### **7️⃣ Verify Authentication**
- Run:
```bash
gcloud auth list
```

### **8️⃣ Run Streamlit Interface**
- In the terminal, run:
```bash
streamlit run code/main.py
```

### 🚩 Next Steps
- Explore LangChain for enhanced system flexibility and accuracy.
- Implement advanced visualization features for richer data insights.
