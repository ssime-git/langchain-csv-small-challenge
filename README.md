### Student Challenge: Using CSV Files in Vector Stores with Langchain

#### Data Source:
[FDIC Failed Bank List dataset](https://catalog.data.gov/dataset/fdic-failed-bank-list)

#### Context:
You have been tasked with identifying the causes of bank failures in North Carolina. To do this, you will use Langchain to integrate CSV file data into vector stores and perform vector searches.

### General Objective:
Explore how to use CSV files in vector stores with Langchain to perform vector searches and answer specific questions based on this data.

---

### Step 1: Install Dependencies

**Objective**: Prepare the environment by installing the necessary libraries.

**Instructions**:
- Use `pip` to install the libraries `langchain`, `chromadb`, and `sentence-transformers`.


<details>
<summary>Show code</summary>

```bash
pip install langchain chromadb sentence-transformers
```

</details>


---

### Step 2: Create the Embedding Function

**Objective**: Use `SentenceTransformerEmbeddings` to create an embedding function.

**Instructions**:
- Import `SentenceTransformerEmbeddings` from `langchain.embeddings.sentence_transformer`.
- Instantiate `SentenceTransformerEmbeddings` with the model `"all-MiniLM-L6-v2"`.


<details>
<summary>Show code</summary>

```python
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
```

</details>

**Tip**: The `"all-MiniLM-L6-v2"` model is effective for creating sentence embeddings.

---

### Step 3: Load Documents from the CSV File

**Objective**: Use `CSVLoader` to load data from the CSV file.

**Instructions**:
- Import `CSVLoader` from `langchain.document_loaders`.
- Instantiate `CSVLoader` with the file path and encoding.
- Load the documents with `loader.load()`.


<details>
<summary>Show code</summary>

```python
from langchain.document_loaders import CSVLoader

loader = CSVLoader("./banklist.csv", encoding="windows-1252")
documents = loader.load()
```

</details>


**Tip**: Ensure the file path is correct and the encoding matches the CSV file.

---

### Step 4: Create a Chroma Database from Documents

**Objective**: Use `Chroma.from_documents` to create a Chroma database from the documents and embedding function.

**Instructions**:
- Import `Chroma` from `langchain.vectorstores`.
- Use `Chroma.from_documents` with the documents and embedding function.


<details>
<summary>Show code</summary>

```python
from langchain.vectorstores import Chroma

db = Chroma.from_documents(documents, embedding_function)
```

</details>


**Tip**: Verify that the documents are correctly loaded before creating the database.

---

### Step 5: Perform a Vector Search

**Objective**: Use `db.similarity_search(query)` to perform a search and answer a specific question.

**Instructions**:
- Define a search query.
- Use `db.similarity_search` to search the query.
- Display the content of the most relevant document.


<details>
<summary>Show code</summary>

```python
query = "Did a bank fail in North Carolina?"
docs = db.similarity_search(query)
print(docs[0].page_content)
```

</details>


**Tip**: The question should be precise to obtain the most relevant results.

---

### Advanced Objective: Integrating Chroma DB with CSV Data in a Langchain Pipeline

---

### Step 1: Install Dependencies

**Objective**: Prepare the environment by installing additional libraries for advanced integration.

**Instructions**:
- Use `pip` to install the libraries `langchain`, `chromadb`, `openai`, and `tiktoken`.


<details>
<summary>Show code</summary>

```bash
pip install langchain chromadb openai tiktoken
```

</details>


---

### Step 2: Create the Embedding Function

**Objective**: Use `OpenAIEmbeddings` to create an embedding function.

**Instructions**:
- Import `OpenAIEmbeddings` from `langchain.embeddings`.
- Instantiate `OpenAIEmbeddings`.


<details>
<summary>Show code</summary>

```python
from langchain.embeddings import OpenAIEmbeddings

embedding_function = OpenAIEmbeddings()
```

</details>


---

### Step 3: Load Documents from the CSV File

**Objective**: Use `CSVLoader` to load data from the CSV file.

**Instructions**:
- Import `CSVLoader` from `langchain.document_loaders`.
- Instantiate `CSVLoader` with the file path and encoding.
- Load the documents with `loader.load()`.


<details>
<summary>Show code</summary>

```python
from langchain.document_loaders import CSVLoader

loader = CSVLoader("./banklist.csv", encoding="windows-1252")
documents = loader.load()
```

</details>


---

### Step 4: Create a Chroma Database from Documents

**Objective**: Use `Chroma.from_documents` to create a Chroma database from the documents and embedding function.

**Instructions**:
- Import `Chroma` from `langchain.vectorstores`.
- Use `Chroma.from_documents` with the documents and embedding function.


<details>
<summary>Show code</summary>

```python
from langchain.vectorstores import Chroma

db = Chroma.from_documents(documents, embedding_function)
```

</details>


---

### Step 5: Configure the Retriever and Question-Answer Pipeline

**Objective**: Configure a retriever to retrieve documents and create a question-answer pipeline using a chat model.

**Instructions**:
- Import the necessary modules for the chat model, prompt template, and parsers.
- Configure the retriever, prompt template, and chat model.
- Create the question-answer pipeline.


<details>
<summary>Show code</summary>

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

retriever = db.as_retriever()

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
model = ChatOpenAI()

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

print(chain.invoke("What bank failed in North Carolina?"))
```

</details>


**Tip**: Ensure each component of the pipeline is correctly configured for smooth integration.

---

Use these instructions to complete the exercise and explore how embeddings and vector searches can be integrated into your machine learning projects. For more details, see [this link](https://how.wtf/how-to-use-csv-files-in-vector-stores-with-langchain.html).