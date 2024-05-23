# langchain-csv-small-challenge
Simple challenge pour l'utilisation de langchain avec un fichier CSV

### Défi pour les étudiants : Utilisation de fichiers CSV dans les vector stores avec Langchain

#### Source de Données :
[FDIC Failed Bank List dataset](https://catalog.data.gov/dataset/fdic-failed-bank-list)

#### Contexte :
Vous avez été chargé d'identifier les causes de faillite des banques en Caroline du Nord. Pour ce faire, vous utiliserez Langchain pour intégrer des données de fichiers CSV dans des vector stores et réaliser des recherches vectorielles.

### Objectif Général :
Explorer comment utiliser les fichiers CSV dans les vector stores avec Langchain pour effectuer des recherches vectorielles et répondre à des questions spécifiques basées sur ces données.

---

### Étape 1 : Installer les dépendances

**Objectif** : Préparer l'environnement en installant les bibliothèques nécessaires.

**Instructions** :
- Utilisez `pip` pour installer les bibliothèques `langchain`, `chromadb`, et `sentence-transformers`.

```bash
pip install langchain chromadb sentence-transformers
```

---

### Étape 2 : Créer la fonction d'embedding

**Objectif** : Utiliser `SentenceTransformerEmbeddings` pour créer une fonction d'embedding.

**Instructions** :
- Importez `SentenceTransformerEmbeddings` depuis `langchain.embeddings.sentence_transformer`.
- Instanciez `SentenceTransformerEmbeddings` avec le modèle `"all-MiniLM-L6-v2"`.

```python
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
```

**Conseil** : Le modèle `"all-MiniLM-L6-v2"` est performant pour créer des embeddings de phrases.

---

### Étape 3 : Charger les documents depuis le fichier CSV

**Objectif** : Utiliser `CSVLoader` pour charger les données depuis le fichier CSV.

**Instructions** :
- Importez `CSVLoader` depuis `langchain.document_loaders`.
- Instanciez `CSVLoader` avec le chemin du fichier et l'encodage.
- Chargez les documents avec `loader.load()`.

```python
from langchain.document_loaders import CSVLoader

loader = CSVLoader("./banklist.csv", encoding="windows-1252")
documents = loader.load()
```

**Conseil** : Assurez-vous que le chemin du fichier est correct et que l'encodage correspond à celui du fichier CSV.

---

### Étape 4 : Créer une base de données Chroma à partir des documents

**Objectif** : Utiliser `Chroma.from_documents` pour créer une base de données Chroma à partir des documents et de la fonction d'embedding.

**Instructions** :
- Importez `Chroma` depuis `langchain.vectorstores`.
- Utilisez `Chroma.from_documents` avec les documents et la fonction d'embedding.

```python
from langchain.vectorstores import Chroma

db = Chroma.from_documents(documents, embedding_function)
```

**Conseil** : Vérifiez que les documents sont correctement chargés avant de créer la base de données.

---

### Étape 5 : Effectuer une recherche vectorielle

**Objectif** : Utiliser `db.similarity_search(query)` pour effectuer une recherche et répondre à une question spécifique.

**Instructions** :
- Définissez une question de recherche.
- Utilisez `db.similarity_search` pour rechercher la question.
- Affichez le contenu du document le plus pertinent.

```python
query = "Did a bank fail in North Carolina?"
docs = db.similarity_search(query)
print(docs[0].page_content)
```

**Conseil** : La question doit être précise pour obtenir les résultats les plus pertinents.

---

### Objectif Avancé : Intégration de Chroma DB avec des données CSV dans une chaîne Langchain

---

### Étape 1 : Installer les dépendances

**Objectif** : Préparer l'environnement en installant des bibliothèques supplémentaires nécessaires pour l'intégration avancée.

**Instructions** :
- Utilisez `pip` pour installer les bibliothèques `langchain`, `chromadb`, `openai`, et `tiktoken`.

```bash
pip install langchain chromadb openai tiktoken
```

---

### Étape 2 : Créer la fonction d'embedding

**Objectif** : Utiliser `OpenAIEmbeddings` pour créer une fonction d'embedding.

**Instructions** :
- Importez `OpenAIEmbeddings` depuis `langchain.embeddings`.
- Instanciez `OpenAIEmbeddings`.

```python
from langchain.embeddings import OpenAIEmbeddings

embedding_function = OpenAIEmbeddings()
```

---

### Étape 3 : Charger les documents depuis le fichier CSV

**Objectif** : Utiliser `CSVLoader` pour charger les données depuis le fichier CSV.

**Instructions** :
- Importez `CSVLoader` depuis `langchain.document_loaders`.
- Instanciez `CSVLoader` avec le chemin du fichier et l'encodage.
- Chargez les documents avec `loader.load()`.

```python
from langchain.document_loaders import CSVLoader

loader = CSVLoader("./banklist.csv", encoding="windows-1252")
documents = loader.load()
```

---

### Étape 4 : Créer une base de données Chroma à partir des documents

**Objectif** : Utiliser `Chroma.from_documents` pour créer une base de données Chroma à partir des documents et de la fonction d'embedding.

**Instructions** :
- Importez `Chroma` depuis `langchain.vectorstores`.
- Utilisez `Chroma.from_documents` avec les documents et la fonction d'embedding.

```python
from langchain.vectorstores import Chroma

db = Chroma.from_documents(documents, embedding_function)
```

---

### Étape 5 : Configurer le retriever et le pipeline de question-réponse

**Objectif** : Configurer un retriever pour récupérer les documents et créer un pipeline de question-réponse en utilisant un modèle de chat.

**Instructions** :
- Importez les modules nécessaires pour le modèle de chat, le prompt template, et les parsers.
- Configurez le retriever, le prompt template, et le modèle de chat.
- Créez le pipeline de question-réponse.

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

**Conseil** : Vérifiez que chaque composant du pipeline est correctement configuré pour assurer une intégration fluide.

---

Utilisez ces instructions pour réaliser l'exercice et explorez comment les embeddings et les recherches vectorielles peuvent être intégrés dans vos projets de machine learning. Pour plus de détails, consultez [ce lien](https://how.wtf/how-to-use-csv-files-in-vector-stores-with-langchain.html).