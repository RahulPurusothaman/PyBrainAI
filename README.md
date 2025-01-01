Here's an updated **README.md** with emojis included to make it more engaging:

---

# PyBrainAI 🤖

**PyBrainAI** is an intelligent **text-based query system** that leverages **advanced embedding techniques** and **cosine similarity** to provide detailed, accurate answers to **Python**-related queries. The system focuses on delivering answers related to **Python language fundamentals** and **Object-Oriented Programming (OOP)** concepts, ensuring that users receive relevant information within seconds. ⏱️

## **Concept & Aim 🎯**

The primary goal of **PyBrainAI** is to bridge the gap between **Python learners** and relevant information. By utilizing state-of-the-art **Sentence Transformers** and **semantic similarity matching**, the system is capable of understanding and responding to queries related to **Python programming** efficiently. Whether you're a beginner looking for an introduction to Python's core concepts or an advanced user seeking clarification on complex OOP topics, **PyBrainAI** provides answers quickly and accurately. 🧑‍💻

---

## **Features 🌟**
- **Advanced Embedding Techniques**: Converts Python-related content into embeddings using a pre-trained transformer model to capture the semantic meaning. 🔑
- **Cosine Similarity**: Matches user queries to relevant content based on semantic similarity, providing more contextually accurate answers. 🧠
- **Python Fundamentals & OOP Focus**: Tailored to help users learn Python programming, with special emphasis on core programming concepts and OOP. 🐍💻
- **Real-Time Response**: Retrieves the most relevant answer in seconds, ensuring a seamless learning experience. ⚡

---

## **How It Works ⚙️**

1. **Model Loading and Embedding Generation**:
   - A pre-trained **Sentence Transformer** model (`'all-MiniLM-L6-v2'`) is used to generate embeddings for Python-related content. These embeddings represent the semantic meaning of the text. 🔍

2. **Saving Model and Data**:
   - The model, generated embeddings, and the associated Python-related topics and contents are saved for future use. This allows for quick retrieval during query processing. 💾

3. **Query Processing**:
   - When a query is made, the system converts the query into an embedding and compares it to precomputed content embeddings using **cosine similarity**. 🧩
   - A similarity threshold is applied to filter content based on how relevant it is to the user's query. 🎯

4. **Output**:
   - The system returns the most relevant **Python topic** and **content** that matches the user's query. 📝

---

## **Installation 🛠️**

### Prerequisites ⚡
- Python 3.x
- pip

### Steps 📝
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/pybrainai.git
   cd pybrainai
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## **Usage 🚀**

1. **Prepare Data**: The content (Python-related text data) and topics need to be available. Modify the `contents` and `topics` variables in the script to fit your dataset. 🗂️
2. **Run the Script**:
   - Example of running a query:
     ```python
     query = "What is a class in Python?"
     response = get_response_with_fixed_threshold(query, model, embeddings, topics, contents)
     print("Answer:", response['content'])
     ```

3. **Saved Files**:
   - Model: `saved_model_with_data/pythonmodel`
   - Embeddings: `saved_model_with_data/embeddings.pt`
   - Topics: `saved_model_with_data/topics.json`
   - Contents: `saved_model_with_data/contents.json`

---

## **Example Code 📄**

```python
from sentence_transformers import SentenceTransformer
import torch
import json
import random
import os

# Load model and precomputed embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
with open("saved_model_with_data/topics.json", "r") as file:
    topics = json.load(file)
with open("saved_model_with_data/contents.json", "r") as file:
    contents = json.load(file)
embeddings = torch.load("saved_model_with_data/embeddings.pt")

# Function to retrieve the most relevant response
def get_response_with_fixed_threshold(query, model, embeddings, topics, contents):
    query_embedding = model.encode(query, convert_to_tensor=True)
    scores = torch.cosine_similarity(query_embedding, embeddings)
    
    thresholds = [0.65, 0.70, 0.75, 0.80, 0.85]
    threshold = random.choice(thresholds)
    filtered_indices = [i for i, score in enumerate(scores) if score >= threshold]
    
    if not filtered_indices:
        threshold = 0.2
        filtered_indices = [i for i, score in enumerate(scores) if score >= threshold]

    selected_index = random.choice(filtered_indices)
    return {'topic': topics[selected_index], 'content': contents[selected_index]}

# Example query
query = "What is a class in Python?"
response = get_response_with_fixed_threshold(query, model, embeddings, topics, contents)
print("Answer:", response['content'])
```

---

## **Files Overview 📂**
- `model.py`: Contains logic for loading the model, generating embeddings, and saving them. 🗃️
- `data.py`: Handles data saving/loading operations for topics, content, and embeddings. 🔑
- `retrieve.py`: Implements the query handling logic, including similarity comparison and response selection. 🔍
- `requirements.txt`: Lists the dependencies required to run the project. 📋

---

## **Dependencies 🔧**
- `sentence-transformers`
- `torch`
- `json`
- `random`
- `os`

Install the dependencies with:
```bash
pip install -r requirements.txt
```

---


