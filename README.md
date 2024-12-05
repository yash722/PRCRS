
# Personalized Restaurant And Cafe Recommendation System

This repository contains Python notebooks designed to preprocess data, implement a Retrieval-Augmented Generation (RAG) system, and create a vector database using the Yelp Dataset. These notebooks collectively enable a recommendation system that provides personalized suggestions for restaurants and cafes based on user queries.

## Contents

1. **Dataset-Preprocessing.ipynb**
   - This notebook handles the preprocessing of the Yelp Dataset.
   - Key Features:
     - Loads and filters the dataset.
     - Extracts relevant fields for the recommendation system.
     - Performs data cleaning and transformation.

2. **Vector-Database-Creation.ipynb**
   - Creates a vector database from the preprocessed dataset.
   - Key Features:
     - Converts textual data into embeddings.
     - Stores embeddings for efficient retrieval in the RAG pipeline with relevant metadata.
     - Optimized for scalability and fast query execution.

3. **RAG.ipynb**
   - Implements a Retrieval-Augmented Generation (RAG) pipeline.
   - Key Features:
     - Handles user text queries to generate recommendations.
     - Leverages preprocessed data and vectorized embeddings for retrieval.
     - Integrates an LLM for response generation.

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook or an equivalent environment
- Required Python libraries:
  - `pandas`
  - `numpy`
  - `chromadb` (or another vector database library)
  - `openai` (for LLM integration)

### Usage
1. Open the notebooks in your Jupyter environment.
2. Run the `Dataset-Preprocessing.ipynb` notebook to preprocess the Yelp dataset.
3. Use `Vector-Database-Creation.ipynb` to create a vector database.
4. Execute `RAG.ipynb` to test the RAG-based recommendation system.

## Future Work
- Enhance vectorization techniques for better recommendations.
- Multi-Query Transalation for better recommendations
- Geoloacation Filtering
- Add support for additional datasets and multilingual queries.
- Implement a web-based interface for end-user interactions.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request to suggest improvements or add features.

## Acknowledgments
- Yelp Dataset for providing rich and diverse data.
- OpenAI for their LLM APIs.
- Developers and contributors to open-source libraries used in this project (Yash Rao, Shreyansh Kumar, Palak Kothari).
