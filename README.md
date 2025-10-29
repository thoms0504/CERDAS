# CERDAS

## Overview

This repository contains the codebase for CERDAS, a Python-based application that leverages data analysis, machine learning, and NLP to provide intelligent recommendations. It seems designed to assist with PNS (Pegawai Negeri Sipil) or Civil Servant related tasks, potentially job matching or skill analysis. It uses Streamlit for a user-friendly interface.

## Key Features & Benefits

- **Data-Driven Recommendations:** Utilizes data analysis and potentially machine learning models to provide informed suggestions.
- **User-Friendly Interface:** Built with Streamlit for easy interaction and visualization.
- **Flexible Configuration:** Offers customizable settings to tailor the application to specific needs.
- **Comprehensive Data Handling:** Includes scripts for generating sample data and processing various data sources.
- **Search History:** Tracks user searches, potentially for improved recommendations or analysis.

## Prerequisites & Dependencies

Before running this application, ensure you have the following installed:

- **Python:** Version 3.8 or higher is recommended.
- **Pip:** Python package installer.

The following Python packages are required. They can be installed using pip:

```bash
pip install -r requirements.txt
```

Here's a breakdown of the key dependencies:

| Library                 | Version  | Description                                                                 |
|-------------------------|----------|-----------------------------------------------------------------------------|
| streamlit              | >=1.32.0 | Framework for creating interactive web applications.                          |
| pandas                 | >=2.0.0  | Data analysis and manipulation library.                                       |
| numpy                  | >=1.24.0 | Numerical computing library.                                                  |
| scikit-learn           | >=1.3.0  | Machine learning library.                                                      |
| sentence-transformers  | >=2.5.0  | Library for generating sentence embeddings.                                     |
| rank-bm25              | >=0.2.2  | Implementation of the BM25 ranking function.                                  |
| fuzzywuzzy             | >=0.18.0 | Library for fuzzy string matching.                                            |
| python-Levenshtein      | >=0.23.0 |  Library that improves the performance of fuzzywuzzy                           |
| openai                 | >=1.12.0 |  Library for accessing OpenAI API (if LLM integration is used).               |
| google-generativeai    | >=0.3.0  | Library for accessing Google Generative AI models (if LLM integration is used). |
| plotly                 | >=5.18.0 | Interactive plotting library.                                                |
| openpyxl               | >=3.1.0  | Library for reading and writing Excel files.                                  |
| xlsxwriter             | >=3.1.9  | Library for writing Excel files.                                              |
| python-dateutil        | >=2.8.2  | Extensions to the standard Python datetime module.                           |

## Installation & Setup Instructions

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/thoms0504/CERDAS.git
    cd CERDAS
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Configuration:**
    - Review the `config.py` file and adjust settings such as the embedding model based on your needs and resources.
    - If using OpenAI or Google Generative AI, ensure you have set up the necessary API keys and configured them appropriately, usually in the `secrets.toml` file or environment variables.
    -  (Optional) Populate the `secrets.toml` file with your OpenAI or Google API keys as needed. Streamlit recognizes secrets defined here.

5.  **Run the application:**

    ```bash
    streamlit run app_optimized.py
    ```

## Usage Examples

The main entry point for the application is `app_optimized.py`.  Once you run this file using Streamlit, a web browser will open with the user interface.  Interact with the application through the web interface to explore its features.

The application likely involves:

-   Data input (e.g., uploading CSV files).
-   Searching for specific items or categories.
-   Generating recommendations based on search criteria and data.
-   Displaying results in a structured format.

## Configuration Options

The `config.py` file allows you to customize various aspects of the application. Key configuration options include:

-   **Embedding Model:** Select the SentenceTransformer model to use for generating text embeddings.  Options include `"paraphrase-multilingual-MiniLM-L12-v2"` (default) and `"paraphrase-multilingual-mpnet-base-v2"`.
-   **Other parameters:** The file may contain other adjustable parameters related to data processing, search algorithms, and the behavior of specific functionalities.

API keys for external services (like OpenAI or Google AI) are typically stored in the `.streamlit/secrets.toml` file.  Refer to the Streamlit documentation for details on how to manage secrets:

```toml
# .streamlit/secrets.toml
OPENAI_API_KEY = "your_openai_api_key"
GOOGLE_API_KEY = "your_google_api_key"
```

## Contributing Guidelines

We welcome contributions to this project! To contribute:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Implement your changes, ensuring thorough testing.
4.  Submit a pull request with a clear description of your changes.

Please adhere to the existing coding style and conventions.  Include relevant documentation for any new features.

## License Information

License information is not specified. All rights are reserved by the owner, thoms0504.

## Acknowledgments

- This project utilizes several open-source libraries, including Streamlit, pandas, scikit-learn, and Sentence Transformers.  We thank the developers and maintainers of these libraries for their contributions to the open-source community.
