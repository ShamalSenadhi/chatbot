# E-commerce Customer Support Chatbot

A Streamlit-based customer support chatbot for e-commerce platforms built with LangChain, FAISS, and Hugging Face transformers.

## Features

- **Web Data Loading**: Load product information directly from e-commerce URLs
- **Vector Search**: FAISS-powered similarity search for accurate information retrieval
- **Interactive Chat Interface**: User-friendly Streamlit interface
- **Real-time Responses**: Get instant answers about products, prices, and offers
- **Configurable**: Easy to modify URLs and parameters

## Project Structure

```
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── packages.txt        # System packages for Streamlit Cloud
├── README.md          # Project documentation
└── .streamlit/
    └── config.toml    # Streamlit configuration (optional)
```

## Installation & Setup

### Local Development

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd ecommerce-chatbot
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
streamlit run app.py
```

### Streamlit Cloud Deployment

1. **Push to GitHub**
```bash
git add .
git commit -m "Initial commit"
git push origin main
```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repository
   - Select the repository and branch
   - Set `app.py` as the main file
   - Deploy!

## Usage

1. **Load Data**: Enter a URL in the sidebar and click "Load Data"
2. **Ask Questions**: Use the chat interface to ask about products, prices, or offers
3. **Get Answers**: The chatbot will provide relevant information based on the loaded data

### Sample URLs

- Default: mCentre special offers page
- You can use any e-commerce URL with structured product information

### Sample Questions

- "What is the price of Canon PIXMA E470?"
- "Is Canon PIXMA E470 in stock?"
- "What special offers are available?"
- "What products are available for students?"

## Technical Details

### Architecture

- **Frontend**: Streamlit for web interface
- **Backend**: LangChain for document processing and retrieval
- **Vector Store**: FAISS for similarity search
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Text Processing**: Unstructured for web content parsing
- **LLM**: Hugging Face Transformers (configurable)

### Key Components

1. **Document Loading**: UnstructuredURLLoader fetches web content
2. **Text Splitting**: RecursiveCharacterTextSplitter chunks documents
3. **Embeddings**: HuggingFace embeddings convert text to vectors
4. **Vector Store**: FAISS enables fast similarity search
5. **QA Chain**: RetrievalQA combines retrieval with generation

## Configuration

### Model Settings

- **Chunk Size**: 200 characters (adjustable in code)
- **Chunk Overlap**: 20 characters
- **Embedding Model**: all-MiniLM-L6-v2
- **Max Tokens**: 200 for responses

### Environment Variables

For production deployment, consider setting:

```bash
export TRANSFORMERS_CACHE=/tmp/transformers_cache
export HF_DATASETS_CACHE=/tmp/datasets_cache
```

## Customization

### Adding New URLs

Modify the `default_url` in `app.py` or add multiple URLs:

```python
urls = [
    "https://example.com/products",
    "https://example.com/offers"
]
```

### Changing Models

Replace the embedding model:

```python
embeddings = HuggingFaceEmbeddings(model_name="your-preferred-model")
```

### Adjusting Chunk Size

Modify text splitting parameters:

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Increase for more context
    chunk_overlap=50
)
```

## Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce chunk size or use smaller models
2. **URL Loading Errors**: Check if the website allows scraping
3. **Model Loading**: Ensure stable internet for first-time model downloads
4. **Dependency Conflicts**: Use virtual environments

### Performance Tips

- Use GPU-enabled deployment for faster processing
- Cache embeddings to avoid recomputation
- Implement session management for multiple users
- Consider using faster embedding models for production

## Limitations

- Depends on website structure for data extraction
- Limited by the LLM's knowledge and context window
- Requires internet connection for model downloads
- Performance varies with document size and complexity

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- LangChain for the framework
- Hugging Face for pre-trained models
- Streamlit for the web interface
- FAISS for efficient vector search
- The Open University of Sri Lanka for the project inspiration


