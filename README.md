# 🤖 Computer Science Chatbot Advisor (KWASU)

An intelligent chatbot designed to assist Computer Science students at Kwara State University (KWASU) with academic guidance, course information, career advice, and technical support.

## Features

- **Intelligent Response System**: Uses sentence transformers for semantic understanding of user queries
- **Fallback AI**: Integrates DialoGPT for handling queries outside the knowledge base
- **Interactive Web Interface**: Built with Streamlit for easy deployment and usage
- **Comprehensive Knowledge Base**: Covers topics like programming languages, exam preparation, internships, tools, and career guidance
- **Real-time Chat**: Instant responses with conversation history
- **Suggested Questions**: Quick-start buttons for common queries

## Tech Stack

- **Frontend**: Streamlit
- **NLP Models**: 
  - Sentence Transformers (paraphrase-MiniLM-L6-v2)
  - Microsoft DialoGPT-small
- **ML Libraries**: PyTorch, Transformers
- **Data**: JSON-based knowledge base

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd cs-chatbot-kwasu
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## Usage

### Starting the Chatbot
- Launch the app using `streamlit run app.py`
- Access the web interface at `http://localhost:8501`
- Start asking questions or use the suggested question buttons

### Sample Questions
- "How do I register my courses?"
- "What programming languages should I learn?"
- "Give me final year project ideas"
- "How do I prepare for exams?"
- "What career options do I have after CS?"

## Project Structure

```
├── app.py                 # Main Streamlit application
├── chatbot_app.py         # Alternative app version with enhanced UI
├── chatbot_qa.json        # Knowledge base with intents and responses
├── requirements.txt       # Python dependencies
└── README.md             # Project documentation
```

## Knowledge Base

The chatbot's knowledge is stored in `chatbot_qa.json` and covers:

- **Academic Support**: Course registration, exam preparation, study tips
- **Programming**: Language recommendations, tools, development environments
- **Career Guidance**: Job opportunities, internship advice, SIWES guidance
- **Technical Topics**: Software engineering, cybersecurity, AI/ML basics
- **Department Info**: Policies, attendance rules, administrative guidance

## How It Works

1. **Input Processing**: User queries are processed using sentence transformers
2. **Similarity Matching**: Cosine similarity determines the best matching intent
3. **Response Selection**: Random response selected from matched intent category
4. **Fallback Handling**: DialoGPT generates responses for unmatched queries
5. **UI Rendering**: Streamlit displays the conversation with custom styling

## Configuration

### Similarity Threshold
The chatbot uses a similarity threshold of 0.6 to determine if a query matches the knowledge base. Queries below this threshold trigger the fallback AI response.

### Model Caching
Both sentence transformer and DialoGPT models are cached using Streamlit's caching decorators for improved performance.

## Contributing

To add new knowledge to the chatbot:

1. Edit `chatbot_qa.json`
2. Add new intents with patterns and responses
3. Follow the existing JSON structure:
```json
{
    "tag": "intent_name",
    "patterns": ["example question 1", "example question 2"],
    "responses": ["response 1", "response 2"]
}
```

## Dependencies

- `streamlit`: Web application framework
- `sentence-transformers`: Semantic text similarity
- `torch`: PyTorch for deep learning
- `transformers`: Hugging Face transformers library

## License

This project is designed for educational purposes at KWASU Computer Science Department.

## Support

For technical issues or feature requests, contact the development team or submit an issue in the repository.

---

**Developed by: Jst Kira The Dev**