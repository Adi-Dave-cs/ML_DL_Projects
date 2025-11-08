# 📄 AI Brochure Maker

An intelligent web application that automatically generates professional company brochures from website URLs using advanced AI models. Simply provide a website URL, and the application will scrape, analyze, and create a comprehensive brochure in your preferred style.

## ✨ Features

- **Multi-Provider AI Support**: Compatible with OpenAI, Anthropic (Claude), and Google Gemini
- **Intelligent Web Scraping**: Automatically extracts relevant content from websites
- **Smart Link Selection**: Uses AI to identify and include relevant company pages (About, Careers, etc.)
- **Multiple Brochure Styles**: Choose from Professional, Creative, Minimalist, Corporate, or Modern styles
- **Customizable Parameters**: Adjust temperature and token limits for fine-tuned control
- **Multiple Export Formats**: Download as TXT, Markdown, or HTML
- **User-Friendly Interface**: Clean, intuitive Streamlit-based UI
- **Additional Context Support**: Add custom instructions to personalize your brochure

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- API key from at least one of the following providers:
  - OpenAI
  - Anthropic
  - Google (Gemini)

### Installation

1. **Clone the repository** (or download the files)

2. **Install required dependencies**:
```bash
pip install streamlit openai anthropic google-generativeai requests beautifulsoup4
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

### Requirements.txt

Create a `requirements.txt` file with:
```
streamlit>=1.28.0
openai>=1.0.0
anthropic>=0.7.0
google-generativeai>=0.3.0
requests>=2.31.0
beautifulsoup4>=4.12.0
```

## 🎯 Usage

1. **Start the application**:
```bash
streamlit run app.py
```

2. **Configure your settings** (in the sidebar):
   - Select your AI provider (OpenAI, Anthropic, or Google Gemini)
   - Enter your API key
   - Choose your preferred model
   - Adjust advanced settings (temperature, max tokens) if needed

3. **Generate a brochure**:
   - Enter the website URL you want to create a brochure from
   - (Optional) Add additional context or instructions
   - Select your preferred brochure style
   - Click "🚀 Generate Brochure"

4. **Download your brochure**:
   - Once generated, download in your preferred format (TXT, MD, or HTML)

## 🔧 Configuration

### API Providers

#### OpenAI
- **Models**: gpt-4o, gpt-4-turbo ..(more)
- **Get API Key**: [OpenAI Platform](https://platform.openai.com/api-keys)

#### Anthropic (Claude)
- **Models**: claude-sonnet-4-5-20250929, claude-3-5-sonnet-20241022, claude-3-opus-20240229
- **Get API Key**: [Anthropic Console](https://console.anthropic.com/)

#### Google (Gemini)
- **Models**: gemini-pro, gemini-1.5-pro
- **Get API Key**: [Google AI Studio](https://makersuite.google.com/app/apikey)

### Advanced Settings

- **Temperature** (0.0 - 1.0): Controls creativity vs. accuracy
  - Lower values (0.1-0.3): More focused and factual
  - Higher values (0.7-1.0): More creative and diverse
  
- **Max Tokens** (100 - 4000): Controls the length of generated content

### Brochure Styles

- **Professional**: Clean, business-focused tone
- **Creative**: Engaging and imaginative approach
- **Minimalist**: Simple, to-the-point style
- **Corporate**: Formal, enterprise-level presentation
- **Modern**: Contemporary, trendy language and format

## 📋 How It Works

1. **URL Submission**: User provides a website URL
2. **Content Extraction**: Application scrapes the main page content
3. **Link Analysis**: AI identifies relevant pages (About, Careers, etc.)
4. **Deep Scraping**: Fetches content from identified relevant pages
5. **AI Processing**: Analyzes all collected content
6. **Brochure Generation**: Creates a structured, markdown-formatted brochure
7. **Output**: Displays result with download options

## 🛠️ Technical Details

### Architecture

```
User Input → Model Initialization → Web Scraping → AI Analysis → Content Generation → Output
```

### Key Components

- **`initialize_model()`**: Sets up the AI client based on provider
- **`fetch_information_from_web()`**: Extracts content from web pages
- **`select_relevant_links()`**: Uses AI to identify important links
- **`create_brochure_helper()`**: Orchestrates the brochure generation process
- **`generate_responses()`**: Handles API calls to different AI providers

### Error Handling

The application includes comprehensive error handling for:
- Invalid API keys
- Network request failures
- Malformed URLs
- AI API errors
- Web scraping issues

## 🔒 Security & Privacy

- API keys are entered securely (password-masked input)
- Keys are not stored or logged
- All data processing happens in real-time
- No user data is persisted

## ⚠️ Limitations

- Web scraping may fail on JavaScript-heavy sites
- Some websites may block automated access
- Content is truncated to 5,000 characters for processing
- AI responses depend on model capabilities and API limits
- Rate limits apply based on your API provider plan

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🐛 Troubleshooting

### Common Issues

**"Failed to initialize model"**
- Verify your API key is correct
- Check your internet connection
- Ensure you have API credits/quota remaining

**"Error fetching URL"**
- Verify the URL is accessible
- Check if the website blocks automated access
- Try a different website

**"Error generating response"**
- Check your API quota/limits
- Reduce max_tokens if hitting limits
- Verify the model name is correct

**Slow generation**
- This is normal for comprehensive websites
- Complex sites with many links take longer
- Consider using a faster model

## 📧 Support

For issues, questions, or contributions, please open an issue on GitHub or contact the maintainer.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Powered by OpenAI, Anthropic, and Google AI APIs
- Web scraping with [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/)
