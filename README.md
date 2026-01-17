# 📄 Resume Tailor Agent

An AI-powered application that automatically optimizes your resume to match any job description using LangChain and OpenAI's GPT models.

## 🎯 Project Overview

This intelligent agent analyzes job descriptions and rewrites your resume to highlight relevant skills, experience, and keywords while maintaining authenticity. Built as part of my AI learning journey - Day 11 of exploring AI agents and LangChain!

## ✨ Features

- **Smart Resume Optimization**: Automatically tailors your resume to match job requirements
- **ATS-Friendly**: Optimizes for Applicant Tracking Systems with relevant keywords
- **AI-Powered Analysis**: Provides detailed insights on what changed and why
- **User-Friendly Interface**: Clean Streamlit web interface
- **Secure**: API keys stored securely in environment variables
- **Download Option**: Export your tailored resume instantly

## 🛠️ Tech Stack

- **LangChain**: Framework for building LLM applications
- **OpenAI GPT-4o-mini**: Advanced language model for resume optimization
- **Streamlit**: Web application framework
- **Python 3.9+**: Core programming language
- **python-dotenv**: Environment variable management

## 📋 Prerequisites

- Python 3.9 or higher
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))
- Basic understanding of Python

## 📁 Project Structure

```
resume-tailor-project/
│
├── .env                    # API keys (DO NOT COMMIT)
├── .gitignore             # Git ignore file
├── resume.py              # Main application file
├── check.ipynb            # upload api key
├── requirements.txt       # Python dependencies
└── README.md             # Project documentation
```

## 💡 How It Works

1. **Input**: Paste your current resume and target job description
2. **Analysis**: AI analyzes the job requirements and identifies key skills
3. **Optimization**: Resume is rewritten to emphasize relevant experience
4. **Output**: Get a tailored, ATS-optimized resume ready for submission

### Technical Flow

```python
User Input → LangChain Prompt Template → OpenAI GPT-4o-mini → Optimized Resume
```

The app uses:
- `ChatPromptTemplate` for structured prompts
- `ChatOpenAI` for LLM integration
- `StrOutputParser` for response processing
- LCEL (LangChain Expression Language) chains for workflow

## 🎓 What I Learned

- Building AI agents with LangChain
- Prompt engineering for specific tasks
- Integrating LLMs into web applications
- Managing API keys securely
- Creating user-friendly AI interfaces
- Chain-based LLM workflows

## 🔒 Security Notes

- API keys are stored in `.env` file (never hardcoded)
- `.env` is included in `.gitignore`
- User data is not stored or logged
- All processing happens in real-time

## 💰 Cost Estimation

Using GPT-4o-mini:
- ~$0.01-0.05 per resume tailoring
- $5 credit = ~100-500 resume optimizations
- Very affordable for personal use


## 🙏 Acknowledgments

- OpenAI for GPT models
- LangChain team for the amazing framework
- Streamlit for the easy-to-use web framework
