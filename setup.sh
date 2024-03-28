#!/bin/bash
# Activate the virtual environment
source /workspaces/pdf_chat_langchain/venv/bin/activate

# Download SpaCy model
python -m spacy download en_core_web_sm
