# Chatabot: Espiando a la Competencia

This repository contains the chatbot developed for the final group project of the Text Mining course in the Data Mining Master's program at UBA.
Project Overview

The project focused on creating:

* A Weekly Review summarizing the activities of competitors for a consulting firm.
* A chatbot allowing employees to interact and gain insights about competitors' recent news.

To achieve this, we performed web scraping on company websites, Yahoo Finance, and Google News to gather the latest information. Leveraging this data, the Weekly Review and chatbot were developed using GPT-based large language models (LLMs).
Chatbot Details

The chatbot was implemented as a Conversational-RAG (Retrieval-Augmented Generation) using:

* LangChain for conversational capabilities.
* Pinecone as the vector database for document retrieval.

![imagen](https://github.com/user-attachments/assets/663ad1f3-1a45-41d7-85b9-cbabdd0113cb)

The chatbot is hosted as a Streamlit app, accessible at: textminingchatbot.streamlit.app.
