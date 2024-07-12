# Emojeez 💎

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)


## Description

Emojeez 💎 is an AI-powered semantic search platform designed to help users discover and explore emojis based on text queries. This application leverages advanced natural language processing technologies to process text queries and respond with relevant emojis. With multilingual support for over 50 languages, Emojeez 💎 retrieves emojis based on a simple sentence description or a few keywords, which enables exploratory search and enhanced digital communication for users with diverse linguistic backgrounds. Experience Emojeez 💎 live [here](https://emojeez.streamlit.app/).


## Core Features

 - **Advanced Semantic Emoji Search** ✨ Utilize our embedding-based search algorithm to find emojis that best match the contextual meaning of phrases and commonly used expressions. This functionality is powered by the integration of [Qdrant](https://qdrant.tech/)'s vector database and the powerful 🤗 [Transformers](https://huggingface.co/docs/transformers/en/index) library for sentence embedding models.

- **Efficient Indexing of Over 5000 Emojis** 📑  Explore more than 5000 emojis based on the most recent Unicode standards. While each emoji is searchable by its standard name, you can also discover emojis by their common usage in today's text-based communication. For example, the text query *great ambition* returns 🚀, while *idea* returns 💡.

- **Comprehensive Multilingual Support** 🌐 Interact with the app using your native language, facilitated by a Transformer-based multilingual encoder that supports over 50 languages.

- **Intuitive User Interface** 🖥️ Emojeez features a streamlined, user-friendly interface built with [Streamlit](https://streamlit.io/), ensuring seamless navigation and an efficient user experience.


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

What you need to install the software:

- Python 3.11+
- Qdrant
- Sentence Transformers
- Streamlit


### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/badrex/emojeez.git

2. Navigate to the project directory:

    ```bash
    cd emojeez

3. Install the required packages:

    ```bash
    pip install -r requirements.txt

4. Running the Application

    ```bash
    streamlit run search_emojis_app.py

### Usage

After launching the app, you will see a text input where you can type your query. Enter a phrase or a keyword or a keyphrase, and the app will display emojis based on semantic similarity with your text input. The app use is optimized when the search query is a phrase or a full sentence.


### Examples 

Here, I provide a few exampels of text queries and the output of the search algorithm


<font size="3">

| Query | Top 10 Emojis |
|----------|----------|
| great ambition  | 🚀 ⚒ 💯 💸 🎯 🧗 🧗‍♂ 🧗🏽‍♂ 🏃‍♀️ 🧗🏾‍♂️  |
| innovative idea  | 💡 🥚 🧰 🧑‍💻 🚀 🧩 🛞 🌱 💭 🪤  |
| extinct  | 🦣 🏚️ 🦖 🏚 🧟‍♀ 🦤 🦕 🧟‍♀️ 🧟‍♂ 🧟‍♂️ |
| animal that builds dams | 🦫 🐃 🐏 🐐 🦬 🦦 🐂 🦛 🐺 🦙 |
| protect from evil eye | 🧿 👓 🥽 👁 🦹🏻 👀 🦹🏿 🛡️ 🦹🏼 🦹🏻‍♂ |
| popular sport in the USA | ⚾ 🏐 🏀 🏈 🥍 🏓 🏑 🤾‍♂ 🤾‍♂️ 🎾 |
| extraterrestrial | 👽 🛸 👾 👩🏼‍🚀 👩‍🚀 🧑‍🚀 👨‍🚀 👩🏽‍🚀 🧑🏻‍🚀 👩🏾‍🚀 |

</font>




### Contributions

If you would like to contribute to Emojeez 💎, I warmly welcome your contribution 😊 Start by forking the repository and then propose your changes through a pull request. Contributions of all kinds are welcome, from new features and bug fixes to development of test cases. Your input will be highly valued ⭐


### Get in Touch! 
Developed with 💚 by [Badr Alabsi](https://github.com/badrex) 👨🏻‍💻