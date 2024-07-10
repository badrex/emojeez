import streamlit as st
import numpy as np
import pickle
from typing import Dict, List, Any
from sentence_transformers import SentenceTransformer
from qdrant_client import models, QdrantClient
import emoji as em
import warnings

warnings.filterwarnings('ignore')

# A function to load the sentence encoder model
@st.cache_resource(show_spinner=False)
def load_encoder(model_name: str) -> SentenceTransformer:
    """Load a sentence encoder model from Hugging Face Hub."""

    sentence_encoder = SentenceTransformer(model_name)
    #st.session_state.sentence_encoder = sentence_encoder
    return sentence_encoder

# A function to load the Qdrant vector DB client
@st.cache_resource(show_spinner=False)
def load_qdrant_client(emoji_dict: Dict[str, Dict[str, Any]]) -> QdrantClient:
    """
    Load a Qdrant client and populate the database with embeddings.
    """
    # Setup the Qdrant client and populate the database
    vector_DB_client = QdrantClient(":memory:")
    embedding_dict = {
        emoji: np.array(metadata['embedding']) 
        for emoji, metadata in emoji_dict.items()
    }

    # Remove the embeddings from the dictionary so it can be used 
    # as payload in Qdrant
    for emoji in list(emoji_dict):
        del emoji_dict[emoji]['embedding']

    embedding_dim = next(iter(embedding_dict.values())).shape[0]

    # Create collection in Qdrant
    vector_DB_client.create_collection(
        collection_name="EMOJIS",
        vectors_config=models.VectorParams(
            size=embedding_dim, 
            distance=models.Distance.COSINE
        ),
    )

    # Upload points to the collection
    vector_DB_client.upload_points( 
        collection_name="EMOJIS",
        points=[
            models.PointStruct(
                id=idx, 
                vector=embedding_dict[emoji].tolist(),
                payload=emoji_dict[emoji]
            )
            for idx, emoji in enumerate(emoji_dict)
        ],
    )

    #st.session_state.vector_DB_client = vector_DB_client
    return vector_DB_client   


# for the offline version this code was faster, but resulted in a resource 
# limits error from online streamlit app 
# it seems that each user has its own session, thus caching does not help
# much here, and the resources are loaded for each user 
# def load_resources():
#     if ('vector_DB_client' not in st.session_state 
#             or 'sentence_encoder' not in st.session_state):
        
#         # Load emoji dictionary
#         with open('emoji_embeddings_dict.pkl', 'rb') as file:
#             emoji_dict = pickle.load(file)

#         # Load sentence encoder
#         embedding_model = 'paraphrase-multilingual-MiniLM-L12-v2'
#         sentence_encoder = SentenceTransformer(embedding_model)
#         st.session_state.sentence_encoder = sentence_encoder

#         # Setup the Qdrant client and populate the database
#         vector_DB_client = QdrantClient(":memory:")
#         embedding_dict = {
#             emoji: np.array(data['embedding']) 
#             for emoji, data in emoji_dict.items()
#         }

#         for emoji in list(emoji_dict):
#             del emoji_dict[emoji]['embedding']

#         embedding_dim = next(iter(embedding_dict.values())).shape[0]

#         # Create collection in Qdrant
#         vector_DB_client.create_collection(
#             collection_name="EMOJIS",
#             vectors_config=models.VectorParams(
#                 size=embedding_dim, 
#                 distance=models.Distance.COSINE
#             ),
#         )

#         # Upload points to the collection
#         vector_DB_client.upload_points(
#             collection_name="EMOJIS",
#             points=[
#                 models.PointStruct(
#                     id=idx, 
#                     vector=embedding_dict[emoji].tolist(),
#                     payload=emoji_dict[emoji]
#                 )
#                 for idx, emoji in enumerate(emoji_dict)
#             ],
#         )
        
#         st.session_state.vector_DB_client = vector_DB_client


#@st.cache_resource
def return_similar_emojis(
        embedding_model: SentenceTransformer,
        vector_DB_client: QdrantClient,
        query: str) -> List[str]:
    """
    Return similar emojis to the query using the sentence encoder and Qdrant. 
    """

    # Embed the query
    query_vector = embedding_model.encode(query).tolist()

    hits = vector_DB_client.search(
        collection_name="EMOJIS",
        query_vector=query_vector,
        limit=100,
    )

    search_emojis = []

    # only add to list if it is not already an item in the list
    for hit in hits:
        if hit.payload['Emoji'] not in search_emojis:
            search_emojis.append(hit.payload['Emoji'])

    return search_emojis

def main():

    # Load the sentence encoder model
    #if 'sentence_encoder' not in st.session_state:
    model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
    sentence_encoder = load_encoder(model_name)



    # Load the Qdrant client
    #if 'vector_DB_client' not in st.session_state:
    embedding_dict_path = 'data/emoji_embeddings_dict.pkl'

    with open(embedding_dict_path, 'rb') as file:
        embedding_dict = pickle.load(file)

    vector_DB_clinet = load_qdrant_client(embedding_dict)    

    st.title("Emojeez 💎 ")

    app_description = """
        AI-powered semantic search for emojis with multilingual support 🌐 
    """
    st.text(app_description) 
    #query = st.text_input("Enter your search query", "")




    # Using columns to layout the input and button next to each other
    with st.form(key="search_form", border=True, ):

        instr = "Enter your search query here"


        col1, col2 = st.columns([3.5, 1])

        with col1:
            query = st.text_input(
                instr, #"Enter text query here...",
                value="",
                placeholder=instr,
                label_visibility='collapsed',
                #label_visibility='visible', 
                #help="exploration discovery", 

            ) #Enter your search query

        with col2:
            trigger_search = st.form_submit_button(
                label="Search ✨", 
                use_container_width=True
            )


        if trigger_search:
            if query:
                results = return_similar_emojis(
                    sentence_encoder,
                    vector_DB_clinet,
                    query
                )

                if results:
                    
                    # Display results as HTML
                    st.markdown(
                        '<h1>' + '\n'.join(results) + '</h1>', 
                        unsafe_allow_html=True
                    )

                else:
                    st.error("No results found.")
            else:
                st.error("Please enter a query of a few keywords to search!")

    # Footer
    footer = """
    <style>
    .footer {
        position: relative;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: transparent;
        color: gray;
        text-align: center;
        padding: 10px;
        font-size: 16px;
    }
    .streamlit-container {
        margin-bottom: 10px;  /* Adjust this value based on your footer height */
    }
    </style>
    <div class="footer">
    Developed with 💚 by <a href="https://github.com/badrex" target="_blank">Badr Alabsi</a>
    </div>
    """

    # Use columns to visually separate the footer from the form content
    footer_column = st.columns(1)  # Creates a full-width column
    with footer_column[0]:
        st.markdown(footer, unsafe_allow_html=True)
 




if __name__ == "__main__":
    main()

