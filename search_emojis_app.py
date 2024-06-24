import streamlit as st
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from qdrant_client import models, QdrantClient
import emoji as em
import warnings

warnings.filterwarnings('ignore')

# Define a function to load resources and check session state
def load_resources():
    if ('vector_DB_client' not in st.session_state 
            or 'sentence_encoder' not in st.session_state):
        
        # Load emoji dictionary
        with open('emoji_embeddings_dict.pkl', 'rb') as file:
            emoji_dict = pickle.load(file)

        # Load sentence encoder
        embedding_model = 'paraphrase-multilingual-MiniLM-L12-v2'
        sentence_encoder = SentenceTransformer(embedding_model)
        st.session_state.sentence_encoder = sentence_encoder

        # Setup the Qdrant client and populate the database
        vector_DB_client = QdrantClient(":memory:")
        embedding_dict = {
            emoji: np.array(data['embedding']) 
            for emoji, data in emoji_dict.items()
        }

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
        
        st.session_state.vector_DB_client = vector_DB_client

def return_similar_emojis(query: str):
    query_vector = st.session_state.sentence_encoder.encode(query).tolist()

    hits = st.session_state.vector_DB_client.search(
        collection_name="EMOJIS",
        query_vector=query_vector,
        limit=400,
    )

    search_emojis = []

    # only add to list if it is not already an item in the list
    for hit in hits:
        if hit.payload['Emoji'] not in search_emojis:
            search_emojis.append(hit.payload['Emoji'])

    return search_emojis

def main():
    load_resources()
    st.title("Emojivan 🪔")
    #st.text("AI-powereed semantic search for emojis 🫣 ")
    #query = st.text_input("Enter your search query", "")


    # Using columns to layout the input and button next to each other
    with st.form(key="search_form", border=False):

        col1, col2 = st.columns([3, 1])
        with col1:
            query = st.text_input(
                label="Enter your search query", 
                key="query", 
                label_visibility='hidden', 
                help="exploration discovery"
            ) #Enter your search query

        with col2:
            trigger_search = st.form_submit_button(label="Go! 🚀")

        st.markdown(
        """
        <style>
        button {
            height: auto;cell
            margin-top:49px !important;

        }
        </style>
        """,
            unsafe_allow_html=True,
        )


        if trigger_search:

        #if trigger_search:# st.button("Search"):
            if query:
                results = return_similar_emojis(query)
                if results:
                    # Display results as HTML
                    st.markdown(
                        '<h1>' + '  '.join(results) + '</h1>', 
                        unsafe_allow_html=True
                    )
                else:
                    st.error("No results found.")
            else:
                st.error("Please enter a query to search.")

if __name__ == "__main__":
    main()


# def main():
#     load_resources()
#     st.title("Emojivan 🪔")

#     # Function to handle search and update results
#     def handle_search():
#         if query:
#             results = return_similar_emojis(query)
#             results_container.empty()  # Clear the results container before displaying new results
#             with results_container.container():
#                 if results:
#                     # Display results as HTML
#                     st.markdown('<h1>' + ' '.join(results) + '</h1>', unsafe_allow_html=True)
#                 else:
#                     st.error("No results found.")
#         else:
#             results_container.empty()
#             st.error("Please enter a query to search.")

#     # Text input for query with action on hitting Enter
#     query = st.text_input("Enter your search query", key="query") #, on_change=handle_search)

#     # Create a container for results
#     results_container = st.empty()

# if __name__ == "__main__":
#     main()


