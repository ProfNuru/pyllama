import ollama

# Vector database
import chromadb

import psycopg
from psycopg.rows import dict_row

import ast

# CLI coloring package
from colorama import Fore

# For loading bar
from tqdm import tqdm


# Ollama model
MODEL='llama3.1'

# Embedding model used to generate embeddings
EMBEDDING_MODEL='nomic-embed-text'

# Postgres database connection parameters.
DB_PARAMS = {
    'dbname':'memory_agent',
    'user':'[DB-USER]',
    'password':'[YOUR_PASSWORD]',
    'host':'localhost',
    'port':'5432',
}

# Database table for storing the conversations
CONVERSATIONS_TABLE='conversations'


# A system messag eto instruct the AI on how to properly use the attached context
system_prompt = (
    'You are an AI assistant that has memory of every conversation you have ever had with this user. '
    'On every prompt from the user, the system has checked for any relevant messages you have had with the user. '
    'If any embedded previous conversations are attached, use them for context to responding to the user, '
    'if the context is relevant and useful to responding. If the recalled conversations are irrelevant, '
    'disregard speaking about them and respond normally as an AI assistant. Do not talk about recalling conversations. '
    'Just use any useful data from the previous conversations and respond normally as an intelligent AI assistant.'
)

# Start vector database client by instantiating chromadb.Client()
client = chromadb.Client()

# List of previous conversations formatted for the Ollama API
convo = [{'role':'system', 'content':system_prompt}]

# Connect to database
def connect_db():
    conn = psycopg.connect(**DB_PARAMS)
    return conn

# Fetch all conversations from database
def fetch_conversations():
    conn = connect_db()
    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute('SELECT * FROM conversations')
        conversations = cursor.fetchall()
        # print(conversations)
    conn.close()
    return conversations


# Store prompt and corresponding respons eto database
def store_conversations(prompt, response):
    conn = connect_db()
    with conn.cursor() as cursor:
        cursor.execute(
            'INSERT INTO conversations (timestamp, prompt, response) VALUES (CURRENT_TIMESTAMP, %s, %s)',
            (prompt, response)
        )
        conn.commit()
    conn.close()

# Delete last conversation form the database
def remove_last_conversation():
    conn = connect_db()
    with conn.cursor() as cursor:
        cursor.execute('DELETE FROM conversations WHERE id = (SELECT MAX(id) FROM conversations)')
        cursor.commit()
    conn.close()

# Use Ollama's streaming response capability to stream responses and not wait till the repsonse is complete
def stream_response(prompt):
    response = ''

    # Get response stream by passing model and previous conversations. Set stream to true
    stream = ollama.chat(model=MODEL,messages=convo,stream=True)
    
    # Print response header with font color light green
    print(Fore.LIGHTGREEN_EX + '\nASSISTANT:')

    # Iterate through chunks of the stream and print them out
    for chunk in stream:
        content = chunk['message']['content']
        response+=content

        # Do not add new line after printing each chunk of the stream
        print(content, end='', flush=True)

    print('\n')

    # Store prompt and corresponding response in database after stream is complete
    store_conversations(prompt=prompt,response=response)

    # Append response to convo
    convo.append({'role':'assistant','content':response})


# Create vector database, generate embeddings from conversations and store in conversations table
def create_vector_db(conversations):
    vector_db_name = CONVERSATIONS_TABLE

    # If table already exists, delete it.
    try:
        client.delete_collection(name=vector_db_name)
    except ValueError:
        pass

    # Create the table
    vector_db = client.create_collection(name=vector_db_name)
    
    # Iterate through the conversations and generate embeddings from each serialized conversation
    for c in conversations:
        serialized_convo = f'prompt: {c['prompt']} response: {c['response']}'
        response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=serialized_convo)
        embedding = response['embedding']

        # Add embedding to table
        vector_db.add(
            ids=[str(c['id'])],
            embeddings=[embedding],
            documents=[serialized_convo]
        )

# Retrive 2 most relevant embeddings per query in the generated queries list.
def retrieve_embeddings(queries, results_per_query=2):
    embeddings = set()
    
    # Iterate through queries list and retrieve best embedding for each query
    # Use tqdm to create loading bar
    for query in tqdm(queries, desc='Processing queries to vector database'):
        response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=query)
        query_embedding = response['embedding']

        vector_db = client.get_collection(name=CONVERSATIONS_TABLE)
        results = vector_db.query(query_embeddings=[query_embedding], n_results=results_per_query)
        best_embeddings = results['documents'][0]

        # Iterate over best embeddings
        for best in best_embeddings:
            if best not in embeddings:
                # Classify each embedding and retrieve the relevant embeddings
                # That is, where the response is 'yes' for whether or not the conext is directly related to the query
                if 'yes' in classify_embedding(query=query, context=best):
                    # Add embedding to embeddings set
                    embeddings.add(best)

    # Return embeddings set
    return embeddings


# Create queries. To be called when prompt does not match context.
def create_queries(prompt):
    # Instruct the agent to generate a Python list of queries relevant to the prompt
    query_msg = (
        'You are a first principle reasoning search query AI agent.'
        'Your list of search queries will be ran on an embedding database of all your conversations'
        'you have ever had with the user. With first principles, create a Python list of queries to '
        'search the embeddings database for any data that would be necessary to have access to in '
        'order to correctly respond to the prompt. Your response must be a Python list with no syntax errors. '
        'Do not explain anything and do not ever generate anything but a perfect syntax Python list'
    )

    # Primer convo for multi-short learning in order for th elanguage to respond in a specific format
    # Also formatted for the Ollama API
    query_convo = [
        {'role':'system', 'content':query_msg},
        {'role':'user', 'content':'Write an email to my car insurance company and create a persuasive request for them to lower my monthly rate.'},
        {'role':'assistant', 'content':'["What is the users name?", "What is the users current auto insurance provider?", "What is the monthly rate the user currently pays for auto insurance?"]'},
        {'role':'user', 'content':'how can i convert the speak function in my llama3 python voice assistant to use pyttsx3?'},
        {'role':'assistant', 'content':'["Llama3 voice assistant", "Python voice assistant", "OpenAI TTS", "openai speak"]'},
        
        # Add the prompt convo as the last conversation
        {'role':'user', 'content':prompt}
    ]
    # This will ensure the response from the assistant role will look similar to the responses passed. That is a Python list of queries.

    # Make query and get response
    response = ollama.chat(model=MODEL, messages=query_convo)

    # Print response from each query generated
    print(Fore.YELLOW + f'\nVector database queries: {response['message']['content']} \n')

    try:
        # Use the 'ast' library to convert the string response into Python list
        # This is why the response from Ollama needs to be a Python list. We achieved this
        # through multi-short learning above
        return ast.literal_eval(response['message']['content'])
    except:
        # If an exception is thrown, return prompt in a list
        return [prompt]
    
# Compare generated query and retrieved context and return 'yes' if context is directly related to the query or 'no' if it is not
def classify_embedding(query, context):
    # Used multi-short learning here to limit responses to 'yes' or 'no'

    # Prompt instructions
    classify_msg = (
        'You are an embedding classification AI agent. Your input will be a prompt and one embedded chunk of text. '
        'You will not respond as an AI assistant. You only respond "yes" or "no". '
        'Determine whether the context contains data that directly is related to the search query. '
        'If the context is seemingly exactly what the search query needs, respond "yes". If it is anything but directly '
        'related, respond "no". Do not respond "yes" unless the content is highly relevant to the search query.'
    )

    # Primer convo for multi-short learning in order for th elanguage to respond in a specific format
    # Also formatted for the Ollama API
    classify_convo = [
        {'role':'system', 'content':classify_msg},
        {'role':'user', 'content':f'SEARCH QUERY: What is the users name? \n\nEMBEDDED CONTEXT: You are Nurudeen. How can I help today?'},
        {'role':'assistant', 'content':'yes'},
        {'role':'user', 'content':f'SEARCH QUERY: Llama3 Python Voice Assistant \n\nEMBEDDED CONTEXT: Siri is a voice assistant on Apple iOS and Mac OS.'},
        {'role':'assistant', 'content':'no'},

        # Add query and context at the end of the multi-short primer
        {'role':'user', 'content':f'SEARCH QUERY: {query} \n\nEMBEDDED CONTEXT: {context}'},
    ]

    response = ollama.chat(model=MODEL, messages=classify_convo)
    return response['message']['content'].strip().lower()

# Uses prompt to create queries and retrive embeddings to add relevant context to prompt
def recall(prompt):
    queries = create_queries(prompt=prompt)
    embeddings = retrieve_embeddings(queries=queries)

    # Append prompt and embeddings to the convo list, formatted for the Ollama API
    convo.append({'role':'user','content':f'MEMORIES: {embeddings} \n\n USER PROMPT: {prompt}'})

    # Print number of embeddings added for context
    print(f'\n{len(embeddings)} message:response embeddings added for context.')


# Fetch conversations
conversations = fetch_conversations()

# Generate conversations' embeddings and store in in-memory vector database
create_vector_db(conversations=conversations)

while True:
    prompt = input(Fore.WHITE + 'USER: \n')

    # Check if this is a recall prompt
    if prompt[:7].lower() == '/recall':
        # Slice off the recall directive from the prompt
        prompt = prompt[8:]

        # Recall and stream the response
        recall(prompt=prompt)
        stream_response(prompt=prompt)
    
    # Check if this is a forget prompt
    elif prompt[:7] == '/forget':
        # Delete last conversation from database
        remove_last_conversation()
        convo = convo[:-2]
        print('\n')

    # Check if this is a memorize prompt
    elif prompt[:9] == '/memorize':
        prompt = prompt[10:]
        # Store conversations without generating a response
        store_conversations(prompt=prompt, response='Memory stored.')
        print('\n')
    else:
        # Else generate response stream normally
        convo.append({'role':'user','content':prompt})
        stream_response(prompt=prompt)
    
