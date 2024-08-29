# Ollama RAG with Memory
A local AI agent to store and recall any conversations
- Start your prompt with `/recall` to recall previous conversations
- Start your prompt with `/forget` to forget the last conversation
- Start your prompt with `/memorize` to pass information to be memorized
- Chat normally with the AI agent

- Install and run Postgres, and set DB_PARAMS to your database connection parameters inside 'assistant.py'

- Create memory-agent database in Postgres
```sql
    CREATE DATABASE memory_agent;
```

- Create conversations table
```sql
    CREATE TABLE conversations (
        id SERIAL PRIMARY KEY,
        timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        prompt TEXT NOT NULL,
        response TEXT NOT NULL,
    );
```

- Insert first conversation
```sql
    INSERT INTO conversations (timestamp, prompt, response) VALUES (CURRENT_TIMESTAMP, 'what is my name?', 'Your name is Nurudeen. Known online as Turnables.');
```

- Install Ollama
    - Pull models:
        - ollama run llama3:8b
        - ollama run nomic-embed-text

- Create and activate virtual environment
- Install dependencies from requirements.txt

- Run `python assistant.py`


# How I improved accuracy of the agent
- I used the system prompts to set the extent of the agent's knowledge base
(Check the 'system_prompt' variable in assistant.py)

- For complex queries, I set up the agent to create queries and query itself multiple times
    - I used multi-short learning technique to prime the structure of the generated responses and then executing the queries dynamically
    - Fine tuning the model will work better than this though.
