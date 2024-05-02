import asyncio
import aiohttp
from langchain_community.embeddings.ollama import OllamaEmbeddings

class AsyncOllamaEmbedder:
    def __init__(self, model='avr/sfr-embedding-mistral:q8_0', base_url='http://localhost:11434'):
        self.sync_embeddings = OllamaEmbeddings(model=model)
        self.base_url = f"{base_url}/api/embeddings"
        self.session = None

    async def init_session(self):
        if self.session is None or self.session.closed:
            self.session = await aiohttp.ClientSession().__aenter__()

    async def close_session(self):
        if self.session and not self.session.closed:
            await self.session.__aexit__(None, None, None)

    def embed_documents(self, texts):
        # Synchronous wrapper for asynchronous embedding
        return self.sync_call(self.async_embed_documents, texts)

    def embed_query(self, query):
        # Single query synchronous wrapper for asynchronous embedding
        return self.sync_call(self.async_embed_documents, [query])[0]

    async def async_embed_documents(self, texts):
        # Initialize session right before use
        await self.init_session()
        tasks = [self.send_embedding_request(text) for text in texts]
        results = await asyncio.gather(*tasks)
        await self.close_session()
        return results

    async def send_embedding_request(self, text):
        await self.init_session()  # Ensure session is available
        async with self.session.post(self.base_url, json={'model': self.sync_embeddings.model, 'prompt': text}) as response:
            if response.status == 200:
                data = await response.json()
                return data.get('embedding')
            else:
                return None  # Handle errors as needed

    def sync_call(self, async_func, *args):
        loop = asyncio.get_event_loop()
        if loop.is_running():
            new_loop = asyncio.new_event_loop()
            result = new_loop.run_until_complete(async_func(*args))
            new_loop.close()
            return result
        else:
            return loop.run_until_complete(async_func(*args))

def get_embedding_function():
    return AsyncOllamaEmbedder()
