"""Setup check; this is a utility, not the student pipeline solution."""
import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv

starter_dir = Path(__file__).resolve().parents[1]
load_dotenv(starter_dir / '.env')
chroma_path = Path(os.getenv('CHROMA_PATH', './data/chroma')).expanduser()
if not chroma_path.is_absolute():
    chroma_path = starter_dir / chroma_path

client = chromadb.PersistentClient(path=str(chroma_path))
client.heartbeat()
print(f'ChromaDB: {chromadb.__version__} | Persistent path: {chroma_path}')
print('Setup OK. Pipeline functions still need to be implemented.')
