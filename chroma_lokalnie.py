__import__('pysqlite3')
import sys

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import re

load_dotenv()


def scrape_wikipedia_article(url):
    """Pobiera treść artykułu z Wikipedii"""
    try:
        # Dodajemy User-Agent żeby Wikipedia nas nie blokowała
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        response = requests.get(url, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # Znajdujemy główną zawartość artykułu
        content_div = soup.find('div', {'id': 'mw-content-text'})

        if not content_div:
            raise Exception("Nie znaleziono głównej treści artykułu")

        # Usuwamy niepotrzebne elementy
        for element in content_div.find_all(['table', 'div', 'span'], class_=['navbox', 'infobox', 'toc']):
            element.decompose()

        # Pobieramy wszystkie paragrafy
        paragraphs = content_div.find_all('p')

        # Czyścimy tekst z tagów HTML i nadmiarowych znaków
        cleaned_paragraphs = []
        for p in paragraphs:
            text = p.get_text()
            # Usuwamy nadmierne białe znaki i znaki specjalne
            text = re.sub(r'\s+', ' ', text).strip()
            # Pomijamy bardzo krótkie paragrafy
            if len(text) > 50:
                cleaned_paragraphs.append(text)

        return cleaned_paragraphs

    except Exception as e:
        print(f"Błąd podczas pobierania danych z Wikipedii: {e}")
        return []


def chunk_text(text, max_length=500):
    """Dzieli długi tekst na mniejsze chunki"""
    if len(text) <= max_length:
        return [text]

    # Próbujemy dzielić po zdaniach
    sentences = re.split(r'[.!?]+', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Jeśli dodanie zdania nie przekroczy limitu
        if len(current_chunk + sentence) <= max_length:
            current_chunk += sentence + ". "
        else:
            # Zapisujemy obecny chunk i zaczynamy nowy
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "

    # Dodajemy ostatni chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def main():
    # URL artykułu o Francji
    wikipedia_url = "https://pl.wikipedia.org/wiki/Francja"

    print("Pobieranie danych z Wikipedii...")
    paragraphs = scrape_wikipedia_article(wikipedia_url)

    if not paragraphs:
        print("Nie udało się pobrać danych z Wikipedii")
        return

    print(f"Pobrano {len(paragraphs)} paragrafów")

    # Dzielimy długie paragrafy na chunki
    all_chunks = []
    for paragraph in paragraphs:
        chunks = chunk_text(paragraph, max_length=500)
        all_chunks.extend(chunks)

    print(f"Utworzono {len(all_chunks)} chunków")

    # Inicjalizacja ChromaDB z lokalnym embeddingiem
    client = chromadb.Client()

    # OPCJA 1: Sentence Transformers (najlepsza dla polskiego tekstu)
    # Wymaga: pip install sentence-transformers
    try:
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            # Inne dobre modele dla polskiego:
            # "sentence-transformers/distiluse-base-multilingual-cased"
            # "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        )
        print("Używam Sentence Transformers (multilingual)")
    except Exception as e:
        print(f"Błąd z Sentence Transformers: {e}")

        # OPCJA 2: HuggingFace Transformers
        # Wymaga: pip install transformers torch
        try:
            embedding_fn = embedding_functions.HuggingFaceEmbeddingFunction(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                # Możesz też użyć polskich modeli:
                # model_name="allegro/herbert-base-cased"
                # model_name="clarin-pl/roberta-polish-kgr10"
            )
            print("Używam HuggingFace Transformers")
        except Exception as e:
            print(f"Błąd z HuggingFace: {e}")

            # OPCJA 3: Ollama (jeśli masz zainstalowane lokalnie)
            # Wymaga zainstalowanego Ollama i pobranego modelu
            try:
                embedding_fn = embedding_functions.OllamaEmbeddingFunction(
                    model_name="nomic-embed-text",  # lub inny model embedding
                    url="http://localhost:11434"  # domyślny URL Ollama
                )
                print("Używam Ollama embeddings")
            except Exception as e:
                print(f"Błąd z Ollama: {e}")

                # OPCJA 4: Fallback - domyślny embedding ChromaDB
                print("Używam domyślnego embedding ChromaDB")
                embedding_fn = embedding_functions.DefaultEmbeddingFunction()

    # Usuwamy kolekcję jeśli istnieje (dla świeżego startu)
    try:
        client.delete_collection("francja_wikipedia")
    except:
        pass

    collection = client.create_collection(
        name="francja_wikipedia",
        embedding_function=embedding_fn
    )

    # Przygotowujemy dane do dodania
    documents = all_chunks
    ids = [f"chunk_{i}" for i in range(len(all_chunks))]

    # Dodajemy metadane (opcjonalnie)
    metadatas = [{"source": "wikipedia_francja", "chunk_id": i} for i in range(len(all_chunks))]

    print("Dodawanie dokumentów do ChromaDB...")

    # Dodajemy dokumenty w batches (ChromaDB może mieć limity)
    batch_size = 50
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        batch_metas = metadatas[i:i + batch_size]

        collection.add(
            documents=batch_docs,
            ids=batch_ids,
            metadatas=batch_metas
        )
        print(f"Dodano batch {i // batch_size + 1}/{(len(documents) - 1) // batch_size + 1}")

    print(f"Pomyślnie dodano {len(documents)} dokumentów do bazy wektorowej!")

    # Test wyszukiwania
    print("\n=== TEST WYSZUKIWANIA ===")
    test_queries = [
        "Jaka jest stolica Francji?",
        "Ile mieszkańców ma Francja?",
        "Jakie są główne miasta Francji?",
        "Historia Francji"
    ]

    for query in test_queries:
        print(f"\nZapytanie: {query}")
        results = collection.query(
            query_texts=[query],
            n_results=2
        )

        for i, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0])):
            print(f"  {i + 1}. (podobieństwo: {1 - distance:.3f}) {doc[:200]}...")


if __name__ == "__main__":
    main()
