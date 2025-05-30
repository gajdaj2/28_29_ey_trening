from chroma_lokalnie import chunk_text

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

# Re-ranking imports
try:
    from sentence_transformers import CrossEncoder

    RERANKING_AVAILABLE = True
    print("Re-ranking dostępny (CrossEncoder)")
except ImportError:
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch

        RERANKING_AVAILABLE = True
        print("Re-ranking dostępny (Transformers)")
    except ImportError:
        RERANKING_AVAILABLE = False
        print("Re-ranking niedostępny - zainstaluj: pip install sentence-transformers")


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


class ReRanker:
    """Klasa do re-rankingu rezultatów wyszukiwania"""

    def __init__(self, model_name=None):
        self.model = None
        self.tokenizer = None
        self.cross_encoder = None

        if not RERANKING_AVAILABLE:
            print("Re-ranking niedostępny - będę używać tylko similarity score")
            return

        if model_name is None:
            # Domyślne modele dla różnych języków
            model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Angielski
            # Dla polskiego można użyć:
            # model_name = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"  # Multilingual

        try:
            # Próbujemy CrossEncoder (sentence-transformers)
            self.cross_encoder = CrossEncoder(model_name)
            print(f"Załadowano CrossEncoder: {model_name}")
        except:
            try:
                # Fallback na zwykłe transformers
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
                print(f"Załadowano model transformers: {model_name}")
            except Exception as e:
                print(f"Błąd ładowania modelu re-ranking: {e}")
                self.model = None

    def rerank(self, query, documents, scores=None, top_k=5):
        """
        Re-rankuje dokumenty względem zapytania

        Args:
            query: zapytanie użytkownika
            documents: lista dokumentów do przereankowania
            scores: oryginalne similarity scores (opcjonalne)
            top_k: ile najlepszych dokumentów zwrócić

        Returns:
            Lista tupli (dokument, nowy_score, oryginalny_idx)
        """
        if not RERANKING_AVAILABLE or (not self.cross_encoder and not self.model):
            # Fallback - zwracamy oryginalne wyniki
            if scores:
                results = [(doc, score, i) for i, (doc, score) in enumerate(zip(documents, scores))]
                return sorted(results, key=lambda x: x[1], reverse=True)[:top_k]
            else:
                return [(doc, 0.0, i) for i, doc in enumerate(documents[:top_k])]

        try:
            if self.cross_encoder:
                # Używamy CrossEncoder
                pairs = [[query, doc] for doc in documents]
                rerank_scores = self.cross_encoder.predict(pairs)

                # Łączymy z oryginalnymi danymi
                results = [(documents[i], float(rerank_scores[i]), i)
                           for i in range(len(documents))]

            else:
                # Używamy zwykłych transformers
                rerank_scores = []
                for doc in documents:
                    inputs = self.tokenizer(query, doc,
                                            return_tensors="pt",
                                            truncation=True,
                                            max_length=512)

                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        # Bierzemy prawdopodobieństwo pozytywnej klasy
                        score = torch.softmax(outputs.logits, dim=1)[0][1].item()
                        rerank_scores.append(score)

                results = [(documents[i], rerank_scores[i], i)
                           for i in range(len(documents))]

            # Sortujemy po nowym score i zwracamy top_k
            results = sorted(results, key=lambda x: x[1], reverse=True)
            return results[:top_k]

        except Exception as e:
            print(f"Błąd podczas re-rankingu: {e}")
            # Fallback na oryginalne wyniki
            if scores:
                results = [(doc, score, i) for i, (doc, score) in enumerate(zip(documents, scores))]
                return sorted(results, key=lambda x: x[1], reverse=True)[:top_k]
            else:
                return [(doc, 0.0, i) for i, doc in enumerate(documents[:top_k])]


def search_with_reranking(collection, query, initial_results=20, final_results=5, reranker=None):
    """
    Wyszukuje dokumenty z opcjonalnym re-rankingiem

    Args:
        collection: ChromaDB collection
        query: zapytanie użytkownika
        initial_results: ile dokumentów pobrać w pierwszym kroku
        final_results: ile dokumentów zwrócić po re-rankingu
        reranker: obiekt ReRanker (opcjonalny)

    Returns:
        Słownik z wynikami podobny do collection.query()
    """
    # Pierwsza faza - pobieramy więcej wyników niż potrzebujemy
    results = collection.query(
        query_texts=[query],
        n_results=initial_results
    )

    documents = results['documents'][0]
    distances = results['distances'][0]
    ids = results['ids'][0] if 'ids' in results else [f"doc_{i}" for i in range(len(documents))]
    metadatas = results['metadatas'][0] if 'metadatas' in results else [{}] * len(documents)

    if not documents:
        return results

    # Konwertujemy distance na similarity score (1 - distance)
    similarity_scores = [1 - dist for dist in distances]

    if reranker:
        print(f"Re-ranking {len(documents)} dokumentów...")
        # Druga faza - re-ranking
        reranked = reranker.rerank(
            query=query,
            documents=documents,
            scores=similarity_scores,
            top_k=final_results
        )

        # Przygotowujemy wyniki w formacie ChromaDB
        final_documents = [item[0] for item in reranked]
        final_scores = [item[1] for item in reranked]
        final_distances = [1 - score for score in final_scores]  # Konwersja z powrotem na distance
        original_indices = [item[2] for item in reranked]

        final_ids = [ids[idx] for idx in original_indices]
        final_metadatas = [metadatas[idx] for idx in original_indices]

        return {
            'documents': [final_documents],
            'distances': [final_distances],
            'ids': [final_ids],
            'metadatas': [final_metadatas],
            'reranked': True,
            'rerank_scores': final_scores
        }
    else:
        # Bez re-rankingu - zwracamy tylko top wyniki
        final_idx = min(final_results, len(documents))
        return {
            'documents': [documents[:final_idx]],
            'distances': [distances[:final_idx]],
            'ids': [ids[:final_idx]],
            'metadatas': [metadatas[:final_idx]],
            'reranked': False
        }
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

    # Inicjalizacja re-rankera
    print("\n=== INICJALIZACJA RE-RANKERA ===")
    reranker = ReRanker()
    # Możesz też wybrać konkretny model:
    # reranker = ReRanker("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")  # Multilingual

    # Test wyszukiwania
    print("\n=== TEST WYSZUKIWANIA Z RE-RANKINGIEM ===")
    test_queries = [
        "Jaka jest stolica Francji?",
        "Ile mieszkańców ma Francja?",
        "Jakie są główne miasta Francji?",
        "Historia Francji",
        "Gospodarka francuska",
        "Kultura i tradycje Francji"
    ]

    for query in test_queries:
        print(f"\n{'=' * 50}")
        print(f"ZAPYTANIE: {query}")
        print('=' * 50)

        # Test bez re-rankingu
        print("\n--- BEZ RE-RANKINGU ---")
        results_basic = collection.query(
            query_texts=[query],
            n_results=3
        )

        for i, (doc, distance) in enumerate(zip(results_basic['documents'][0], results_basic['distances'][0])):
            similarity = 1 - distance
            print(f"  {i + 1}. (similarity: {similarity:.3f}) {doc[:150]}...")

        # Test z re-rankingiem
        print("\n--- Z RE-RANKINGIEM ---")
        results_reranked = search_with_reranking(
            collection=collection,
            query=query,
            initial_results=10,  # Pobieramy 10 kandydatów
            final_results=3,  # Zwracamy 3 najlepsze po re-rankingu
            reranker=reranker
        )

        if results_reranked.get('reranked', False):
            # Mamy wyniki po re-rankingu
            rerank_scores = results_reranked.get('rerank_scores', [])
            for i, (doc, rerank_score) in enumerate(zip(results_reranked['documents'][0], rerank_scores)):
                print(f"  {i + 1}. (rerank: {rerank_score:.3f}) {doc[:150]}...")
        else:
            # Re-ranking nie był dostępny
            for i, (doc, distance) in enumerate(
                    zip(results_reranked['documents'][0], results_reranked['distances'][0])):
                similarity = 1 - distance
                print(f"  {i + 1}. (similarity: {similarity:.3f}) {doc[:150]}...")

    # Statystyki porównawcze
    print(f"\n{'=' * 50}")
    print("STATYSTYKI PORÓWNAWCZE")
    print('=' * 50)

    if RERANKING_AVAILABLE and reranker.cross_encoder:
        print("✅ Re-ranking aktywny (CrossEncoder)")
    elif RERANKING_AVAILABLE and reranker.model:
        print("✅ Re-ranking aktywny (Transformers)")
    else:
        print("❌ Re-ranking nieaktywny")
        print("   Zainstaluj: pip install sentence-transformers")

    print(f"📊 Dokumentów w bazie: {len(documents)}")
    print("🔍 Strategia wyszukiwania: Embedding similarity + Re-ranking")
    print("📈 Re-ranking może znacznie poprawić relevance wyników!")


if __name__ == "__main__":
    main()
