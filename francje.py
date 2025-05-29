import streamlit as st
import requests
import wikipedia
import json

# Funkcja pobierająca artykuł z Wikipedii
def get_wikipedia_content(topic):
    try:
        page = wikipedia.page(topic, auto_suggest=False)
        return page.content
    except Exception as e:
        return f"Nie udało się pobrać artykułu: {e}"

# Funkcja wywołująca model Ollama
def query_ollama(prompt, model="llama3"):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(url, json=payload)
    if response.ok:
        return response.json()["response"]
    else:
        return "Błąd połączenia z Ollama."

# Streamlit UI
st.title("Ekstrakcja informacji z Wikipedii przy użyciu Ollama (z wynikiem JSON)")

country = st.text_input("Podaj nazwę kraju (lub miasta):", "Polska")
model_name = st.text_input("Nazwa modelu Ollama:", "llama3")

if st.button("Pobierz i przeanalizuj"):
    with st.spinner("Pobieranie artykułu z Wikipedii..."):
        content = get_wikipedia_content(country)

    if content.startswith("Nie udało się"):
        st.error(content)
    else:
        st.subheader("Fragment artykułu z Wikipedii (pierwsze 1500 znaków):")
        st.write(content[:1500] + "...")

        prompt = f"""
Oto fragment artykułu z Wikipedii o kraju lub mieście:

\"\"\"{content}\"\"\"

Wyodrębnij następujące informacje (jeśli są dostępne):
1. Powierzchnia (w km²)
2. Prezydent (lub inny głowa państwa)
3. Krótkie podsumowanie (max 2-3 zdania) na temat kraju/miasta.

Zwróć odpowiedź w formacie JSON, np.:
{{
  "powierzchnia": "...",
  "prezydent": "...",
  "podsumowanie": "..."
}}
"""

        with st.spinner("Analiza przez model Ollama..."):
            result = query_ollama(prompt, model=model_name)

        st.subheader("Wynik od Ollama (tekst):")
        st.code(result, language="json")

        # Próba parsowania odpowiedzi jako JSON
        try:
            result_json = json.loads("\n".join(result[1:-1]))
            st.write(result_json)
            st.subheader("Wynik w formie JSON (podgląd):")
            st.json(result_json)
        except Exception as e:
            st.warning("Nie udało się sparsować odpowiedzi jako JSON. Upewnij się, że model odpowiedział w poprawnym formacie.")
