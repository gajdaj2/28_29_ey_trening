from vectordb import Memory

from cw_9_dane import text2, text

memory = Memory(chunking_strategy={'mode':'sliding_window', 'window_size': 128, 'overlap': 16})

metadata = {"title": "Introduction to Machine Learning", "url": "https://example.com/introduction-to-machine-learning"}

memory.save(text, metadata)

metadata2 = {"title": "Introduction to Artificial Intelligence", "url": "https://example.com/introduction-to-artificial-intelligence"}

memory.save(text2, metadata2)

query = "What is the relationship between AI and machine learning?"

results = memory.search(query, top_n=3)

print("Answer: "+str(results[0]['chunk']))
print("URL:"+str(results[0]['metadata']['url']))
