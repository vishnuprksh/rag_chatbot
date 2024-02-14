import torch
from transformers import RAGTokenizer, RAGSequenceForGeneration, DPRQuestionEncoder, DPRContextEncoder

# Sample knowledge base (replace with your data)
documents = [
    "London is the capital of England.",
    "Machine learning is a field of AI.",
    "Python is a popular programming language."
]

# Load pre-trained models
tokenizer = RAGTokenizer.from_pretrained("facebook/rag-sequence-nq")
retriever = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base") 
generator = RAGSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq")

# Create the index 
faiss_index = faiss.IndexFlatL2(retriever.config.hidden_size)  # Use FAISS index

# Index the documents
input_dict = tokenizer(documents, return_tensors="pt", padding=True)  
input_ids = input_dict["input_ids"]
with retriever.no_grad():
    passage_embeddings = retriever.embed_contexts(input_ids)
faiss_index.add(passage_embeddings.numpy())

# RAG interaction example
question = "What is the capital of the UK?"
input_dict = tokenizer.prepare_seq2seq_batch(src_texts=[question], return_tensors="pt")

with generator.no_grad():
   retrieved_doc_ids = retriever(input_dict["input_ids"], faiss_index) 
   generated = generator.generate(
       context_input_ids=retrieved_doc_ids[0],  # Retrieved indices
       input_ids=input_dict["input_ids"]
   )
print(tokenizer.decode(generated[0], skip_special_tokens=True))  # Decode output
