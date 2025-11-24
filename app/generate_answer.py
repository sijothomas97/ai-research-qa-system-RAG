# File: generate_answer.py
from transformers import T5Tokenizer, T5ForConditionalGeneration
from retrieve import retrieve_relevant_chunks

def generate_answer(query, model_name="t5-small"):
    """
    Generate an answer to the query using retrieved chunks and T5.
    Args:
        query (str): User query.
        model_name (str): Hugging Face T5 model name.
    Returns:
        str: Generated answer.
    """
    # Retrieve relevant chunks
    chunks = retrieve_relevant_chunks(query)
    if not chunks:
        return "No relevant information found."
    
    # Combine chunks into context
    context = " ".join([chunk['text'] for chunk in chunks])
    
    # Initialize T5 model and tokenizer
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    # Prepare input for T5
    input_text = f"question: {query} context: {context}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True) # 512 is the common default for T5 models
    
    # Generate answer
    outputs = model.generate(
        inputs['input_ids'],
        max_length=150,
        num_beams=5,
        early_stopping=True
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return answer

# Test answer generation
if __name__ == "__main__":
    query = "What are the latest advancements in neural networks?"
    answer = generate_answer(query)
    print(f"Query: {query}")
    print(f"Answer: {answer}")