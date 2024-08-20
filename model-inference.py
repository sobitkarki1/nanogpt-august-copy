from transformers import GPT2LMHeadModel, GPT2Tokenizer



def generate_text(model_dir, prompt_text, max_length=50, temperature=1.0, top_p=0.9, repetition_penalty=1.2):
    tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    
    inputs = tokenizer.encode(prompt_text, return_tensors='pt')

    outputs = model.generate(
        inputs,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        num_return_sequences=1
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text

# Generate text
print(generate_text('output_directory', " मेरो", max_length=1000))
