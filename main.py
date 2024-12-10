import json

from transformers import AutoModel, AutoTokenizer
import torch
import psycopg2
import os
import glob
from torch.cuda.amp import autocast
import gc
import json

access_token = "HUGGING_FACE_TOKEN_HERE"

connection = psycopg2.connect(
)

bookmarks_connection = psycopg2.connect(
)

cursor = connection.cursor()
bookmarks_cursor = bookmarks_connection.cursor()

insert_query = 'INSERT INTO test_vectors (my_vector, text, starting_text) VALUES (%s, %s, %s);'
update_query = 'UPDATE test_vectors SET starting_text = %s WHERE text = %s;'
similar_embeddings_query = 'SELECT text, my_vector <-> %s AS distance, starting_text FROM test_vectors ORDER BY distance ASC;'

# Check if a GPU is available and set PyTorch to use it
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# Function to check if processing can be done on GPU
def can_process_on_gpu(tokenizer, text, device):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    try:
        # Just a dry run to check if inputs can be processed on GPU
        inputs = inputs.to(device)
        return True
    except RuntimeError as e:
        return False
    finally:
        # Cleanup to free memory
        del inputs
        torch.cuda.empty_cache()

def read_markdown_files(directory):
    # This list will store the dictionaries with filename and text
    all_files_data = []

    # Walk through the directory and its subdirectories
    for root, dirs, files in os.walk(directory):
        # Find all markdown files in current directory
        for file in glob.glob(os.path.join(root, '*.md')):
            # Open and read the markdown file
            with open(file, 'r', encoding='utf-8') as md_file:
                try:
                    text = md_file.read()
                    # Create a dictionary with filename and text, then append to the list
                    file_data = {'filename': file, 'text': text}
                    all_files_data.append(file_data)
                except Exception as e:
                    print(f'Error reading file {file}: {e}')

    return all_files_data


# Function to dynamically process text
# Returns a 2d array of all the embeddings found, broken up into chunks of 512
def process_text(text, tokenizer, model, device):
    if text is None or len(text) == 0:
        print("No text to process")
        return []
    # Estimate the average token size (including spaces/punctuation)
    estimated_token_size = 5  # This is an estimate; adjust based on your data
    max_chunk_char_length = (512 - tokenizer.num_special_tokens_to_add()) * estimated_token_size

    # Split the text into chunks based on estimated token size
    text_chunks = [text[i:i+max_chunk_char_length] for i in range(0, len(text), max_chunk_char_length)]

    embeddings_list = []
    for chunk_text in text_chunks:
        if can_process_on_gpu(tokenizer, text, device):
            print("Processing on GPU")
            inputs = tokenizer(chunk_text, return_tensors="pt", max_length=512, truncation=True).to(device)
        else:
            print("Fallback to CPU")
            inputs = tokenizer(chunk_text, return_tensors="pt", max_length=512, truncation=True)
            model.to("cpu")

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                outputs = model(**inputs)
                chunk_embeddings = outputs.last_hidden_state.mean(dim=1).squeeze(0).tolist()
                embeddings_list.append(chunk_embeddings)

    # Cleanup
    del inputs, outputs
    torch.cuda.empty_cache()
    gc.collect()

    return embeddings_list

def create_embeddings_for_markdown_folder():
    # Specify the directory to search in
    directory = os.path.expanduser('~/Documents/Notes')

    # Call the function and store the result
    text_maps = read_markdown_files(directory)

    for text_map in text_maps:
        print(f"Processing File - {text_map['filename']}")
        filename = text_map['filename']
        # todo, summarize the text first
        embeddings_list = process_text(text_map['text'], tokenizer, model, device)

        for i, embeddings in enumerate(embeddings_list):
            cursor.execute(insert_query, (embeddings, f'{filename}-{str(i)}', embeddings[:100],))
        connection.commit()



def create_embeddings_for_bookmarks():
    # Specify the directory to search in
    QUERY = 'SELECT url, text FROM url;'
    bookmarks_cursor.execute(QUERY)
    rows = bookmarks_cursor.fetchall()

    for row in rows:
        print(f"Processing Bookmark - ID: {row[0]}")
        embeddings_list = process_text(row[1], tokenizer, model, device)
        for i, embeddings in enumerate(embeddings_list):
            cursor.execute(insert_query, (embeddings, f'{row[0]}-{str(i)}', embeddings[:100],))
        connection.commit()


def find_similar_embeddings():
    text = ''
    while text != 'exit':
        text = input('Enter your query: ')

        text_embeddings = process_text(text, tokenizer, model, device)
        text_embedding = json.dumps( text_embeddings[0])

        cursor.execute(similar_embeddings_query, (text_embedding,))
        rows = cursor.fetchall()
        index = 0
        for row in rows[:100]:
            index += 1
            starting_text_inline = row[2].replace('\n', '\\n')
            print(f"{str(index)} - Text: {row[0]} - Distance: {row[1]}- Starting Text: {starting_text_inline}")


# markdown_texts now contains all the text from markdown files in the directory and its subdirectories


# For LLaMA, replace "model_name" with the specific LLaMA model you want to use, if available.
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
model = AutoModel.from_pretrained(model_name, token=access_token)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.half()
model.to(device)

# create_embeddings_for_bookmarks()
# create_embeddings_for_markdown_folder()
find_similar_embeddings()

# Close the cursor and connection
cursor.close()
connection.close()
