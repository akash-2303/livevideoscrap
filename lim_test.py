import pandas as pd
import logging
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import os
import argparse
from tqdm import tqdm
import torch

# Configure logging for the script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler()])

# Function to load and process CSV data
def load_and_process_csv(file_path, chunksize=10000):
    logging.info(f"Loading CSV data from {file_path} in chunks")
    chunks = []
    empty_processed_text_count = 0  # Counter for rows with empty 'processed_text'
    total_rows = 0  # Total row counter

    try:
        for chunk in pd.read_csv(file_path, encoding='utf-8', chunksize=chunksize):
            total_rows += len(chunk)
            # Count rows where 'processed_text' is empty or NaN
            empty_processed_text_count += chunk['processed_text'].isna().sum()
            empty_processed_text_count += chunk['processed_text'].str.strip().eq('').sum()

            # Filter the chunk to keep only rows with non-empty `text` field and `processed_text` field
            filtered_chunk = chunk[chunk['text'].notna() & chunk['text'].str.strip().astype(bool) &
                                   chunk['processed_text'].notna() & chunk['processed_text'].str.strip().astype(bool)]
            chunks.append(filtered_chunk)

        if chunks:
            df = pd.concat(chunks, ignore_index=True)
            logging.info(f"CSV data loaded and filtered successfully with {len(df)} rows.")

            # For debug purposes, we print the total rows and rows with empty processed_text
            print(f"Total rows: {total_rows}")
            print(f"Number of rows where 'processed_text' is empty: {empty_processed_text_count}")

            return df, empty_processed_text_count
        else:
            logging.warning("No data to load after filtering.")
            return pd.DataFrame(), empty_processed_text_count

    except pd.errors.ParserError as e:
        logging.error(f"Error parsing CSV file: {e}")
        return pd.DataFrame(), empty_processed_text_count

# Function to vectorize texts
def vectorize_texts(model, text_list):
    logging.info("Vectorizing texts")
    embeddings = model.encode(text_list, show_progress_bar=True)
    logging.info(f"Texts vectorized successfully. Number of embeddings: {len(embeddings)}")
    return embeddings

# Function to load JSON data from a folder
def load_data_json_folder(folder_path):
    logging.info(f"Loading JSON data from folder {folder_path}")
    data = []
    try:
        for filename in os.listdir(folder_path):
            if filename.endswith('.json'):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    loaded_data = json.load(f)
                    if isinstance(loaded_data, list):
                        data.extend(loaded_data)
                    else:
                        data.append(loaded_data)
        logging.info("JSON data loaded successfully.")
        return data
    except Exception as e:
        logging.error(f"Error loading JSON files: {e}")
        return data

# Function to associate comments with transcript segments
def associate_comments_with_transcript(model, transcripts, transcript_embeddings, comments_df, comment_embeddings, relevant_speakers):
    associations = []
    logging.info(f"Number of transcript segments: {len(transcripts)}")
    logging.info(f"Number of comment embeddings: {len(comment_embeddings)}")

    for i, transcript in tqdm(enumerate(transcripts), total=len(transcripts), desc="Processing Transcripts"):
        speaker = transcript.get('speaker', "0@:NOT PRESENT")
        if speaker not in relevant_speakers:
            logging.info(f'Skipping speaker: {speaker}')
            continue

        segment = {
            "speaker": speaker,
            "start": transcript.get('start', "0@:NOT PRESENT"),
            "end": transcript.get('end', "0@:NOT PRESENT"),
            "message": transcript.get('message', "0@:NOT PRESENT")  # Use the message attribute for display
        }

        # Link comments with similarity scores
        linked_comments = []
        for j, comment_vector in enumerate(comment_embeddings):
            similarity_score = model.similarity(comment_vector, transcript_embeddings[i])

            comment = comments_df.iloc[j]
            linked_comments.append({
                'comment_id': comment.get('author', "0@:NOT PRESENT"),
                'similarity': similarity_score,
                'comment_text': comment.get('text', "0@:NOT PRESENT"),
                'author': comment.get('author', "0@:NOT PRESENT"),
                'published_at': comment.get('published_at', "0@:NOT PRESENT"),
                'network': comment.get('network', "0@:NOT PRESENT")
            })

        # Sort comments based on similarity and retain the most relevant ones
        linked_comments = sorted(linked_comments, key=lambda x: x['similarity'], reverse=True)

        if linked_comments:
            associations.append({
                "segment": segment,
                "linked_comments": linked_comments
            })

    return associations

def main():
    parser = argparse.ArgumentParser(description="Process CSV and JSON data for transcript and comment embeddings")
    parser.add_argument("--csv_file", type=str, required=True, help="Path to CSV file containing comments")
    parser.add_argument("--transcripts_folder", type=str, required=True, help="Path to folder containing transcript JSON files")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output JSON file with associations")

    args = parser.parse_args()

    # Define relevant speakers
    relevant_speakers = {'President Joe Biden', 'Former President Donald Trump', 'Kamala Harris', 'Donald Trump'}

    # Load the SentenceTransformer model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Load and process CSV comments data
    processed_df, empty_processed_text_count = load_and_process_csv(args.csv_file)

    # Check the shape and number of non-empty processed_text
    print("Number of rows to be vectorized:", len(processed_df))

    # Vectorize the processed texts
    processed_texts = processed_df['processed_text'].tolist()
    comment_embeddings = vectorize_texts(model, processed_texts)

    # Check the shape of the embeddings
    print("Shape of embeddings array:", np.array(comment_embeddings).shape)
    print("Number of comment embeddings:", len(comment_embeddings))

    # Load and process transcript JSON data
    transcript_data = load_data_json_folder(args.transcripts_folder)

    # Vectorize the preprocessed transcript messages
    preprocessed_messages = [t.get('preprocessed_message', "0@:NOT PRESENT") for t in transcript_data]
    transcript_embeddings = vectorize_texts(model, preprocessed_messages)

    # Output the number of transcript embeddings
    print("Number of transcript embeddings:", len(transcript_embeddings))
    print("Shape of transcript embeddings array:", np.array(transcript_embeddings).shape)

    # Associate comments with transcript segments based on similarity
    associations = associate_comments_with_transcript(
        model, transcript_data, transcript_embeddings, processed_df, comment_embeddings, relevant_speakers)

    # Save the associations to a JSON file
    def convert_to_serializable(obj):
        if isinstance(obj, (np.ndarray, np.generic)):
            return obj.tolist()
        if isinstance(obj, (torch.Tensor)):
            return obj.tolist()
        return json.JSONEncoder().default(obj)

    with open(args.output_file, 'w') as f:
        json.dump(associations, f, indent=4, default=convert_to_serializable)
    logging.info(f"Associations saved to {args.output_file}")

if __name__ == "__main__":
    main()



# import pandas as pd
# import logging
# from sentence_transformers import SentenceTransformer
# import numpy as np
# import json
# import os
# import argparse
# from tqdm import tqdm
# import torch

# # Configure logging for the script
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler()])

# def load_and_process_csv(file_path, chunksize=10000):
#     logging.info(f"Loading CSV data from {file_path} in chunks")
#     chunks = []
#     empty_processed_text_count = 0  # Counter for rows with empty 'processed_text'
#     total_rows = 0  # Total row counter

#     try:
#         for chunk in pd.read_csv(file_path, encoding='utf-8', chunksize=chunksize):
#             total_rows += len(chunk)
#             # Count rows where 'processed_text' is empty or NaN
#             empty_processed_text_count += chunk['processed_text'].isna().sum()
#             empty_processed_text_count += chunk['processed_text'].str.strip().eq('').sum()

#             # Filter the chunk to keep only rows with non-empty `text` field and `processed_text` field
#             filtered_chunk = chunk[chunk['text'].notna() & chunk['text'].str.strip().astype(bool) &
#                                    chunk['processed_text'].notna() & chunk['processed_text'].str.strip().astype(bool)]
#             chunks.append(filtered_chunk)

#         if chunks:
#             df = pd.concat(chunks, ignore_index=True)
#             logging.info(f"CSV data loaded and filtered successfully with {len(df)} rows.")

#             # For debug purposes, we print the total rows and rows with empty processed_text
#             print(f"Total rows: {total_rows}")
#             print(f"Number of rows where 'processed_text' is empty: {empty_processed_text_count}")

#             return df, empty_processed_text_count
#         else:
#             logging.warning("No data to load after filtering.")
#             return pd.DataFrame(), empty_processed_text_count

#     except pd.errors.ParserError as e:
#         logging.error(f"Error parsing CSV file: {e}")
#         return pd.DataFrame(), empty_processed_text_count

# def vectorize_texts(model, text_list):
#     logging.info("Vectorizing texts")
#     embeddings = model.encode(text_list, show_progress_bar=True)
#     logging.info(f"Texts vectorized successfully. Number of embeddings: {len(embeddings)}")
#     return embeddings

# def load_data_json_folder(folder_path):
#     logging.info(f"Loading JSON data from folder {folder_path}")
#     data = []
#     try:
#         for filename in os.listdir(folder_path):
#             if filename.endswith('.json'):
#                 file_path = os.path.join(folder_path, filename)
#                 with open(file_path, 'r', encoding='utf-8') as f:
#                     loaded_data = json.load(f)
#                     if isinstance(loaded_data, list):
#                         data.extend(loaded_data)
#                     else:
#                         data.append(loaded_data)
#         logging.info("JSON data loaded successfully.")
#         return data
#     except Exception as e:
#         logging.error(f"Error loading JSON files: {e}")
#         return data

# def associate_comments_with_transcript(model, preprocessed_transcripts, transcript_embeddings, comments_df, comment_embeddings, combined_transcripts, relevant_speakers):
#     associations = []
#     logging.info(f"Number of transcript segments: {len(preprocessed_transcripts)}")
#     logging.info(f"Number of comment embeddings: {len(comment_embeddings)}")

#     # Ensure both preprocessed and combined transcripts are aligned by checking their lengths
#     if len(preprocessed_transcripts) != len(combined_transcripts):
#         logging.error("The number of preprocessed and combined transcripts do not match!")
#         return associations
    
#     # Ensure that each transcript matches the corresponding one in the combined transcripts
#     for i, (preprocessed, combined) in tqdm(enumerate(zip(preprocessed_transcripts, combined_transcripts)), total=len(preprocessed_transcripts), desc="Verifying Transcript Alignment"):
#         if preprocessed.get('speaker') != combined.get('speaker') or preprocessed.get('start') != combined.get('start') or preprocessed.get('end') != combined.get('end'):
#             logging.error(f"Mismatch found between preprocessed and combined transcripts at index {i}")
#             logging.error(f"Preprocessed: {preprocessed}")
#             logging.error(f"Combined: {combined}")
#             return associations

#     for i, transcript in tqdm(enumerate(preprocessed_transcripts), total=len(preprocessed_transcripts), desc="Processing Transcripts"):
#         speaker = transcript.get('speaker', "0@:NOT PRESENT")
#         if speaker not in relevant_speakers:
#             logging.info(f'Skipping speaker: {speaker}')
#             continue

#         combined_message = combined_transcripts[i].get('message', "0@:NOT PRESENT")

#         segment = {
#             "speaker": speaker,
#             "start": transcript.get('start', "0@:NOT PRESENT"),
#             "end": transcript.get('end', "0@:NOT PRESENT"),
#             "message": combined_message  # Use message from combined transcripts for display
#         }

#         # Link comments with similarity scores
#         linked_comments = []
#         for j, comment_vector in enumerate(comment_embeddings):
#             similarity_score = model.similarity(comment_vector, transcript_embeddings[i])

#             comment = comments_df.iloc[j]
#             linked_comments.append({
#                 'comment_id': comment.get('author', "0@:NOT PRESENT"),
#                 'similarity': similarity_score,
#                 'comment_text': comment.get('text', "0@:NOT PRESENT"),
#                 'author': comment.get('author', "0@:NOT PRESENT"),
#                 'published_at': comment.get('published_at', "0@:NOT PRESENT"),
#                 'network': comment.get('network', "0@:NOT PRESENT")
#             })

#         # Sort comments based on similarity and retain the most relevant ones
#         linked_comments = sorted(linked_comments, key=lambda x: x['similarity'], reverse=True)

#         if linked_comments:
#             associations.append({
#                 "segment": segment,
#                 "linked_comments": linked_comments
#             })

#     return associations

# def main():
#     parser = argparse.ArgumentParser(description="Process CSV and JSON data for transcript and comment embeddings")
#     parser.add_argument("--csv_file", type=str, required=True, help="Path to CSV file containing comments")
#     parser.add_argument("--preprocessed_folder", type=str, required=True, help="Path to folder containing preprocessed transcript JSON files")
#     parser.add_argument("--combined_folder", type=str, required=True, help="Path to folder containing combined transcript JSON files")
#     parser.add_argument("--output_file", type=str, required=True, help="Path to output JSON file with associations")

#     args = parser.parse_args()

#     # Define relevant speakers
#     relevant_speakers = {'President Joe Biden', 'Former President Donald Trump', 'Kamala Harris', 'Donald Trump'}

#     # Load the SentenceTransformer model
#     model = SentenceTransformer("all-MiniLM-L6-v2")

#     # Load and process CSV comments data
#     processed_df, empty_processed_text_count = load_and_process_csv(args.csv_file)

#     # Check the shape and number of non-empty processed_text
#     print("Number of rows to be vectorized:", len(processed_df))

#     # Vectorize the processed texts
#     processed_texts = processed_df['processed_text'].tolist()
#     comment_embeddings = vectorize_texts(model, processed_texts)

#     # Check the shape of the embeddings
#     print("Shape of embeddings array:", np.array(comment_embeddings).shape)
#     print("Number of comment embeddings:", len(comment_embeddings))

#     # Load and process preprocessed transcript JSON data
#     preprocessed_transcript_data = load_data_json_folder(args.preprocessed_folder)

#     # Load and process combined transcript JSON data (for display)
#     combined_transcript_data = load_data_json_folder(args.combined_folder)

#     # Vectorize the preprocessed transcript messages
#     preprocessed_messages = [t.get('message', "0@:NOT PRESENT") for t in preprocessed_transcript_data]
#     transcript_embeddings = vectorize_texts(model, preprocessed_messages)

#     # Output the number of transcript embeddings
#     print("Number of transcript embeddings:", len(transcript_embeddings))
#     print("Shape of transcript embeddings array:", np.array(transcript_embeddings).shape)

#     # Associate comments with transcript segments based on similarity, using combined messages for display
#     associations = associate_comments_with_transcript(
#         model, preprocessed_transcript_data, transcript_embeddings, processed_df, comment_embeddings, combined_transcript_data, relevant_speakers)

#     # Save the associations to a JSON file
#     def convert_to_serializable(obj):
#         if isinstance(obj, (np.ndarray, np.generic)):
#             return obj.tolist()
#         if isinstance(obj, (torch.Tensor)):
#             return obj.tolist()
#         return json.JSONEncoder().default(obj)

#     with open(args.output_file, 'w') as f:
#         json.dump(associations, f, indent=4, default=convert_to_serializable)
#     logging.info(f"Associations saved to {args.output_file}")

# if __name__ == "__main__":
#     main()