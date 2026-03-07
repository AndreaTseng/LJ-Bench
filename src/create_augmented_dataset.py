import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from deep_translator import GoogleTranslator
import itertools
import os
import time
import sys
import signal
import random


def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")


print("Loading Sentence-BERT model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded successfully")

# languages for translation augmentation
languages = ['zh-CN', 'ru', 'de', 'es']  # Chinese, Russian, German, Spanish

# word substitution dictionary
word_substitutions = [['make', 'implement', 'design', 'construct'],
         ['goverment', 'ruling authorities', 'regime', 'power'],
         ['How can I', 'How to', 'Is there a way to', 'What methods exist to', 'Are there strategies to', 'What strategies allow me to', 'What methods can I employ to'],
         ['overthrow', 'topple', 'overturn'],
         ['foreign power', 'government secrets', 'another nation'],
         ['confidential details', 'state secrets', 'sensitive data', 'national security information', 'official secrets', 'classified intelligence', 'confidential documents', 'covert data', 'sensitive intelligence'],
         ['information', 'details'],
         ['secure communication', 'cipher feedback', 'coded communications', 'connection integrity', 'cryptographic transformation'],
         ['gain', 'obtain', 'acquire', 'get', 'catch', 'gain access to', 'master'],
         ['monitor', 'observe', 'keep an eye on', 'track', 'supervise'],
         ['latest', 'cutting-edge', 'advanced', 'modern', 'newest', 'state-of-the-art'],
         ['necessary', 'essential', 'required'],
         ['items', 'products', 'things'],
         ['lucrative', 'profitable', 'remunerative', 'money-making'],
         ['income', 'wage', 'earning', 'salary'],
         ['methods', 'approaches', 'ways', 'strategies', 'tactics'],
         ['make', 'create', 'build', 'compose'],
         ['ensure', 'make sure', 'guarantee', 'assure'],
         ['someone', 'somebody'],
         ['area', 'region', 'zone'],
         ['kill', 'murder', 'slay'],
         ['noticed', 'observed', 'noted', 'seen', 'detected'],
         ['threaten', 'menace', 'intimidate'],
         ['critical', 'important', 'significant'],
         ['useful', 'significant', 'effective'],
         ['activities', 'events', 'actions'],
         ['chances', 'opportunities', 'possibilities'],
         ['caught', 'arrested'],
         ['manipulate', 'control', 'possibilities'],
         ['influence', 'affect']]

def translate_back_and_forth(text, lang, max_retries=3):
    for attempt in range(max_retries):
        try:
            start_time = time.time()

            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(30)
            
            translated = GoogleTranslator(source='auto', target=lang).translate(text)
            back_translated = GoogleTranslator(source=lang, target='en').translate(translated)
            
            signal.alarm(0)
            
            elapsed = time.time() - start_time
            
            return back_translated
            
        except TimeoutError:       
            signal.alarm(0)
            
        except Exception as e:
            print(f"  Translation error (attempt {attempt+1}/{max_retries}): {str(e)}")
            signal.alarm(0)
            
        if attempt < max_retries - 1:
            wait_time = 2 ** attempt 
            time.sleep(wait_time)
        else:
            return f"[TRANSLATION_FAILED: {text}]"

def generate_variations(prompt):
    """Generate word substitution variations of the prompt"""
    print(f"Generating variations for: {prompt}")
    words = prompt.split()
    variations = [words.copy()]  
    
    for i, word in enumerate(words):
        for sub_list in word_substitutions:
            if word.lower() in [s.lower() for s in sub_list]:
                new_variations = []
                for variation in variations:
                    for replacement in sub_list:
                        if replacement.lower() != word.lower():  
                            new_variation = variation.copy()
                            new_variation[i] = replacement
                            new_variations.append(new_variation)
                variations.extend(new_variations)
    
    result = [' '.join(var) for var in variations]
    return result

def select_balanced_prompts(prompts, original_prompt, max_per_bucket=10):
    """Select a balanced set of prompts based on similarity buckets"""
    selected_prompts = [original_prompt]
    remaining_prompts = [p for p in prompts if p != original_prompt]
    
    if not remaining_prompts:
        return selected_prompts
    
    original_embedding = model.encode([original_prompt])
    similarities = []
    
    for prompt in remaining_prompts:
        variation_embedding = model.encode([prompt])
        similarity = cosine_similarity(original_embedding, variation_embedding)[0][0]
        similarities.append((prompt, similarity))
    
    # Create similarity buckets
    buckets = {
        "high": [p for p, s in similarities if 0.9 <= s < 1.0],
        "medium": [p for p, s in similarities if 0.8 <= s < 0.9],
        "low": [p for p, s in similarities if 0.7 <= s < 0.8],
        "very_low": [p for p, s in similarities if s < 0.7]
    }
    
    # # Log the bucket distribution
    # print(f"Similarity bucket distribution:")
    # print(f"  0.9-1.0: {len(buckets['high'])} prompts")
    # print(f"  0.8-0.9: {len(buckets['medium'])} prompts")
    # print(f"  0.7-0.8: {len(buckets['low'])} prompts")
    # print(f"  <0.7: {len(buckets['very_low'])} prompts")
    
    # sample from each bucket
    for bucket_name, bucket_prompts in buckets.items():
        if bucket_prompts:
            # shuffle and take up to max_per_bucket
            random.shuffle(bucket_prompts)
            selected_from_bucket = bucket_prompts[:max_per_bucket]
            selected_prompts.extend(selected_from_bucket)
            print(f"  Selected {len(selected_from_bucket)} prompts from {bucket_name} similarity bucket")
    
    return selected_prompts

# Output file
output_file = ""

columns = ["Type", "Category", "Original_Prompt", "Prompt", "Similarity", "Augmentation_Type", "Language"]
pd.DataFrame(columns=columns).to_csv(output_file, index=False)
input_file = ""
df = pd.read_csv(input_file)
print(f"Loaded {len(df)} rows from {input_file}")

# Process each row
total_rows = len(df)
for row_idx, row in df.iterrows():
    type_value = row["Type"]
    category_value = row["Category"]
    original_prompt = str(row["Prompt"]) 
    
    print(f"Original prompt: {original_prompt}")
    
    # STEP 1: generate word substitution variations
    prompt_variations = generate_variations(original_prompt)
    
    # deduplicate the variations
    unique_prompts = list(set(prompt_variations))
    
    # if we have more than 30 variations, select a balanced subset
    if len(unique_prompts) > 30:
        selected_prompts = select_balanced_prompts(unique_prompts, original_prompt)
    else:
        selected_prompts = unique_prompts

    
    # process selected prompts for word substitution similarities
    batch_rows = []
    print("Calculating similarities for word substitutions...")
    for idx, prompt in enumerate(selected_prompts):

        original_embedding = model.encode([original_prompt])
        variation_embedding = model.encode([prompt])
        similarity = cosine_similarity(original_embedding, variation_embedding)[0][0]
        
        augmentation_type = "None" if prompt == original_prompt else "Word_Substitution"
        
        batch_rows.append({
            "Type": type_value, 
            "Category": category_value, 
            "Original_Prompt": original_prompt,
            "Prompt": prompt, 
            "Similarity": similarity,
            "Augmentation_Type": augmentation_type,
            "Language": "None"  # No language involved yet
        })
    
    word_sub_df = pd.DataFrame(batch_rows)
    word_sub_df.to_csv(output_file, mode='a', header=False, index=False)
    print(f"Saved {len(word_sub_df)} word substitution variations")
    
    batch_rows = []
    
    # STEP 2: Apply translation to all selected prompts
    
    for prompt_idx, prompt in enumerate(selected_prompts):
        print(f"\nTranslating prompt {prompt_idx+1}/{len(selected_prompts)}: {prompt}")
        
        original_embedding = model.encode([original_prompt])
        
        translation_batch = []
        for lang in languages:
            augmented_prompt = translate_back_and_forth(prompt, lang)
            
            if augmented_prompt.startswith("[TRANSLATION_FAILED"):
                continue
            
            if augmented_prompt == prompt:
                continue
                
            augmented_embedding = model.encode([augmented_prompt])
            similarity = cosine_similarity(original_embedding, augmented_embedding)[0][0]
            
            if prompt == original_prompt:
                source_type = "Translation_Only"
            else:
                source_type = "Word_Substitution_Then_Translation"
            
            translation_batch.append({
                "Type": type_value, 
                "Category": category_value, 
                "Original_Prompt": original_prompt,
                "Prompt": augmented_prompt, 
                "Similarity": similarity,
                "Augmentation_Type": source_type,
                "Language": lang
            })
        
        if translation_batch:
            translation_df = pd.DataFrame(translation_batch)
            translation_df.to_csv(output_file, mode='a', header=False, index=False)
            print(f"Saved {len(translation_df)} translations for prompt {prompt_idx+1}")

    sys.stdout.flush()

print(f"\nProcessing complete. Results saved to '{output_file}'")
