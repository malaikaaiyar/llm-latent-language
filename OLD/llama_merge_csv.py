# %%
import pandas as pd
import os
# %%
def filter_matching_translations(df, **kwargs):
    # Identify columns that represent language translations by excluding probability columns
    lang_cols = [col for col in df.columns if col in ["en", "fr", "zh", "de", "ru", "ko"]]
    
    # Define a filter function to detect any rows with duplicate translations
    def has_duplicate_translations(row):
        # Check the row values for the language columns, if there are duplicates among them
        translations = row[lang_cols].tolist()
        return len(set(translations)) != len(translations)
    
    # Apply the filter function to identify rows with duplicate translations
    mask = df.apply(has_duplicate_translations, axis=1)
    
    # Filter out the rows where any translations are duplicated
    return df[~mask]
# %%

def filter_by_probability_threshold(df, trans_thresh = -1, **kwargs):
    # Identify columns that end with '_prob'
    prob_columns = [col for col in df.columns if col.endswith('_prob')]

    # Use DataFrame's filter with conditions based on the threshold
    # Keep only rows where all probability columns meet or exceed the threshold
    df_filtered = df[df[prob_columns].min(axis=1) >= trans_thresh]
    df_filtered = df_filtered.reset_index(drop=True)
    return df_filtered    

def construct_dataset(src_lang=None, 
                      latent_lang = None, 
                      dest_lang = None, 
                      dataset_path = "data/synth_llama2",
                      **kwargs):
        # Load the full data from each file
    lang_cols = [src_lang, latent_lang, dest_lang]
    assert len(set(lang_cols)) == 3, "ERROR: src, latent, and dest must be different"
    
    data_frames = []
    for lang in lang_cols:
        if lang != "en":
            data_frames.append(pd.read_csv(os.path.join(dataset_path, f"llama2_en_to_{lang}.csv")))

    # Merge the data_frames
    df_merged = data_frames[0]
    for df in data_frames[1:]:
        df_merged = df_merged.merge(df, on="en", how="inner")
    # Shuffle the columns so that src, latent, and dest are the first three columns
    new_cols = lang_cols + [col for col in df_merged.columns if col not in lang_cols]
    df_merged = df_merged[new_cols] #reorder columns
    df_merged = filter_matching_translations(df_merged, **kwargs) #remove rows with duplicate translations
    df_merged = filter_by_probability_threshold(df_merged, **kwargs) #remove rows with low probability translations
    df_merged = df_merged.reset_index(drop=True)
    return df_merged



# %%

# %%

# %%
def gen_batched_translation_task(df, vocab, **kwargs):
    """
    Generate a dataset for training a model using the given dataframe, vocabulary, and configuration.

    Args:
        df (pandas.DataFrame): The input dataframe containing the data.
        vocab (list): The vocabulary used for tokenization.
        cfg (Config): The configuration object containing the language settings and other parameters.

    Returns:
        list: A list of dictionaries, where each dictionary represents a datapoint in the dataset. Each dictionary contains the following keys:
            - 'prompt': The prompt string used for training.
            - 'out_ids': The token IDs of the output tokens.
            - 'out_str': The string representation of the output tokens.
            - 'latent_ids': The token IDs of the latent tokens.
            - 'latent_str': The string representation of the latent tokens.
            - 'in_str': The string representation of the input tokens.
    """
    src_lang = kwargs.get('src_lang', 'fr')
    dest_lang = kwargs.get('dest_lang', 'zh')
    latent_lang = kwargs.get('latent_lang', 'en')
    k = kwargs.get('num_multi_shot', 1)
    unique_prompt = kwargs.get('unique_prompt', True)
    prompt_bank = kwargs.get('prompt_bank', 0)
    seed = kwargs.get('seed', 42)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    prompt = generate_translation_prompt(src_lang, dest_lang, None)
    common_suffixes = generate_common_suffixes(src_lang, dest_lang, df[src_lang], df[dest_lang])
    
    
    
    
    
    dataset = []
    for ind in tqdm(range(len(df))):
        df = df.reset_index(drop=True)
        temp = df[df.index!=ind]
        sample = pd.concat([temp.sample(k), df[df.index==ind]], axis=0)
        prompt = ""
        src_space = "" if src_lang == "zh" else " "
        dest_space = "" if dest_lang == "zh" else " "
        for idx, (df_idx, row) in enumerate(sample.iterrows()):
            if idx < k-1:
                prompt += f'{lang2name[src_lang]}: "{src_space}{row[src_lang]}" - {lang2name[dest_lang]}: "{dest_space}{row[dest_lang]}"\n'
            elif idx == k-1:
                prompt += f'{lang2name[src_lang]}: "{src_space}{row[src_lang]}" - {lang2name[dest_lang]}: "'
                if dest_lang == 'zh':
                    prompt += ' '
                in_str, out_str, latent_str = row[src_lang], row[dest_lang], row[latent_lang]
                out_ids = find