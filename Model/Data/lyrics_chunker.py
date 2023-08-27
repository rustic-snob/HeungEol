import pandas as pd
import yaml
import re
import logging
import random

from tqdm.auto import tqdm
from copy import deepcopy

if __name__ == "__main__":
    # Put config.yaml in './Configurations' folder.
    with open('./Configurations/config.yaml') as f:
        CFG = yaml.load(f, Loader=yaml.FullLoader)

# Put raw data in './Dataset/raw' folder.
df = pd.read_csv('./Dataset/raw/' + CFG['data_name'])
print('Raw Data: ' + str(len(df)))

# DataCleaning
df['가사'] = df['가사'].apply(lambda x: re.sub('[^가-힣a-zA-Z \n]+', '', x))
df['가사'] = df['가사'].apply(lambda x: re.sub('[\n]+', '\n', x))

def first_word_space_sequence(s):
    match = re.search('[\w\s]+', s)
    if match:
        return match.group()
    return None

df['제목'] = df['제목'].apply(first_word_space_sequence)

# Exclude Eng line
df['가사'] = df['가사'].str.split('\n')
df['가사'] = df['가사'].apply(lambda x: [item for item in x if not re.search('[a-zA-Z]', item)])

df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# If use_line == True: every new line is sample, every space is unit
if CFG['use_line']:
    
    df_use_line = deepcopy(df.explode('가사').reset_index(drop=True)).drop_duplicates(subset = ['가사']).dropna()
    
    df_use_line['gen_lyrics'] = df_use_line['가사'].apply(lambda x: ' / '.join(x.split()))
    
    df_use_line['gen_notes'] = df_use_line['가사'].apply(lambda x: ' / '.join([str(len(word)) for word in x.split()]))
    
    print('Exploded by Line Data: ' + str(len(df_use_line)))
    
def split_at_center_space(s):
    if len(s.replace(' ', '')) > CFG['threshold']:
        spaces = [i for i, char in enumerate(s) if char == ' ']
        if spaces and (random.uniform(0, 1) > CFG['long_prob']):
            center_space = min(spaces, key=lambda x: abs(x - len(s) / 2))
            return s[:center_space], s[center_space+1:]
    return s,

# Split segments with more than 10 characters at the most central space
df['가사'] = df['가사'].apply(lambda x: [segment for part in x for segment in split_at_center_space(part)])

# If use_whole == True: whole lyrics is sample, every new line is unit
if CFG['use_whole']:
    df_use_whole = deepcopy(df)
    
    df_use_whole['gen_lyrics'] = df_use_whole['가사'].apply(lambda x: ' / '.join(x))
    
    df_use_whole['gen_notes'] = df_use_whole['가사'].apply(lambda x: ' / '.join([str(len(word.replace(' ',''))) for word in x]))
    
    print('Whole song Data: ' + str(len(df_use_whole)))

def process_sample(sample, chunk_len):
    notes = [len(segment.replace(' ', '')) for segment in sample]
    
    new_lyrics, new_notes = [], []
    acc_notes, temp_lyrics, temp_notes = 0, [], []
    
    for l, n in zip(sample, notes):
        if acc_notes + n > chunk_len:
            new_lyrics.append(' / '.join(temp_lyrics))
            new_notes.append(' / '.join(map(str, temp_notes)))
            temp_lyrics, temp_notes = [], []
            acc_notes = 0
        temp_lyrics.append(l)
        temp_notes.append(n)
        acc_notes += n
    
    if temp_lyrics:
        new_lyrics.append(' / '.join(temp_lyrics))
        new_notes.append(' / '.join(map(str, temp_notes)))
    
    return new_lyrics, new_notes

all_lyrics, all_notes = [], []

for chunk_len in CFG['chunk_len']:
    chunked_lyrics, chunked_notes = zip(*df['가사'].apply(lambda x: process_sample(x, chunk_len)))
    all_lyrics.append(chunked_lyrics)
    all_notes.append(chunked_notes)

# Create an empty DataFrame with the same columns as df
new_df = pd.DataFrame(columns=df.columns)
new_df['gen_lyrics'] = None
new_df['gen_notes'] = None

all_rows = []

for idx, row in tqdm(df.iterrows()):
    lyrics_lists = [lst[idx] for lst in all_lyrics]
    notes_lists = [lst[idx] for lst in all_notes]
    for lyrics_list, notes_list in zip(lyrics_lists, notes_lists):
        for l, n in zip(lyrics_list, notes_list):
            new_row = row.copy()
            new_row['gen_lyrics'] = l
            new_row['gen_notes'] = n
            all_rows.append(new_row)

new_df = pd.DataFrame(all_rows)

final_df = pd.concat([new_df, df_use_line, df_use_whole], ignore_index=True)

final_df.drop_duplicates(subset=['gen_lyrics'], inplace = True)
final_df.dropna(inplace = True, subset=['gen_lyrics', 'gen_notes', '장르', '제목'])
final_df.drop(['가사', '가수'], axis = 1, inplace=True)

print('Final Data: ' + str(len(final_df)))

final_df.to_csv('./Dataset/ready/cooked_' + CFG['data_name'], sep=',', na_rep='NaN',index=False)