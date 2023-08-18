import requests
import os
import streamlit as st
import whisper

from audiorecorder import audiorecorder
from transformers import pipeline
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Humming2Text

def transcribe(whisper_model, audio):
    
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(whisper_model.device)

    # detect the spoken language
    _, probs = whisper_model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")

    # decode the audio
    options = whisper.DecodingOptions()
    result = whisper.decode(whisper_model, mel, options)
    return result.text

def make_syl_structure(hum):
    return ' / '.join([str(len(h)) for h in hum.split()])

def generate_lyrics(HeungEol_model, conditions):

    # make template for prompts
    gen_notes = conditions['gen_notes']
    title = conditions['title']
    genre = conditions['genre']

    template = """
    ### Instruction(명령어):
    다음 조건에 어울리는 가사를 생성하시오. 주어진 음절 수를 절대 벗어나지 말 것. 제목과 장르에 어울려야 할 것. 생성 형식은 ['가사 / 가사 / 가사 / 가사']와 같음.
    
    ### Input(입력):
    음절 수는 [{gen_notes}], 제목은 [{title}], 장르는 [{genre}]이다.
    
    ### Response(응답): 

    """

    st.write('input as follow:\n' + template)

    prompt = PromptTemplate(template=template, input_variables=["gen_notes", "title", "genre"])

    # Chain them!
    HeungEol_chain = LLMChain(llm=HeungEol_model, prompt=prompt, verbose=True)

    st.write('now generating lyrics...')

    lyrics = HeungEol_chain.predict(gen_notes = gen_notes, title = title, genre = genre)

    print(lyrics)

    return lyrics


#### Now for streamlit

st.set_page_config(page_title="HeungEol_Demo", page_icon="🎵")

st.header("You 흥얼, I 옹알.")
audio = audiorecorder("Click to record", "Recording...")

whisper_model = whisper.load_model("base")

# initialize HF LLM
HeungEol_model = HuggingFaceHub(
    repo_id="snob/HeungEol-KoAlpaca-12.8B-v1.0",
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
    model_kwargs={"temperature":0.8, "max_length":512, "repetition_penalty": 1.2}
)

if len(audio) > 0:
    # To play audio in frontend:
    st.audio(audio.tobytes())
    
    # To save audio to a file:
    wav_file = open("audio.mp3", "wb")
    wav_file.write(audio.tobytes())

    hum = transcribe(whisper_model, "audio.mp3")

    st.write(hum)

    syl_structure = make_syl_structure(hum)

    st.text("Song's Syl_structure is: "+ syl_structure)

    title = st.text_input("Write Song's Title and Press Enter")
    st.write("Song's Title:", title)

    genre = st.radio(
        "Select Song's Genre and Press Enter",
        ('발라드', '인디음악', '댄스', '포크/블루스'))

    st.write("Song's Genre:", genre)

    if syl_structure and title and genre:
        if st.button('HeungEol!'):
            conditions = {"gen_notes":syl_structure,
                          "title":title,
                          "genre":genre}

            lyrics = generate_lyrics(HeungEol_model, conditions)

            st.write(lyrics)