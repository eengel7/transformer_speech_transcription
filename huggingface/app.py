from transformers import pipeline
import gradio as gr
import os
import deepl
import openai
from pytube import YouTube

TARGET_LANG = "EN-GB"
deepl_key = os.environ.get('DEEPL_KEY')

translator = deepl.Translator(deepl_key)
pipe = pipeline(model="torileatherman/whisper_small_sv")  # change to "your-username/the-name-you-picked"

def transcribe(audio):
    text_sv = pipe(audio)["text"]
    print(f"Audio transcribed: {text_sv}")
    text_en = translator.translate_text(text_sv, target_lang=TARGET_LANG).text
    print(f"Text translated: {text_en}")
    return text_sv, text_en

def transcribe_url(url):
    youtube = YouTube(str(url))
    audio = youtube.streams.filter(only_audio=True).first().download('yt_video')
    text_sv = pipe(audio)["text"]
    text_en = translator.translate_text(text_sv, target_lang=TARGET_LANG).text
    return text_sv, text_en

url_demo = gr.Interface(
    fn=transcribe_url, 
    inputs="text", 
    outputs=[gr.Textbox(label="Transcribed text"),
             gr.Textbox(label="English translation")],
    title="Swedish video speech to english text",
    description="Transcribing swedish video to text and translating to english!",
)

voice_demo = gr.Interface(
    fn=transcribe, 
    inputs=gr.Audio(source="microphone", type="filepath"), 
    outputs=[gr.Textbox(label="Transcribed text"),
             gr.Textbox(label="English translation")],
    title="Swedish recorded speech to english text",
    description="Transcribing swedish speech to text and translating to english!",
)

demo = gr.TabbedInterface([url_demo, voice_demo], ["Swedish YouTube Video to English Text", "Swedish Audio to English Text"])

demo.launch()
