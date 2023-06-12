import os
import json
from google.cloud import texttospeech
from google.cloud import translate_v2 as translate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import openai


class TextToSpeechConverter:
    def __init__(self):
        credentials_path = os.path.join(os.path.dirname(__file__), r'credentials.json')
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path

        with open("Text-To-Speech/config.json", "r") as file:
            self.config_data = json.load(file)

        openai.api_key = "xyz"

        self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

    def translate_text(self, text, target_language):
        translate_client = translate.Client()
        translation = translate_client.translate(text, target_language, source_language='en')
        translated_text = translation['translatedText']
        return translated_text

    def synthesize_speech(self, text, target_language):
        tts_client = texttospeech.TextToSpeechClient()

        input_language_data = self.config_data.get(target_language)
        if input_language_data:
            voice_language_code = input_language_data["voice_language_code"]
            voice_name = input_language_data["voice_name"]
            audio_config_data = input_language_data["audio_config"]

            input_text = texttospeech.SynthesisInput(text=text)
            audio_config = texttospeech.AudioConfig(audio_config_data)

            response = tts_client.synthesize_speech(
                input=input_text,
                voice=texttospeech.VoiceSelectionParams(
                    language_code=voice_language_code, name=voice_name
                ),
                audio_config=audio_config
            )

            output_file = f"output_{target_language}.mp3"
            with open(output_file, "wb") as out:
                out.write(response.audio_content)

            return output_file

    def summarise_gpt(self, content, words, target_language):
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=[
                {"role": "system", "content": f"Write a summary of the text in {target_language}"},
                {"role": "user", "content": content}
            ],
            max_tokens=words,
            n=1
        )
        summary = response.choices[0].message.content
        return summary

    def flow_one(self, content, target_language):
        long_summary = self.summarise_gpt(content, 200, 'english')
        translate_content = self.translate_text(long_summary, target_language)
        audio = self.synthesize_speech(translate_content, target_language)
        return audio

    def convert_to_speech(self, content_file_path, target_language):
        with open(content_file_path, 'r') as file:
            content = file.read()

        audio = self.flow_one(content, target_language)
        return audio

converter = TextToSpeechConverter()
target_language = 'ml'
content_file_path = r'Text-To-Speech\content.txt'
audio_file = converter.convert_to_speech(content_file_path, target_language)
print(audio_file)
