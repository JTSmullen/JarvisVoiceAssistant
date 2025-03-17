import speech_recognition as sr
import pyttsx3
from vosk import Model, KaldiRecognizer
import os
import json
import google.generativeai as genai
from datetime import datetime

class VoiceAssistant:
    def __init__(self, google_api_key, vosk_model_path):
        genai.configure(api_key=google_api_key)
        self.genai_model = genai.GenerativeModel('gemini-2.0-flash')
        self.recognizer = sr.Recognizer()
        self.engine = pyttsx3.init()
        self.vosk_model_path = vosk_model_path
        self.model = self._load_vosk_model()
        self.conversation_history = []
        self.max_history_length = 5

    def _load_vosk_model(self):
        try:
            return Model(self.vosk_model_path)
        except Exception as e:
            print(f"Error loading Vosk model: {e}")
            print("Please ensure that the MODEL_PATH is correct and the model files are present.")
            exit()

    def speak(self, text):
        self.engine.say(text)
        self.engine.runAndWait()

    def listen_vosk(self):
        with sr.Microphone() as source:
            print("Say something!")
            audio = self.recognizer.listen(source)

        try:
            recognizer = KaldiRecognizer(self.model, source.SAMPLE_RATE)
            raw_audio = audio.get_raw_data()
            recognizer.AcceptWaveform(raw_audio)
            result = recognizer.FinalResult()
            text = json.loads(result)['text'].lower()
            print("You said (Vosk): {}".format(text))
            return text

        except Exception as e:
            print(f"Error during Vosk transcription: {e}")
            return ""

    def generate_response(self, user_input):
        self.conversation_history.append({"role": "user", "content": user_input})

        try:
            prompt_parts = [f"{turn['role']}: {turn['content']}\n" for turn in self.conversation_history]
            prompt = "".join(prompt_parts)

            response = self.genai_model.generate_content(prompt + ". Respond in one or two sentences. Respond in a more human way of speaking. Speak in brain rot terms")
            response_text = response.text
            self.conversation_history.append({"role": "model", "content": response_text})
            return response_text, self.conversation_history

        except Exception as e:
            print(f"Error generating response: {e}")
            error_message = "I encountered an error. Please try again."
            return error_message, self.conversation_history

    def run(self):
        while True:
            user_input = self.listen_vosk()

            if user_input:
                response, self.conversation_history = self.generate_response(user_input)
                print("Response: {}".format(response))
                self.speak(response)

                if "goodbye" in user_input:
                    break

if __name__ == '__main__':
    google_api_key = os.environ["GOOGLE_API_KEY"]
    vosk_model_path = os.path.join(os.path.dirname(__file__), "vosk-model-small-en-us-0.15")
    assistant = VoiceAssistant(google_api_key, vosk_model_path)
    assistant.run()