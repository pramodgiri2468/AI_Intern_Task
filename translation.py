# translation.py
from translate import Translator

def translate(text, src='en', dest='ne'):
    try:
        translator = Translator(from_lang=src, to_lang=dest)
        return translator.translate(text)
    except Exception as e:
        print(f"Translation error: {e}")
        return text  # Fallback to original text