from transformers import MarianMTModel, MarianTokenizer

# Initialize models and tokenizers for both directions
# English to Swahili
en_sw_model_name = 'Helsinki-NLP/opus-mt-en-swc'
en_sw_tokenizer = MarianTokenizer.from_pretrained(en_sw_model_name)
en_sw_model = MarianMTModel.from_pretrained(en_sw_model_name)

# Swahili to English
sw_en_model_name = 'Helsinki-NLP/opus-mt-swc-en'
sw_en_tokenizer = MarianTokenizer.from_pretrained(sw_en_model_name)
sw_en_model = MarianMTModel.from_pretrained(sw_en_model_name)

def translate(text, source_lang='en', target_lang='sw'):
    """Translate text between English and Swahili"""
    if source_lang == 'en' and target_lang == 'sw':
        # English to Swahili
        tokenized = en_sw_tokenizer([text], return_tensors="pt", truncation=True)
        translated = en_sw_model.generate(**tokenized)
        return en_sw_tokenizer.decode(translated[0], skip_special_tokens=True)
    elif source_lang == 'sw' and target_lang == 'en':
        # Swahili to English
        tokenized = sw_en_tokenizer([text], return_tensors="pt", truncation=True)
        translated = sw_en_model.generate(**tokenized)
        return sw_en_tokenizer.decode(translated[0], skip_special_tokens=True)
    else:
        return "Unsupported language pair"

# Test examples
english_text = 'I am going to the market'
swahili_text = 'Ninacheza na mtoto'
# Translate English to Swahili
translated_sw = translate(english_text, 'en', 'sw')
print(f"English: {english_text}")
print(f"Swahili: {translated_sw}")

# Translate Swahili back to English
translated_en = translate(swahili_text, 'sw', 'en')
print(f"\nSwahili: {swahili_text}")
print(f"English: {translated_en}")

# Another test
english_text2 = "The weather is nice today"
swahili_text2 = "Nataka kununua mkate"


print("\nSecond test:")
print(f"English: {english_text2}")
print(f"Swahili: {translate(english_text2, 'en', 'sw')}")
print(f"\nSwahili: {swahili_text2}")
print(f"English: {translate(swahili_text2, 'sw', 'en')}")