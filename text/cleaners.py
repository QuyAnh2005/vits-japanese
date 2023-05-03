import MeCab
import pykakasi

mc = MeCab.Tagger("-Owakati")
kks = pykakasi.kakasi()

# define function to convert text to phonemes
def japanese_to_phonemes(text):
    # convert text to hiragana
    result = kks.convert(text)
    hiragana = ''.join([item['hira'] for item in result])

    # convert hiragana to katakana
    katakana = kks.convert(hiragana)
    katakana = ''.join([item['kana'] for item in katakana])

    # convert katakana to romaji (phonemes)
    romaji = kks.convert(katakana)
    romaji = ''.join([item['hepburn'] for item in romaji])

    return romaji

def japanese_cleaner(text):
    cleaned_text = mc.parse(text)
    phonemes = japanese_to_phonemes(cleaned_text)
    return phonemes

