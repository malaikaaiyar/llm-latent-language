import unittest
from transformers import AutoTokenizer

class TestTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
        self.word_dict = {
            'word1': {'en': 'translation1', 'zh': 'translation2', 'fr': 'translation3'},
            'word2': {'en': 'translation4', 'zh': 'translation5', 'fr': 'translation6'}
        }

    def test_word_dict_in_vocab(self):
        for word, translations in self.word_dict.items():
            for lang, translation in translations.items():
                token = self.tokenizer.encode(translation, add_special_tokens=False)
                self.assertIn(token, self.tokenizer.get_vocab(), f"Missing {lang} token for {translation}")

if __name__ == '__main__':
    unittest.main()