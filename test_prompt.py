from src.prompt import gen_prompt_repeats
from src.constants import LANG2NAME, LANG_BANK

def test_prompts():
    # Test English
    en_words = ['one', 'two', 'three', 'four', 'five', 'six']
    print('=== English Test ===')
    print(gen_prompt_repeats(en_words, 'en', 1))
    print()

    # Test French
    fr_words = ['chat', 'chien', 'oiseau', 'poisson', 'lapin', 'souris']
    print('=== French Test ===')
    print(gen_prompt_repeats(fr_words, 'fr', 2))
    print()

    # Test Chinese (to check spacing)
    zh_words = ['好', '猫', '狗', '鱼', '兔', '鼠']
    print('=== Chinese Test ===')
    print(gen_prompt_repeats(zh_words, 'zh', 1))

if __name__ == "__main__":
    test_prompts() 