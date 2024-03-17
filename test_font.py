import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Noto Sans CJK JP'  # Adjust as needed for specific script support
plt.rcParams['axes.unicode_minus'] = False  # To ensure the minus sign appears correctly

from matplotlib.font_manager import FontProperties

noto_font = FontProperties(family='Noto Sans CJK JP', size=12)  # Example for Japanese
plt.text(0.5, 0.5, 'Some text こんにちは 你好 안녕하세요 Привет', fontproperties=noto_font)
