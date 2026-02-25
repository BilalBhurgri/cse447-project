import random

languages = [
    "en_us",
    "cmn_hans_cn",
    "hi_in",
    "es_419",
    "fr_fr",
    "ar_eg",
    "bn_in",
    "pt_br",
    "ru_ru",
    "ur_pk",
    "id_id",
    "de_de",
    "ja_jp",
    "sw_ke",
    "mr_in",
]

corpus = {
    "en_us": [
        "Happy New Year everyone.",
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world.",
        "Seattle is known for rain and coffee.",
        "This is a character level language model test."
    ],
    "cmn_hans_cn": [
        "新年快乐，祝你万事如意。",
        "人工智能正在改变世界。",
        "学习语言模型非常有趣。",
        "今天天气很好，我们去公园吧。",
        "数据科学需要大量实验。"
    ],
    "hi_in": [
        "नया साल मुबारक हो।",
        "कृत्रिम बुद्धिमत्ता दुनिया बदल रही है।",
        "भारत विविध संस्कृतियों का देश है।",
        "आज मौसम बहुत अच्छा है।",
        "हमें रोज़ अभ्यास करना चाहिए।"
    ],
    "es_419": [
        "Feliz año nuevo a todos.",
        "La inteligencia artificial está cambiando el mundo.",
        "Hoy es un buen día para aprender.",
        "La ciencia de datos es fascinante.",
        "Este es un modelo de lenguaje."
    ],
    "fr_fr": [
        "Bonne année à tous.",
        "L'intelligence artificielle change le monde.",
        "La science des données est passionnante.",
        "Paris est une ville magnifique.",
        "Nous aimons apprendre chaque jour."
    ],
    "ar_eg": [
        "سنة جديدة سعيدة للجميع.",
        "الذكاء الاصطناعي يغير العالم.",
        "اليوم الطقس جميل.",
        "التعلم مهم في حياتنا.",
        "هذا اختبار لنموذج اللغة."
    ],
    "bn_in": [
        "শুভ নববর্ষ সবাইকে।",
        "কৃত্রিম বুদ্ধিমত্তা বিশ্ব বদলে দিচ্ছে।",
        "আজ আবহাওয়া খুব সুন্দর।",
        "আমরা প্রতিদিন শিখি।",
        "ডেটা সায়েন্স খুব গুরুত্বপূর্ণ।"
    ],
    "pt_br": [
        "Feliz ano novo para todos.",
        "A inteligência artificial está mudando o mundo.",
        "Hoje é um ótimo dia.",
        "A ciência de dados é interessante.",
        "Estamos aprendendo algo novo."
    ],
    "ru_ru": [
        "С Новым годом всех.",
        "Искусственный интеллект меняет мир.",
        "Сегодня хорошая погода.",
        "Мы учимся каждый день.",
        "Это тест языковой модели."
    ],
    "ur_pk": [
        "نیا سال مبارک ہو۔",
        "مصنوعی ذہانت دنیا بدل رہی ہے۔",
        "آج موسم بہت اچھا ہے۔",
        "ہم روز سیکھتے ہیں۔",
        "یہ زبان کا ماڈل ہے۔"
    ],
    "id_id": [
        "Selamat tahun baru semuanya.",
        "Kecerdasan buatan mengubah dunia.",
        "Hari ini cuacanya cerah.",
        "Kami belajar setiap hari.",
        "Ini adalah model bahasa."
    ],
    "de_de": [
        "Frohes neues Jahr zusammen.",
        "Künstliche Intelligenz verändert die Welt.",
        "Heute ist ein schöner Tag.",
        "Wir lernen jeden Tag.",
        "Das ist ein Sprachmodell."
    ],
    "ja_jp": [
        "明けましておめでとうございます。",
        "人工知能は世界を変えています。",
        "今日は良い天気です。",
        "毎日勉強しています。",
        "これは言語モデルです。"
    ],
    "sw_ke": [
        "Heri ya mwaka mpya kwa wote.",
        "Akili bandia inabadilisha dunia.",
        "Leo hali ya hewa ni nzuri.",
        "Tunajifunza kila siku.",
        "Huu ni mfano wa lugha."
    ],
    "mr_in": [
        "नवीन वर्षाच्या शुभेच्छा.",
        "कृत्रिम बुद्धिमत्ता जग बदलत आहे.",
        "आज हवामान छान आहे.",
        "आपण दररोज शिकतो.",
        "हा भाषा मॉडेल आहे."
    ],
}

def generate_prefixes(sentence, n=10):
    """
    Generate prefixes where:
    - next char is NOT whitespace
    - previous char is NOT whitespace
    - ensures fully correct next-character alignment
    """

    valid_positions = [
        i for i in range(1, len(sentence))
        if not sentence[i].isspace()
        and not sentence[i-1].isspace()
    ]

    if len(valid_positions) < n:
        n = len(valid_positions)

    positions = random.sample(valid_positions, n)
    positions.sort()

    pairs = []
    for p in positions:
        prefix = sentence[:p]
        next_char = sentence[p]
        pairs.append((prefix, next_char))

    return pairs


input_lines = []
answer_lines = []

for lang in languages:
    sentences = corpus[lang]

    for sentence in sentences:
        pairs = generate_prefixes(sentence, 10)

        for prefix, next_char in pairs:
            input_lines.append(prefix)
            answer_lines.append(next_char)

    # language separator
    input_lines.append("")
    answer_lines.append("")


with open("input.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(input_lines))

with open("answer.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(answer_lines))

print("Done. Generated clean input.txt and answer.txt.")
