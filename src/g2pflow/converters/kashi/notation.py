"""Spelling normalization for kana readings before the Kana phoneme stage."""

_YOTSUGANA_FOLD = str.maketrans({
    "ぢ": "じ", "づ": "ず",
    "ヂ": "ジ", "ヅ": "ズ",
})

_VOWEL_CHARS = {
    "a": "あかさたなはまやらわがざだばぱぁゃゎアカサタナハマヤラワガザダバパァャヮ",
    "i": "いきしちにひみりぎじぢびぴぃイキシチニヒミリギジヂビピィ",
    "u": "うくすつぬふむゆるぐずづぶぷぅゅウクスツヌフムユルグズヅブプゥュ",
    "e": "えけせてねへめれげぜでべぺぇエケセテネヘメレゲゼデベペェ",
    "o": "おこそとのほもよろをごぞどぼぽぉょオコソトノホモヨロヲゴゾドボポォョ",
}
_CHAR_TO_VOWEL = {
    ch: vowel for vowel, chars in _VOWEL_CHARS.items() for ch in chars
}
_ELONGATION_MAP = {
    "a": {"あ", "ア", "ー"},
    "i": {"い", "イ", "ー"},
    "u": {"う", "ウ", "ー"},
    "e": {"い", "え", "イ", "エ", "ー"},
    "o": {"う", "お", "ウ", "オ", "ー"},
}


def fold_notation(text: str) -> str:
    """Unify long-vowel spellings (きょう -> きょー) and merge ゛゛, 1:1 in length."""
    if not text:
        return text
    text = text.translate(_YOTSUGANA_FOLD)
    result: list[str] = []
    for char in text:
        if result:
            prev_vowel = _CHAR_TO_VOWEL.get(result[-1])
            if prev_vowel and char in _ELONGATION_MAP[prev_vowel]:
                result.append("ー")
                continue
        result.append(char)
    return "".join(result)
