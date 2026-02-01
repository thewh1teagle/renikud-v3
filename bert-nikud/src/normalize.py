import regex as re
import unicodedata
from deduplicate import deduplicate

SHVA = "\u05b0"
VOCAL_SHVA = "\u05bd"


def normalize_vocal_shva(text: str) -> str:
    """
    When a letter has both shva and meteg (vocal shva) in its diacritics,
    remove the shva and keep only the meteg as vocal shva.
    """

    def callback(match):
        letter = match.group(1)
        diac = list(match.group(2))
        if SHVA in diac and VOCAL_SHVA in diac:
            diac = [d for d in diac if d != SHVA]
        return letter + "".join(diac)

    return re.sub(r"(\p{L})(\p{M}+)", callback, text)

def sort_diacritics(text: str):
    def sort_diacritics_callback(match):
        letter = match.group(1)
        diac = match.group(2)
        diac_sorted = "".join(sorted(diac)) if diac else ""
        return letter + diac_sorted

    return re.sub(r"(\p{L})(\p{M}+)", sort_diacritics_callback, text)


def clean_dagesh(text: str):
    dagesh = "\u05bc"
    can_have_dagesh = "בכפו"

    def clean_dagesh_callback(match):
        letter = match.group(1)
        diac = list(match.group(2) or "")

        if letter not in can_have_dagesh:
            diac = [d for d in diac if d != dagesh]

        return letter + "".join(diac)

    return re.sub(r"(\p{L})(\p{M}+)", clean_dagesh_callback, text)


def normalize(text: str):
    text = unicodedata.normalize("NFD", text)  # Decomposite
    text = sort_diacritics(text)
    text = clean_dagesh(text)
    text = deduplicate(text)
    return text
