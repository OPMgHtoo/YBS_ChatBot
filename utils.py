def detect_lang(text: str) -> str:
    """Detect whether text is Burmese ('mm') or not ('en')."""
    for ch in text:
        if "\u1000" <= ch <= "\u109F":  # Myanmar Unicode block
            return "mm"
    return "en"
