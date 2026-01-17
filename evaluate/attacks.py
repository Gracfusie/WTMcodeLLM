from typing import Literal

AttackName = Literal["none", "truncate", "reconstruct", "normalize"]


def apply_attack(text: str, attack: AttackName) -> str:
    """
    Apply a simple transformation simulating common attacks:
    - truncate: keep first half of the text
    - reconstruct: strip formatting and compress whitespace
    - normalize: lower-case and remove excessive punctuation
    - none: no change
    """
    if attack == "none":
        return text
    if attack == "truncate":
        return text[: max(1, len(text) // 2)]
    if attack == "reconstruct":
        # strip common formatting and collapse spaces
        stripped = text.replace("\n", " ").replace("\t", " ")
        return " ".join(stripped.split())
    if attack == "normalize":
        lowered = text.lower()
        # remove repeated punctuation (keep single instances)
        out = []
        prev_punct = False
        for ch in lowered:
            is_punct = ch in ".,;:!?"
            if is_punct and prev_punct:
                continue
            out.append(ch)
            prev_punct = is_punct
        return "".join(out)
    # default safety
    return text
