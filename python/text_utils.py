import re


class TextProcessor:
    def __init__(
        self,
        dictionary_path: str = "frequency_dictionary_en_82_765.txt",
        max_edit_distance: int = 2,
    ):
        self.dictionary_path = dictionary_path
        self.max_edit_distance = max_edit_distance
        self._sym = None

    def _load_symspell(self) -> None:
        from symspellpy import SymSpell, Verbosity  # noqa: F401
        self._sym = SymSpell(
            max_dictionary_edit_distance=self.max_edit_distance, prefix_length=7
        )
        self._sym.load_dictionary(self.dictionary_path, term_index=0, count_index=1)

    def normalize(self, text: str) -> str:
        t = re.sub(r"\s+", " ", text.strip())
        letters = [c for c in t if c.isalpha()]
        if letters and sum(c.isupper() for c in letters) / len(letters) > 0.8:
            t = t.lower()
        return t

    def autocorrect(self, text: str) -> str:
        if self._sym is None:
            self._load_symspell()
        t = self.normalize(text)
        result = self._sym.lookup_compound(t, max_edit_distance=2, ignore_non_words=True)
        corrected = result[0].term if result else t
        if corrected:
            corrected = corrected[0].upper() + corrected[1:]
        if corrected and corrected[-1] not in ".!?":
            corrected += "."
        return corrected

    def process(self, text: str) -> str:
        return self.autocorrect(text)
