import os
import re
from typing import List, Dict, Tuple, Optional

import torch
from datasets import load_dataset

from DPModel2.dataset import Vocab
from DPModel2.backbone import BiaffineDependencyParser

os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def simple_tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+|[^\w\s]", text)


def simple_pos_tag(tokens: List[str]) -> List[str]:
    det = {"a", "an", "the", "this", "that", "these", "those"}
    adp = {"in", "on", "at", "by", "with", "from", "to", "into", "over", "under", "near", "of", "for", "about"}
    pron = {"i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them"}
    aux = {"is", "am", "are", "was", "were", "be", "been", "being", "do", "does", "did", "have", "has", "had"}
    cconj = {"and", "or", "but"}
    sconj = {"if", "because", "while", "although", "though", "when", "since"}
    tags = []
    for t in tokens:
        tl = t.lower()
        if re.fullmatch(r"[^\w\s]", t):
            tags.append("PUNCT")
        elif tl in det:
            tags.append("DET")
        elif tl in adp:
            tags.append("ADP")
        elif tl in pron:
            tags.append("PRON")
        elif tl in aux:
            tags.append("AUX")
        elif tl in cconj:
            tags.append("CCONJ")
        elif tl in sconj:
            tags.append("SCONJ")
        elif re.fullmatch(r"\d+", t):
            tags.append("NUM")
        elif t[:1].isupper():
            tags.append("PROPN")
        else:
            tags.append("NOUN")
    return tags


def spacy_pos_tag(text: str, nlp) -> Tuple[List[str], List[str]]:
    doc = nlp(text)
    tokens = [t.text for t in doc]
    upos = [t.pos_ for t in doc]
    return tokens, upos


def build_upos_ids(upos_tags: List[str], vocab: Vocab) -> List[int]:
    str_to_native = {v: k for k, v in vocab.native_upos_id_to_str.items()}
    ids = []
    for tag in upos_tags:
        native_id = str_to_native.get(tag, None)
        if native_id is None:
            ids.append(0)
        else:
            ids.append(vocab.upos_to_id.get(native_id, 0))
    return ids


def build_noun_phrase(idx: int, tokens: List[str], heads: List[int], deprels: List[str]) -> str:
    mods = {"amod", "compound", "nummod", "nmod:poss", "flat"}
    indices = [idx]
    for j, h in enumerate(heads):
        if h == idx and deprels[j] in mods:
            indices.append(j)
    indices = sorted(set(indices))
    return " ".join([tokens[i] for i in indices])


def select_target(indices: List[int], upos: List[str], deprels: List[str]) -> int:
    noun_idx = [i for i in indices if upos[i] in {"NOUN", "PROPN"}]
    if not noun_idx:
        return indices[0] if indices else 0
    for i in noun_idx:
        if deprels[i] == "root":
            return i
    for i in noun_idx:
        if deprels[i] in {"nsubj", "nsubj:pass"}:
            return i
    for i in noun_idx:
        if deprels[i] in {"obj", "iobj"}:
            return i
    return noun_idx[0]


LOCATOR_WORDS = {
    "left", "right", "top", "bottom", "front", "back", "rear", "middle", "center", "centre",
    "upper", "lower", "uppermost", "lowermost", "nearest", "farthest", "frontmost", "backmost",
}

ORDINAL_WORDS = {
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
    "last",
}


def dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        item = item.strip().lower()
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def is_nominal(upos_tag: str) -> bool:
    return upos_tag in {"NOUN", "PROPN"}


def is_locator_word(token: str) -> bool:
    return token.lower() in LOCATOR_WORDS or token.lower() in ORDINAL_WORDS


def extract_background_phrases(tokens: List[str], upos_tags: List[str]) -> List[str]:
    lower_tokens = [t.lower() for t in tokens]
    background = []
    used = set()

    for i in range(1, len(tokens) - 3):
        if lower_tokens[i] == "from" and lower_tokens[i + 1] in LOCATOR_WORDS and lower_tokens[i + 2] == "to" and lower_tokens[i + 3] in LOCATOR_WORDS:
            background.append(f"{lower_tokens[i + 1]} to {lower_tokens[i + 3]}")
            used.update({i + 1, i + 3})

    for i in range(1, len(tokens)):
        tok = lower_tokens[i]
        if tok in ORDINAL_WORDS:
            if i + 1 < len(tokens) and is_nominal(upos_tags[i + 1]):
                background.append(f"{tok} {lower_tokens[i + 1]}")
            else:
                background.append(tok)
            used.add(i)
            continue

        if tok in LOCATOR_WORDS and i not in used:
            background.append(tok)

    return dedupe_keep_order(background)


class DPModel2Parser:
    def __init__(
        self,
        checkpoint_path: str,
        cache_dir: str,
        embed_dim: int = 300,
        hidden_dim: int = 256,
        lstm_layers: int = 3,
        ffnn_dim: int = 256,
        device: str = "cuda:0",
        use_spacy: bool = True,
        spacy_model: str = "en_core_web_sm",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_spacy = use_spacy
        self.spacy_model = spacy_model
        self._nlp = None
        if self.use_spacy:
            try:
                import spacy
                self._nlp = spacy.load(self.spacy_model)
            except Exception as e:
                print(f"[WARN] spaCy unavailable ({e}); falling back to simple POS.")
                self._nlp = None

        ds = load_dataset("universal_dependencies", "en_ewt", cache_dir=cache_dir)
        self.vocab = Vocab(ds)

        self.model = BiaffineDependencyParser(
            vocab_size=len(self.vocab.word_to_id),
            upos_size=len(self.vocab.upos_to_id),
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            lstm_layers=lstm_layers,
            ffnn_dim=ffnn_dim,
            deprel_size=len(self.vocab.deprel_to_id),
        ).to(self.device)

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

    def parse(self, sentence: str) -> Tuple[List[str], List[int], List[str], List[str]]:
        if self._nlp is not None:
            tokens_raw, upos_raw = spacy_pos_tag(sentence, self._nlp)
        else:
            tokens_raw = simple_tokenize(sentence)
            upos_raw = simple_pos_tag(tokens_raw)

        tokens = ["<ROOT>"] + tokens_raw
        upos_tags = ["PAD"] + upos_raw

        word_ids = [self.vocab.word_to_id.get(w, self.vocab.word_to_id["<UNK>"]) for w in tokens]
        upos_ids = build_upos_ids(upos_tags[1:], self.vocab)
        upos_ids = [0] + upos_ids

        word_ids = torch.tensor([word_ids], dtype=torch.long, device=self.device)
        upos_ids = torch.tensor([upos_ids], dtype=torch.long, device=self.device)
        mask = torch.ones_like(word_ids, dtype=torch.bool)

        with torch.no_grad():
            arc_scores, label_scores = self.model(word_ids, upos_ids, mask)
            pred_heads = arc_scores.argmax(dim=-1)[0].tolist()
            B, S, _, L = label_scores.shape
            head_idx = torch.tensor(pred_heads, device=self.device).view(1, S, 1, 1).expand(-1, -1, 1, L)
            label_logits = torch.gather(label_scores, dim=2, index=head_idx).squeeze(2)[0]
            pred_labels = label_logits.argmax(dim=-1).tolist()

        deprels = [self.vocab.id_to_deprel.get(i, "dep") for i in pred_labels]
        return tokens, pred_heads, deprels, upos_tags

    def extract_triplet(self, sentence: str) -> Dict[str, List[str]]:
        tokens, heads, deprels, upos_tags = self.parse(sentence)

        valid_indices = list(range(1, len(tokens)))
        target_idx = select_target(valid_indices, upos_tags, deprels)
        target_phrase = build_noun_phrase(target_idx, tokens, heads, deprels).lower()
        target_tokens = set(target_phrase.split())

        concepts = []
        for i in valid_indices:
            if i == target_idx:
                continue
            token = tokens[i].lower()
            if not is_nominal(upos_tags[i]):
                continue
            if token in target_tokens:
                continue
            if is_locator_word(token):
                continue
            concepts.append(token)

        background = extract_background_phrases(tokens, upos_tags)

        return {
            "target": target_phrase,
            "concepts": dedupe_keep_order(concepts),
            "background": background,
        }


def parse_sentence_to_triplet(
    sentence: str,
    checkpoint_path: str,
    cache_dir: str,
    embed_dim: int = 300,
    hidden_dim: int = 256,
    lstm_layers: int = 3,
    ffnn_dim: int = 256,
    device: str = "cuda:0",
) -> Dict[str, List[str]]:
    parser = DPModel2Parser(
        checkpoint_path=checkpoint_path,
        cache_dir=cache_dir,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        lstm_layers=lstm_layers,
        ffnn_dim=ffnn_dim,
        device=device,
    )
    return parser.extract_triplet(sentence)
