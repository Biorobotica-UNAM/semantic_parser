"""
spacy_srl.py

Semantic Role Labeler based on spaCy's dependency parser.

Produces a list of action dicts with the following schema:
    {
        "action":      str,  # verb lemma or "query"
        "object":      str,  # direct object or query content
        "source":      str,  # prepositional origin
        "destination": str,  # prepositional destination
        "location":    str,  # prepositional location
        "actor":       str,  # always "robot"
    }

This schema is consumed by map_to_primitive() in semantic_node.py.

BioRobotics Laboratory, Facultad de Ingenieria, UNAM
"""

import spacy
from typing import Any, Dict, List, Optional


class SpacySRL:
    """
    Semantic Role Labeler based on spaCy's dependency parser.
    """

    _DEST_PREPS = frozenset({"to", "into", "onto", "toward", "towards"})
    _ORIG_PREPS = frozenset({"from", "off"})

    _LOC_PREPS = frozenset(
        {
            "on",
            "in",
            "at",
            "near",
            "beside",
            "inside",
            "next",
            "above",
            "below",
            "behind",
            "front",
            "between",
            "over",
            "under",
        }
    )

    _OBJECT_PREPS = frozenset({"for", "after", "at", "about"})

    _CLAUSE_SEP = frozenset(
        {
            "then",
            "after",
            "afterwards",
            "subsequently",
            "next",
            "finally",
            "also",
            "first",
            "second",
            "third",
        }
    )

    _ANAPHORS = frozenset({"it", "them", "that", "this", "those", "these", "one"})
    _WH_TAGS = frozenset({"WDT", "WP", "WRB", "WP$"})

    _QUERY_VERBS = frozenset(
        {"tell", "say", "report", "describe", "inform", "show", "explain", "answer"}
    )

    _HUMAN_PRONOUNS = frozenset({"me", "us"})

    def __init__(self, model: str = "en_core_web_trf") -> None:
        self._nlp = spacy.load(model)

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def parse(self, text: str) -> List[Dict[str, Any]]:
        context = self._make_context()
        doc = self._nlp(text)

        clauses = self._split_clauses(doc)

        results = []
        for clause in clauses:
            action = self._extract_action(clause, context)
            if action:
                results.append(action)

        return results

    # ──────────────────────────────────────────────────────────────────────────
    # Context
    # ──────────────────────────────────────────────────────────────────────────

    def _make_context(self) -> Dict[str, Optional[str]]:
        return {"last_object": None, "last_location": None}

    def _update_context(
        self,
        context: Dict,
        obj: Optional[str],
        location: Optional[str],
        is_query: bool,
    ) -> None:
        if obj and obj not in self._ANAPHORS and not is_query:
            context["last_object"] = obj
        if location:
            context["last_location"] = location

    # ──────────────────────────────────────────────────────────────────────────
    # Clause segmentation
    # ──────────────────────────────────────────────────────────────────────────

    def _split_clauses(self, doc) -> List[List]:
        clauses, current = [], []

        for token in doc:
            if self._is_clause_separator(token, current):
                clauses.append(current)
                current = []
                continue

            if self._is_verbal_coordinator(token, current):
                clauses.append(current)
                current = []
                continue

            if self._is_new_conjunct_verb(token, current):
                clauses.append(current)
                current = []

            current.append(token)

        if current:
            clauses.append(current)

        return [c for c in clauses if any(t.pos_ == "VERB" for t in c)]

    def _is_clause_separator(self, token, current: list) -> bool:
        return (
            token.lower_ in self._CLAUSE_SEP
            and bool(current)
            and any(t.pos_ == "VERB" for t in current)
        )

    def _is_verbal_coordinator(self, token, current: list) -> bool:
        if token.lower_ != "and" or token.dep_ != "cc" or token.head.pos_ != "VERB":
            return False
        if not any(t.pos_ == "VERB" for t in current):
            return False
        return token.head.i not in {t.i for t in current}

    def _is_new_conjunct_verb(self, token, current: list) -> bool:
        if token.pos_ != "VERB" or token.dep_ not in ("conj", "dep"):
            return False
        if not any(t.pos_ == "VERB" for t in current):
            return False
        return token.i not in {t.i for t in current}

    # ──────────────────────────────────────────────────────────────────────────
    # Action extraction
    # ──────────────────────────────────────────────────────────────────────────

    def _extract_action(self, tokens: List, context: Dict) -> Optional[Dict[str, Any]]:
        token_set = {t.i for t in tokens}
        root = self._find_root_verb(tokens)

        if root is None:
            return None

        lemma = root.lemma_.lower()

        is_query = self._is_query_verb(root, lemma)

        obj = None
        destination = None
        location = None

        if is_query:
            action_str = "query"

            obj = self._extract_query_content(root, token_set)
            destination = self._find_recipient(root, token_set)
            location = self._find_prep_arg(root, token_set, self._LOC_PREPS)

        else:
            action_str = lemma
            obj = self._find_direct_object(root, token_set, context)

            dest_token = self._find_destination_token(root, token_set)
            location = self._find_prep_arg(root, token_set, self._LOC_PREPS)

            if dest_token:
                destination = self._noun_phrase_base(dest_token, token_set)

                np_location = self._extract_location_from_np(dest_token, token_set)
                if np_location:
                    location = np_location

            destination = self._resolve_recipient_for_transfer(
                root, lemma, destination, token_set
            )

        source = self._find_prep_arg(root, token_set, self._ORIG_PREPS)

        self._update_context(context, obj, location, is_query)

        return {
            "action": action_str,
            "object": obj or "unknown",
            "source": source or "unknown",
            "destination": destination or "unknown",
            "location": location or "unknown",
            "actor": "robot",
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Core helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _find_root_verb(self, tokens) -> Optional[object]:
        for dep in ("ROOT", "advcl", "ccomp", "relcl", "xcomp", "dep", "conj"):
            for t in tokens:
                if t.pos_ == "VERB" and t.dep_ == dep:
                    return t
        return next((t for t in tokens if t.pos_ == "VERB"), None)

    def _find_destination_token(self, verb, token_set: set):
        for child in verb.children:
            if child.dep_ == "prep" and child.lower_ in self._DEST_PREPS:
                for gc in child.children:
                    if gc.dep_ in ("pobj", "obj") and gc.i in token_set:
                        return gc
        return None

    def _extract_location_from_np(self, token, token_set: set) -> Optional[str]:
        for t in token.subtree:
            if t.dep_ == "prep" and t.lower_ in self._LOC_PREPS:
                for gc in t.children:
                    if gc.dep_ in ("pobj", "obj") and gc.i in token_set:
                        return self._noun_phrase(gc, token_set)
        return None

    def _resolve_recipient_for_transfer(
        self, verb, lemma: str, destination: Optional[str], token_set: set
    ) -> Optional[str]:
        _TRANSFER_VERBS = {"give", "bring", "hand", "pass", "offer", "show"}
        if lemma not in _TRANSFER_VERBS:
            return destination
        recipient = self._find_recipient(verb, token_set)
        if recipient and recipient != "user":
            return recipient
        return destination

    # ──────────────────────────────────────────────────────────────────────────
    # NLP helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _find_direct_object(self, verb, token_set: set, context: Dict) -> Optional[str]:
        for child in verb.children:
            if child.dep_ in ("dobj", "obj") and child.i in token_set:
                if child.lower_ in self._HUMAN_PRONOUNS:
                    continue
                np = self._noun_phrase(child, token_set)
                if np.lower() in self._ANAPHORS:
                    return context.get("last_object", np)
                return np
        return None

    def _find_recipient(self, verb, token_set: set) -> str:
        for child in verb.children:
            if child.dep_ in ("dative", "iobj") and child.i in token_set:
                return self._noun_phrase(child, token_set)
            if (
                child.dep_ in ("dobj", "obj")
                and child.lower_ in self._HUMAN_PRONOUNS
                and child.i in token_set
            ):
                return "user"
            if child.dep_ == "prep" and child.lower_ == "to":
                for gc in child.children:
                    if gc.dep_ == "pobj":
                        return self._noun_phrase(gc, token_set)
        return "user"

    def _is_query_verb(self, verb, lemma: str) -> bool:
        """
        True si el verbo es un verbo de query Y su subárbol contiene un
        WH-token (what, where, who, how...).

        FIX: se eliminó el filtro por token_set para que la búsqueda incluya
        cláusulas subordinadas (ccomp) como "Tell me what is the largest food",
        donde "what" es raíz de la ccomp y puede quedar fuera del token_set
        de la cláusula principal tras el split.
        """
        if lemma not in self._QUERY_VERBS:
            return False
        return any(t.tag_ in self._WH_TAGS for t in verb.subtree)

    def _extract_query_content(self, verb, token_set: set) -> str:
        """
        Captura el contenido semántico de la query como cadena snake_case.

        Recorre todo verb.subtree (no solo el token_set de la cláusula) para
        incluir cláusulas subordinadas tipo ccomp. Excluye:
          - signos de puntuación
          - el propio verbo raíz  ("tell", "say", etc.)
          - pronombres humanos    ("me", "us") que indican receptor, no contenido

        Ejemplo:
            "Tell me what is the largest food on the bookshelf"
            → "what_is_the_largest_food_on_the_bookshelf"
        """
        subtree_tokens = sorted(
            [
                t
                for t in verb.subtree
                if not t.is_punct
                and t.lower_ not in self._HUMAN_PRONOUNS
                and t.i != verb.i  # excluye el verbo raíz
            ],
            key=lambda t: t.i,
        )
        return "_".join(t.text.lower() for t in subtree_tokens)

    def _find_prep_arg(self, verb, token_set: set, preps: frozenset) -> Optional[str]:
        for child in verb.children:
            if child.dep_ == "prep" and child.lower_ in preps:
                for gc in child.children:
                    if gc.dep_ in ("pobj", "obj") and gc.i in token_set:
                        return self._noun_phrase(gc, token_set)
        return None

    def _noun_phrase(self, token, token_set: set) -> str:
        parts = sorted(
            [t for t in token.subtree if t.i in token_set and not t.is_punct],
            key=lambda t: t.i,
        )
        return " ".join(t.text.lower() for t in parts)

    def _noun_phrase_base(self, token, token_set: set) -> str:
        parts = []
        for t in sorted(token.subtree, key=lambda x: x.i):
            if t.i not in token_set or t.is_punct:
                continue
            if t.dep_ == "prep" and t.lower_ in self._LOC_PREPS:
                continue
            if any(
                a.dep_ == "prep" and a.lower_ in self._LOC_PREPS for a in t.ancestors
            ):
                continue
            parts.append(t.text.lower())
        return " ".join(parts)
