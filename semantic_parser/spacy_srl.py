"""
spacy_srl.py
Parser de roles semánticos basado en spaCy.
"""

from logging import debug
import spacy
from typing import Any, Dict, List, Optional

# ── Roles preposicionales ────────────────────────────────────────────────────
DEST_PREPS = {"to", "into", "onto", "toward", "towards"}
ORIG_PREPS = {"from", "off"}
LOC_PREPS = {
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

# ── Separadores de cláusula ──────────────────────────────────────────────────
CLAUSE_SEP = {
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

# Pronombres anafóricos
ANAPHORS = {"it", "them", "that", "this", "those", "these", "one"}

# Tags WH
WH_TAGS = {"WDT", "WP", "WRB", "WP$"}

# Verbos que introducen queries
QUERY_VERBS = {"tell", "say", "report", "describe", "inform", "show", "explain"}

# Pronombres que indican destinatario humano
HUMAN_PRONOUNS = {"me", "us"}


class SpacySRL:

    def __init__(self, model: str = "en_core_web_trf"):
        self.nlp = spacy.load(model)

    # ──────────────────────────────────────────────────────────────────────────
    # API pública
    # ──────────────────────────────────────────────────────────────────────────

    def parse(self, text: str) -> List[Dict[str, Any]]:
        context = {"last_object": None, "last_location": None}
        doc = self.nlp(text)

        # DEBUG TEMPORAL
        # debug_lines = []
        # for token in doc:
        #     debug_lines.append(
        #         f"{token.text:15} dep={token.dep_:10} pos={token.pos_:6} head={token.head.text}"
        #     )
        #
        # with open("/tmp/spacy_debug.txt", "w") as f:
        #     f.write("\n".join(debug_lines))

        clauses = self._split_clauses(doc)
        results = []
        for clause in clauses:
            action = self._extract_action(clause, context)
            if action:
                results.append(action)
        return results

    # ──────────────────────────────────────────────────────────────────────────
    # Segmentación en cláusulas
    # ──────────────────────────────────────────────────────────────────────────

    def _split_clauses(self, doc) -> List[List]:
        clauses, current = [], []

        for token in doc:
            if token.lower_ in CLAUSE_SEP and current:
                if any(t.pos_ == "VERB" for t in current):
                    clauses.append(current)
                    current = []
                continue

            # Cortar en "and" + verbo coordinado (conj/dep)
            if (
                token.lower_ == "and"
                and token.dep_ == "cc"
                and token.head.pos_ == "VERB"
                and any(t.pos_ == "VERB" for t in current)
            ):
                current_indices = {t.i for t in current}
                if token.head.i not in current_indices:
                    clauses.append(current)
                    current = []
                    continue

            if (
                token.pos_ == "VERB"
                and token.dep_ in ("conj", "dep")
                and any(t.pos_ == "VERB" for t in current)
            ):
                current_indices = {t.i for t in current}
                if token.i not in current_indices:
                    clauses.append(current)
                    current = []
                    # No hacer continue — el verbo pertenece a la nueva cláusula

            current.append(token)

        if current:
            clauses.append(current)

        return [c for c in clauses if any(t.pos_ == "VERB" for t in c)]

    # ──────────────────────────────────────────────────────────────────────────
    # Extracción de roles por cláusula
    # ──────────────────────────────────────────────────────────────────────────

    def _extract_action(self, tokens: List, context: Dict) -> Optional[Dict[str, Any]]:
        token_set = {t.i for t in tokens}

        root = self._find_root_verb(tokens)
        if root is None:
            return None

        lemma = root.lemma_.lower()

        is_query = lemma in QUERY_VERBS and (
            self._has_wh_subordinate(root, token_set)  # "tell me what is..."
            or self._has_nominal_query(root, token_set)  # "tell me the pose of..."
        )

        # DEBUG
        # with open("/tmp/spacy_debug.txt", "a") as f:
        #     f.write(f"\n--- _extract_action --- \n")
        #     f.write(f"  root={root.text} lemma={lemma}\n")
        #     f.write(f"  lemma in QUERY_VERBS: {lemma in QUERY_VERBS}\n")
        #     f.write(f"  is_query={is_query}\n")

        if is_query:
            action_str = "query"
            obj = self._extract_query_content(root, tokens, token_set)
            destination = self._find_recipient(root, token_set)
        else:
            action_str = lemma
            obj = self._find_direct_object(root, token_set, context)
            destination = self._find_prep_arg(
                root, token_set, DEST_PREPS
            ) or self._find_prep_arg(root, token_set, LOC_PREPS)

        source = self._find_prep_arg(root, token_set, ORIG_PREPS)
        location = self._find_prep_arg(root, token_set, LOC_PREPS)

        if not is_query and lemma in ("give", "bring", "hand", "pass", "offer", "show"):
            recipient = self._find_recipient(root, token_set)
            if recipient and recipient != "user":
                destination = recipient

        if obj and obj not in ANAPHORS and not is_query:
            context["last_object"] = obj
        if destination:
            context["last_location"] = destination

        return {
            "action": action_str,
            "object": obj or "unknown",
            "source": source or "unknown",
            "destination": destination or "unknown",
            "location": location or "unknown",
            "actor": "robot",
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers — verbo raíz
    # ──────────────────────────────────────────────────────────────────────────

    def _find_root_verb(self, tokens) -> Optional[object]:
        for dep in ("ROOT", "advcl", "ccomp", "relcl", "xcomp", "dep", "conj"):
            for t in tokens:
                if t.pos_ == "VERB" and t.dep_ == dep:
                    return t
        for t in tokens:
            if t.pos_ == "VERB":
                return t
        return None

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers — objeto directo
    # ──────────────────────────────────────────────────────────────────────────

    def _find_direct_object(self, verb, token_set: set, context: Dict) -> Optional[str]:
        for child in verb.children:
            if child.dep_ in ("dobj", "obj") and child.i in token_set:
                if child.lower_ in HUMAN_PRONOUNS:
                    continue
                np = self._noun_phrase(child, token_set)
                if np.lower() in ANAPHORS:
                    return context.get("last_object", np)
                return np

        for child in verb.children:
            if child.dep_ == "prep" and child.i in token_set:
                for gc in child.children:
                    if gc.dep_ == "pobj" and gc.i in token_set:
                        if gc.lower_ in ANAPHORS:
                            return context.get("last_object", gc.lower_)

        return None

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers — destinatario humano
    # ──────────────────────────────────────────────────────────────────────────

    def _find_recipient(self, verb, token_set: set) -> str:
        """
        Busca destinatario humano: 'me', 'us', o NP indirecto.
        Cubre: dative, iobj, dobj con pronombre humano, prep 'to'.
        """
        for child in verb.children:
            # Caso: dativo explícito
            if child.dep_ in ("dative", "iobj") and child.i in token_set:
                return self._noun_phrase(child, token_set)
            if (
                child.dep_ in ("dobj", "obj")
                and child.lower_ in HUMAN_PRONOUNS
                and child.i in token_set
            ):
                return "user"
            # Prep "to person"
            if child.dep_ == "prep" and child.lower_ == "to" and child.i in token_set:
                for gc in child.children:
                    if gc.dep_ == "pobj":
                        return self._noun_phrase(gc, token_set)
        return "user"

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers — detección de queries
    # ──────────────────────────────────────────────────────────────────────────

    def _has_wh_subordinate(self, verb, token_set: set) -> bool:
        for child in verb.children:
            if child.dep_ in ("ccomp", "xcomp", "relcl") and child.i in token_set:
                # Buscar WH en todo el subárbol (no solo hijos directos)
                for t in child.subtree:
                    if t.tag_ in WH_TAGS:
                        return True
        # Fallback: WH directo como hijo del verbo (p.ej. "tell what")
        for child in verb.children:
            if child.tag_ in WH_TAGS and child.i in token_set:
                return True
        return False

    def _has_nominal_query(self, verb, token_set: set) -> bool:
        has_human_recipient = False
        has_nominal_object = False
        debug_lines = []

        for child in verb.children:
            debug_lines.append(
                f"  child: {child.text} dep={child.dep_} pos={child.pos_} lower={child.lower_} in_set={child.i in token_set}"
            )
            if (
                child.dep_ in ("dobj", "obj", "dative", "iobj")
                and child.lower_ in HUMAN_PRONOUNS
                and child.i in token_set
            ):
                has_human_recipient = True
                debug_lines.append(f"  → has_human_recipient = True")
            if (
                child.dep_ in ("dobj", "obj")
                and child.lower_ not in HUMAN_PRONOUNS
                and child.pos_ in ("NOUN", "DET", "PRON")
                and child.i in token_set
            ):
                has_nominal_object = True
                debug_lines.append(f"  → has_nominal_object = True")

        debug_lines.append(
            f"  Result: recipient={has_human_recipient} nominal={has_nominal_object}"
        )

        # with open("/tmp/spacy_debug.txt", "a") as f:
        #     f.write("\n--- _has_nominal_query ---\n")
        #     f.write("\n".join(debug_lines) + "\n")

        return has_human_recipient and has_nominal_object

    def _extract_query_content(self, verb, tokens: List, token_set: set) -> str:
        result = "unknown"

        # Caso 1: cláusula WH subordinada
        for child in verb.children:
            if child.dep_ in ("ccomp", "xcomp", "relcl") and child.i in token_set:
                span = sorted(
                    [t for t in child.subtree if t.i in token_set and not t.is_punct],
                    key=lambda t: t.i,
                )
                result = "_".join(t.text.lower() for t in span)
                break

        # Caso 2: NP nominal directo
        if result == "unknown":
            for child in verb.children:
                if (
                    child.dep_ in ("dobj", "obj")
                    and child.lower_ not in HUMAN_PRONOUNS
                    and child.i in token_set
                ):
                    np = self._noun_phrase(child, token_set)
                    result = np.replace(" ", "_")
                    break

        # Fallback
        if result == "unknown":
            verb_idx = next((i for i, t in enumerate(tokens) if t == verb), None)
            if verb_idx is not None:
                rest = [
                    t
                    for t in tokens[verb_idx + 1 :]
                    if not t.is_punct and t.lower_ not in HUMAN_PRONOUNS
                ]
                result = "_".join(t.text.lower() for t in rest)

        # with open("/tmp/spacy_debug.txt", "a") as f:
        #     f.write(f"\n--- _extract_query_content ---\n")
        #     f.write(f"  result={result}\n")

        return result

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers — argumentos preposicionales
    # ──────────────────────────────────────────────────────────────────────────

    def _find_prep_arg(self, verb, token_set: set, preps: set) -> Optional[str]:
        for child in verb.children:
            # Caso normal: prep
            if child.dep_ == "prep" and child.lower_ in preps and child.i in token_set:
                for gc in child.children:
                    if gc.dep_ in ("pobj", "obj") and gc.i in token_set:
                        return self._noun_phrase(gc, token_set)

            if child.dep_ == "dative" and "to" in preps and child.i in token_set:
                for gc in child.children:
                    if gc.dep_ in ("pobj", "obj") and gc.i in token_set:
                        return self._noun_phrase(gc, token_set)

        return None

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers — reconstrucción de NP
    # ──────────────────────────────────────────────────────────────────────────

    def _noun_phrase(self, token, token_set: set) -> str:
        parts = sorted(
            [t for t in token.subtree if t.i in token_set and not t.is_punct],
            key=lambda t: t.i,
        )
        return " ".join(t.text.lower() for t in parts)
