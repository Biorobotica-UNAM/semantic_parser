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

    Responsibility: parse a natural language command and return
    a list of action dicts with semantic roles extracted.

    Public API:
        parse(text: str) -> List[Dict[str, Any]]

    All other methods and constants are private implementation details.
    """

    # ──────────────────────────────────────────────────────────────────────────
    # Private constants — implementation details, not part of the public API.
    # Encapsulated here so that subclasses or alternative parsers can override
    # them independently without affecting global state.
    # ──────────────────────────────────────────────────────────────────────────

    # Prepositions that indicate physical destination
    _DEST_PREPS: frozenset = frozenset({"to", "into", "onto", "toward", "towards"})

    # Prepositions that indicate physical origin
    _ORIG_PREPS: frozenset = frozenset({"from", "off"})

    # Prepositions that indicate spatial location
    _LOC_PREPS: frozenset = frozenset(
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

    # Prepositions whose object is the verb's logical direct object
    # e.g. "look FOR a person", "search FOR an apple"
    _OBJECT_PREPS: frozenset = frozenset({"for", "after", "at", "about"})

    # Discourse connectives that separate clauses in a sequence
    _CLAUSE_SEP: frozenset = frozenset(
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

    # Pronouns that must be resolved to the last mentioned object
    _ANAPHORS: frozenset = frozenset(
        {"it", "them", "that", "this", "those", "these", "one"}
    )

    # spaCy POS tags that indicate a WH-word (embedded question)
    _WH_TAGS: frozenset = frozenset({"WDT", "WP", "WRB", "WP$"})

    # Verbs that introduce a perceptual query when followed by
    # a WH-clause or a nominal object ("tell me the pose of...")
    _QUERY_VERBS: frozenset = frozenset(
        {"tell", "say", "report", "describe", "inform", "show", "explain", "answer"}
    )

    # Pronouns that indicate a human recipient, not a direct object
    # e.g. "give ME the cup" → ME is recipient, not object
    _HUMAN_PRONOUNS: frozenset = frozenset({"me", "us"})

    # ──────────────────────────────────────────────────────────────────────────
    # Constructor
    # ──────────────────────────────────────────────────────────────────────────

    def __init__(self, model: str = "en_core_web_trf") -> None:
        """
        Load the spaCy model once at construction time.

        Args:
            model: Name of the spaCy model to load.
                   Defaults to the transformer-based model for best accuracy.
        """
        self._nlp = spacy.load(model)

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def parse(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse a natural language command and return semantic roles.

        Args:
            text: Natural language command in English.

        Returns:
            List of action dicts, one per detected clause.
            Each dict has keys: action, object, source,
            destination, location, actor.
        """
        context = self._make_context()
        doc = self._nlp(text)

        # DEBUG TEMPORAL
        debug_lines = []
        for token in doc:
            debug_lines.append(
                f"{token.text:15} dep={token.dep_:10} pos={token.pos_:6} head={token.head.text}"
            )

        with open("/tmp/spacy_debug.txt", "w") as f:
            f.write("\n".join(debug_lines))

        clauses = self._split_clauses(doc)

        results = []
        for clause in clauses:
            action = self._extract_action(clause, context)
            if action:
                results.append(action)

        return results

    # ──────────────────────────────────────────────────────────────────────────
    # Private — context
    # ──────────────────────────────────────────────────────────────────────────

    def _make_context(self) -> Dict[str, Optional[str]]:
        """
        Create a fresh anaphora resolution context.

        The context travels across all clauses in a single parse call,
        allowing pronouns like 'it' or 'them' to be resolved to the
        last mentioned object or location.
        """
        return {"last_object": None, "last_location": None}

    # ──────────────────────────────────────────────────────────────────────────
    # Private — clause segmentation
    # ──────────────────────────────────────────────────────────────────────────

    def _split_clauses(self, doc) -> List[List]:
        """
        Split the spaCy doc into sub-lists of tokens, one per main verb.

        Three splitting rules are applied in order:
          1. Explicit sequence connectives (then, after, finally...)
          2. Coordinating 'and' introducing a new verbal clause
          3. Verbs with dep=conj or dep=dep not yet in current clause
        """
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
                # Do NOT continue — this verb belongs to the new clause

            current.append(token)

        if current:
            clauses.append(current)

        return [c for c in clauses if any(t.pos_ == "VERB" for t in c)]

    def _is_clause_separator(self, token, current: list) -> bool:
        """True if token is an explicit sequence connective and current
        clause already contains a verb."""
        return (
            token.lower_ in self._CLAUSE_SEP
            and bool(current)
            and any(t.pos_ == "VERB" for t in current)
        )

    def _is_verbal_coordinator(self, token, current: list) -> bool:
        """True if 'and' introduces a new verbal clause (not just two NPs)."""
        if token.lower_ != "and" or token.dep_ != "cc" or token.head.pos_ != "VERB":
            return False
        if not any(t.pos_ == "VERB" for t in current):
            return False
        current_indices = {t.i for t in current}
        return token.head.i not in current_indices

    def _is_new_conjunct_verb(self, token, current: list) -> bool:
        """True if token is a conj/dep verb not yet captured in current clause."""
        if token.pos_ != "VERB" or token.dep_ not in ("conj", "dep"):
            return False
        if not any(t.pos_ == "VERB" for t in current):
            return False
        current_indices = {t.i for t in current}
        return token.i not in current_indices

    # ──────────────────────────────────────────────────────────────────────────
    # Private — action extraction
    # ──────────────────────────────────────────────────────────────────────────

    def _extract_action(self, tokens: List, context: Dict) -> Optional[Dict[str, Any]]:
        """
        Extract semantic roles from a single clause.

        Updates the anaphora context with the extracted object and location
        so that subsequent clauses can resolve pronouns correctly.
        """
        token_set = {t.i for t in tokens}
        root = self._find_root_verb(tokens)

        if root is None:
            return None

        lemma = root.lemma_.lower()
        is_query = self._is_query_verb(root, lemma, token_set)

        if is_query:
            action_str = "query"
            obj = self._extract_query_content(root, tokens, token_set)
            destination = self._find_recipient(root, token_set)
        else:
            action_str = lemma
            obj = self._find_direct_object(root, token_set, context)
            destination = self._find_destination(root, token_set)

        source = self._find_prep_arg(root, token_set, self._ORIG_PREPS)
        location = self._find_prep_arg(root, token_set, self._LOC_PREPS)

        destination = self._resolve_recipient_for_transfer(
            root, lemma, destination, token_set
        )

        self._update_context(context, obj, destination, is_query)

        return {
            "action": action_str,
            "object": obj or "unknown",
            "source": source or "unknown",
            "destination": destination or "unknown",
            "location": location or "unknown",
            "actor": "robot",
        }

    def _is_query_verb(self, verb, lemma: str, token_set: set) -> bool:
        """True if the verb introduces a perceptual query."""
        return lemma in self._QUERY_VERBS and (
            self._has_wh_subordinate(verb, token_set)
            or self._has_nominal_query(verb, token_set)
            or self._has_implicit_query(verb, lemma, token_set)
        )

    def _find_destination(self, verb, token_set: set) -> Optional[str]:
        """Find destination: first try DEST_PREPS, fallback to LOC_PREPS."""
        return self._find_prep_arg(
            verb, token_set, self._DEST_PREPS
        ) or self._find_prep_arg(verb, token_set, self._LOC_PREPS)

    def _resolve_recipient_for_transfer(
        self, verb, lemma: str, destination: Optional[str], token_set: set
    ) -> Optional[str]:
        """
        For transfer verbs (give, bring, hand...), replace the destination
        with the human recipient if one is found.
        e.g. "give it to me" → destination becomes "user"
        """
        _TRANSFER_VERBS = {"give", "bring", "hand", "pass", "offer", "show"}
        if lemma not in _TRANSFER_VERBS:
            return destination
        recipient = self._find_recipient(verb, token_set)
        if recipient and recipient != "user":
            return recipient
        return destination

    def _update_context(
        self,
        context: Dict,
        obj: Optional[str],
        destination: Optional[str],
        is_query: bool,
    ) -> None:
        """Update anaphora context with the current clause's object and location."""
        if obj and obj not in self._ANAPHORS and not is_query:
            context["last_object"] = obj
        if destination:
            context["last_location"] = destination

    # ──────────────────────────────────────────────────────────────────────────
    # Private — root verb detection
    # ──────────────────────────────────────────────────────────────────────────

    def _find_root_verb(self, tokens) -> Optional[object]:
        """
        Find the main verb of a clause by dependency priority.
        Priority: ROOT > advcl > ccomp > relcl > xcomp > dep > conj > any VERB
        """
        _PRIORITY_DEPS = ("ROOT", "advcl", "ccomp", "relcl", "xcomp", "dep", "conj")
        for dep in _PRIORITY_DEPS:
            for t in tokens:
                if t.pos_ == "VERB" and t.dep_ == dep:
                    return t
        for t in tokens:
            if t.pos_ == "VERB":
                return t
        return None

    # ──────────────────────────────────────────────────────────────────────────
    # Private — direct object extraction
    # ──────────────────────────────────────────────────────────────────────────

    def _find_direct_object(self, verb, token_set: set, context: Dict) -> Optional[str]:
        """
        Find the direct object of the verb with anaphora resolution.

        Search order:
          1. Standard dobj/obj children (skip human pronouns)
          2. pobj of object-introducing prepositions (for, at, about...)
          3. Anaphoric pobj of any preposition
        """
        # 1. Standard direct object
        for child in verb.children:
            if child.dep_ in ("dobj", "obj") and child.i in token_set:
                if child.lower_ in self._HUMAN_PRONOUNS:
                    continue
                np = self._noun_phrase(child, token_set)
                if np.lower() in self._ANAPHORS:
                    return context.get("last_object", np)
                return np

        # 2. Object of search/perception prepositions ("look for X")
        for child in verb.children:
            if (
                child.dep_ == "prep"
                and child.lower_ in self._OBJECT_PREPS
                and child.i in token_set
            ):
                for gc in child.children:
                    if gc.dep_ == "pobj" and gc.i in token_set:
                        np = self._noun_phrase(gc, token_set)
                        if np.lower() in self._ANAPHORS:
                            return context.get("last_object", np)
                        return np

        # 3. Anaphoric pronoun as pobj of any preposition
        for child in verb.children:
            if child.dep_ == "prep" and child.i in token_set:
                for gc in child.children:
                    if gc.dep_ == "pobj" and gc.i in token_set:
                        if gc.lower_ in self._ANAPHORS:
                            return context.get("last_object", gc.lower_)

        return None

    # ──────────────────────────────────────────────────────────────────────────
    # Private — recipient detection
    # ──────────────────────────────────────────────────────────────────────────

    def _find_recipient(self, verb, token_set: set) -> str:
        """
        Find a human recipient for transfer/communication verbs.

        Covers: dative, iobj, dobj with human pronoun, prep 'to'.
        Returns "user" as default when no specific recipient is found.
        """
        for child in verb.children:
            if child.dep_ in ("dative", "iobj") and child.i in token_set:
                return self._noun_phrase(child, token_set)
            if (
                child.dep_ in ("dobj", "obj")
                and child.lower_ in self._HUMAN_PRONOUNS
                and child.i in token_set
            ):
                return "user"
            if child.dep_ == "prep" and child.lower_ == "to" and child.i in token_set:
                for gc in child.children:
                    if gc.dep_ == "pobj":
                        return self._noun_phrase(gc, token_set)
        return "user"

    # ──────────────────────────────────────────────────────────────────────────
    # Private — query detection
    # ──────────────────────────────────────────────────────────────────────────

    def _has_wh_subordinate(self, verb, token_set: set) -> bool:
        """
        True if the verb has a WH-subordinate clause.
        e.g. "tell me WHAT IS the heaviest object"
        Searches the full subtree, not just direct children.
        """
        for child in verb.children:
            if child.dep_ in ("ccomp", "xcomp", "relcl") and child.i in token_set:
                for t in child.subtree:
                    if t.tag_ in self._WH_TAGS:
                        return True
        for child in verb.children:
            if child.tag_ in self._WH_TAGS and child.i in token_set:
                return True
        return False

    def _has_nominal_query(self, verb, token_set: set) -> bool:
        """
        True if the verb introduces a nominal perceptual query.
        e.g. "tell me THE POSE of the person"
        Pattern: QUERY_VERB + human pronoun + nominal object
        """
        has_human_recipient = False
        has_nominal_object = False

        for child in verb.children:
            if (
                child.dep_ in ("dobj", "obj", "dative", "iobj")
                and child.lower_ in self._HUMAN_PRONOUNS
                and child.i in token_set
            ):
                has_human_recipient = True

            if (
                child.dep_ in ("dobj", "obj")
                and child.lower_ not in self._HUMAN_PRONOUNS
                and child.pos_ in ("NOUN", "DET", "PRON")
                and child.i in token_set
            ):
                has_nominal_object = True

        return has_human_recipient and has_nominal_object

    def _has_implicit_query(self, verb, lemma: str, token_set: set) -> bool:
        """
        True if the verb implies a query without an explicit human recipient.
        Covers: "Answer the question", "Report the status", "Describe the person"
        Pattern: QUERY_VERB + nominal dobj (no human pronoun required)
        """
        _IMPLICIT_QUERY_VERBS = {"answer", "report", "describe", "explain"}
        if lemma not in _IMPLICIT_QUERY_VERBS:
            return False
        for child in verb.children:
            if (
                child.dep_ in ("dobj", "obj")
                and child.lower_ not in self._HUMAN_PRONOUNS
                and child.pos_ in ("NOUN", "DET", "PROPN")
                and child.i in token_set
            ):
                return True
        return False

    def _extract_query_content(self, verb, tokens: List, token_set: set) -> str:
        """
        Extract the full content of a query as snake_case.

        Priority:
          1. WH subordinate clause (ccomp/xcomp/relcl)
          2. Nominal direct object (not a human pronoun)
          3. Fallback: everything after the verb
        """
        # 1. WH subordinate clause
        for child in verb.children:
            if child.dep_ in ("ccomp", "xcomp", "relcl") and child.i in token_set:
                span = sorted(
                    [t for t in child.subtree if t.i in token_set and not t.is_punct],
                    key=lambda t: t.i,
                )
                return "_".join(t.text.lower() for t in span)

        # 2. Nominal direct object
        for child in verb.children:
            if (
                child.dep_ in ("dobj", "obj")
                and child.lower_ not in self._HUMAN_PRONOUNS
                and child.i in token_set
            ):
                return self._noun_phrase(child, token_set).replace(" ", "_")

        # 3. Fallback: everything after the verb excluding human pronouns
        verb_idx = next((i for i, t in enumerate(tokens) if t == verb), None)
        if verb_idx is not None:
            rest = [
                t
                for t in tokens[verb_idx + 1 :]
                if not t.is_punct and t.lower_ not in self._HUMAN_PRONOUNS
            ]
            return "_".join(t.text.lower() for t in rest)

        return "unknown"

    # ──────────────────────────────────────────────────────────────────────────
    # Private — prepositional argument extraction
    # ──────────────────────────────────────────────────────────────────────────

    def _find_prep_arg(self, verb, token_set: set, preps: frozenset) -> Optional[str]:
        """
        Find the NP argument of a prepositional modifier of the verb.

        Covers both standard prep children and explicit dative ('to person').
        """
        for child in verb.children:
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
    # Private — noun phrase reconstruction
    # ──────────────────────────────────────────────────────────────────────────

    def _noun_phrase(self, token, token_set: set) -> str:
        """
        Reconstruct the full noun phrase rooted at token.
        Includes determiners, adjectives, and prepositional modifiers
        that are within the current clause's token_set.
        """
        parts = sorted(
            [t for t in token.subtree if t.i in token_set and not t.is_punct],
            key=lambda t: t.i,
        )
        return " ".join(t.text.lower() for t in parts)
