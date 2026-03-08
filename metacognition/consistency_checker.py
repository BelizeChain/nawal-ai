"""
Consistency Checker — detects contradictions across multiple candidate responses.

Given a list of candidate responses to the same query, the ConsistencyChecker:

  1. Extracts "key claims" from each response (Phase 3: regex + NLP heuristics).
  2. Compares claims pairwise for potential contradictions.
  3. Issues a consistency score [0, 1] and a contradiction report.
  4. Selects the most-consistent candidate (highest pairwise agreement).

Phase 5 upgrade hook:
  Replace ``_extract_claims`` with an embedding-based claim extractor and
  replace ``_claims_contradict`` with a semantic entailment model test.

Usage::

    cc = ConsistencyChecker()
    candidates = [
        "The population of Belize is approximately 400,000 people.",
        "Belize has nearly 450,000 inhabitants.",
        "Belize's population is 3 million.",
    ]
    result = cc.check(candidates, context={"goal": "population of Belize"})
    print(result.score)          # 0.67 — two of three agree
    print(result.contradictions) # [(0,2,"population…"), (1,2,"population…")]
    best = cc.most_consistent(candidates)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger


# --------------------------------------------------------------------------- #
# Data structures                                                              #
# --------------------------------------------------------------------------- #

@dataclass
class ContradictionPair:
    """
    Describes a potential factual contradiction between two candidates.

    Attributes:
        idx_a, idx_b : Indices into the candidate list.
        claim_a      : The claim from candidate A.
        claim_b      : The conflicting claim from candidate B.
        reason       : Human-readable explanation.
    """
    idx_a: int
    idx_b: int
    claim_a: str
    claim_b: str
    reason: str


@dataclass
class ConsistencyResult:
    """
    Output of a ConsistencyChecker.check() call.

    Attributes:
        score          : 0.0 (fully inconsistent) → 1.0 (fully consistent).
        contradictions : List of detected ContradictionPairs.
        claim_map      : Per-candidate claim list (for inspection).
        most_supported : Index of the candidate with most agreement.
    """
    score: float
    contradictions: List[ContradictionPair] = field(default_factory=list)
    claim_map: List[List[str]] = field(default_factory=list)
    most_supported: Optional[int] = None


# --------------------------------------------------------------------------- #
# ConsistencyChecker                                                           #
# --------------------------------------------------------------------------- #

class ConsistencyChecker:
    """
    Checks a set of candidate responses for mutual consistency.

    Phase 3 implementation uses:
      - Rule-based claim extraction (numeric facts, named entities, verb-object)
      - Heuristic contradiction detection (conflicting numbers, negation)

    Phase 5 upgrade:
      - Replace ``_extract_claims()`` with embedding-based NLI/claim extractor.
      - Replace ``_claims_contradict()`` with entailment model.

    Args:
        min_claim_overlap : Minimum Jaccard overlap between two string tokens
                            to consider them "about the same topic" before
                            testing for contradiction. Default 0.2.
    """

    def __init__(self, min_claim_overlap: float = 0.2) -> None:
        self._min_overlap = min_claim_overlap

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def check(
        self,
        candidates: List[str],
        context: Optional[Dict[str, Any]] = None,
    ) -> ConsistencyResult:
        """
        Compare *candidates* for mutual consistency.

        Args:
            candidates : List of response strings.
            context    : Optional context (goal, etc.) — not used in Phase 3
                         but available for Phase 5 NLI integration.

        Returns:
            ConsistencyResult with score, contradiction list, claim map,
            and the index of the most-supported candidate.
        """
        ctx = context or {}
        if len(candidates) < 2:
            return ConsistencyResult(
                score=1.0,
                contradictions=[],
                claim_map=[[]] if candidates else [],
                most_supported=0 if candidates else None,
            )

        # Step 1: Extract claims per candidate
        claim_map = [self._extract_claims(c) for c in candidates]

        # Step 2: Pairwise contradiction detection
        contradictions: List[ContradictionPair] = []
        n = len(candidates)
        for i in range(n):
            for j in range(i + 1, n):
                for ci in claim_map[i]:
                    for cj in claim_map[j]:
                        if self._claims_overlap(ci, cj) and self._claims_contradict(ci, cj):
                            contradictions.append(ContradictionPair(
                                idx_a=i,
                                idx_b=j,
                                claim_a=ci,
                                claim_b=cj,
                                reason=self._describe_contradiction(ci, cj),
                            ))

        # Step 3: Consistency score
        total_pairs = n * (n - 1) // 2
        # Deduplicate: only count unique (i,j) pairs with contradictions
        conflicting_pairs = len({(c.idx_a, c.idx_b) for c in contradictions})
        score = round(1.0 - conflicting_pairs / max(total_pairs, 1), 4)

        # Step 4: Most supported candidate (fewest contradiction appearances)
        conflict_counts = [0] * n
        for c in contradictions:
            conflict_counts[c.idx_a] += 1
            conflict_counts[c.idx_b] += 1
        most_supported = int(conflict_counts.index(min(conflict_counts)))

        logger.info(
            f"ConsistencyChecker: candidates={n} contradictions={len(contradictions)} "
            f"score={score:.3f} most_supported_idx={most_supported}"
        )
        return ConsistencyResult(
            score=score,
            contradictions=contradictions,
            claim_map=claim_map,
            most_supported=most_supported,
        )

    def most_consistent(
        self,
        candidates: List[str],
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Return the single most-consistent candidate, or None if empty.
        """
        if not candidates:
            return None
        result = self.check(candidates, context)
        if result.most_supported is None:
            return candidates[0]
        return candidates[result.most_supported]

    # ------------------------------------------------------------------ #
    # Claim extraction (Phase 3: rule-based)                              #
    # ------------------------------------------------------------------ #

    def _extract_claims(self, text: str) -> List[str]:
        """
        Extract atomic fact-like sub-strings from *text*.

        Extracts:
          - Sentences containing numeric values.
          - Sentences with named entities (Title-Cased words ≥ 4 chars).
          - Negation patterns.
        """
        claims: List[str] = []
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            # Keep sentences with numbers, proper nouns, or negation
            has_number   = bool(re.search(r"\b\d[\d,\.]*\b", sent))
            has_np       = bool(re.search(r"\b[A-Z][a-z]{3,}\b", sent))
            has_negation = bool(re.search(r"\b(not|never|no|neither|nor|none)\b", sent, re.I))
            if has_number or has_np or has_negation:
                claims.append(sent)

        # Fallback: if nothing extracted, use full text as one claim
        return claims if claims else [text.strip()]

    # ------------------------------------------------------------------ #
    # Contradiction detection (Phase 3: heuristic)                        #
    # ------------------------------------------------------------------ #

    def _claims_overlap(self, a: str, b: str) -> bool:
        """
        Return True if claims share enough vocabulary to be 'about the same topic'.
        """
        tok_a = set(re.findall(r"\w+", a.lower())) - _STOPWORDS
        tok_b = set(re.findall(r"\w+", b.lower())) - _STOPWORDS
        if not tok_a or not tok_b:
            return False
        jaccard = len(tok_a & tok_b) / len(tok_a | tok_b)
        return jaccard >= self._min_overlap

    def _claims_contradict(self, a: str, b: str) -> bool:
        """
        Heuristic test: do the two claims assert different numeric values
        or directly negate each other?
        """
        # Compare numeric values
        nums_a = [float(n.replace(",", "")) for n in re.findall(r"\b\d[\d,\.]*\b", a)]
        nums_b = [float(n.replace(",", "")) for n in re.findall(r"\b\d[\d,\.]*\b", b)]
        if nums_a and nums_b:
            # Both have numbers — check if they differ by more than 20%
            main_a = nums_a[0]
            main_b = nums_b[0]
            if main_a > 0 and abs(main_a - main_b) / main_a > 0.2:
                return True

        # Direct negation: one has "not/never/no" around a shared key term, other does not
        neg_a = bool(re.search(r"\b(not|never|no)\b", a, re.I))
        neg_b = bool(re.search(r"\b(not|never|no)\b", b, re.I))
        if neg_a != neg_b:
            # One negates something the other affirms; rough test
            core_a = re.sub(r"\b(not|never|no|is|are|was|were|the|a|an)\b", "", a.lower())
            core_b = re.sub(r"\b(not|never|no|is|are|was|were|the|a|an)\b", "", b.lower())
            tok_a = set(core_a.split())
            tok_b = set(core_b.split())
            shared = len(tok_a & tok_b)
            if shared >= 2:
                return True

        return False

    def _describe_contradiction(self, a: str, b: str) -> str:
        nums_a = re.findall(r"\b\d[\d,\.]*\b", a)
        nums_b = re.findall(r"\b\d[\d,\.]*\b", b)
        if nums_a and nums_b and nums_a[0] != nums_b[0]:
            return f"numeric conflict: {nums_a[0]!r} vs {nums_b[0]!r}"
        return "possible logical negation conflict"


# Common English stop-words
_STOPWORDS = frozenset({
    "a", "an", "the", "is", "it", "in", "on", "at", "to", "for", "of",
    "and", "or", "but", "not", "no", "with", "by", "from", "as", "be",
    "was", "are", "were", "has", "have", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "can", "this",
    "that", "these", "those", "i", "me", "my", "we", "you", "he", "she",
    "they", "them", "their", "its", "about", "which", "who", "what",
    "when", "where", "there", "here",
})
