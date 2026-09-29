"""A bounded assumption-based truth maintenance system.

Labels are antichains of minimal sets of assumptions that imply a node. They
are symbolic: disabling an assumption changes the current context, not labels.
Rules are conjunctions; multiple rules represent alternative explanations.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable
from itertools import product

Environment = frozenset[str]


class EnvironmentOverflow(RuntimeError):
    """A node's exact minimal label exceeds its configured environment budget."""


class ATMS:
    """Maintain exact minimal supports, with an explicit per-node label budget.

    Updates are atomic when a label exceeds ``max_environments``. The limit
    bounds stored labels, not runtime: exact support enumeration can still be
    exponential. Names are nonempty strings. Nogoods contain assumption names.
    """

    def __init__(self, max_environments: int = 10_000) -> None:
        if (
            isinstance(max_environments, bool)
            or not isinstance(max_environments, int)
            or max_environments < 1
        ):
            raise ValueError("max_environments must be a positive integer")
        self.max_environments = max_environments
        self._assumptions: dict[str, bool] = {}
        self._rules: dict[str, list[tuple[str, ...]]] = defaultdict(list)
        self._children: dict[str, set[str]] = defaultdict(set)
        self._labels: dict[str, set[Environment]] = {}
        self._nogoods: set[Environment] = set()
        self._revision = 0

    @property
    def nodes(self) -> frozenset[str]:
        """All known assumption, premise, and conclusion names."""
        return frozenset(self._labels)

    @property
    def revision(self) -> int:
        """Number of effective, successfully committed model/context updates."""
        return self._revision

    @staticmethod
    def _name(name: str) -> str:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("node names must be nonempty strings")
        return name

    def _descendants(self, roots: Iterable[str]) -> set[str]:
        found = set(roots)
        queue = deque(found)
        while queue:
            for child in self._children.get(queue.popleft(), ()):
                if child not in found:
                    found.add(child)
                    queue.append(child)
        return found

    def _consistent(self, environment: Environment) -> bool:
        return not any(nogood <= environment for nogood in self._nogoods)

    @staticmethod
    def _minimal(environments: Iterable[Environment]) -> set[Environment]:
        result: set[Environment] = set()
        for environment in sorted(set(environments), key=lambda e: (len(e), tuple(sorted(e)))):
            if not any(existing <= environment for existing in result):
                result.add(environment)
        return result

    def _recompute(self, affected: set[str]) -> set[str]:
        """Rebuild an affected dependency region to its least fixed point.

        Clearing the whole region removes unsupported cycles; external premises
        retain their labels. Labels are staged until the complete region fits.
        """
        previous = {node: self._labels[node] for node in affected}
        labels = dict(self._labels)
        labels.update({node: set() for node in affected})
        queue = deque(sorted(affected))
        queued = set(affected)
        while queue:
            node = queue.popleft()
            queued.remove(node)
            candidates = set(labels[node])
            if node in self._assumptions:
                candidates.add(frozenset({node}))
            for premises in self._rules.get(node, ()):
                if not premises:
                    candidates.add(frozenset())
                elif all(labels[premise] for premise in premises):
                    for combination in product(*(labels[premise] for premise in premises)):
                        environment = frozenset().union(*combination)
                        if self._consistent(environment):
                            # Reduce as we go so dominated products don't accumulate.
                            if not any(existing <= environment for existing in candidates):
                                candidates = {
                                    existing
                                    for existing in candidates
                                    if not environment < existing
                                }
                                candidates.add(environment)
            updated = self._minimal(e for e in candidates if self._consistent(e))
            if len(updated) > self.max_environments:
                raise EnvironmentOverflow(
                    f"node {node!r} needs {len(updated)} minimal environments; "
                    f"limit is {self.max_environments}"
                )
            if updated != labels[node]:
                labels[node] = updated
                for child in self._children.get(node, ()):
                    if child in affected and child not in queued:
                        queue.append(child)
                        queued.add(child)
        self._labels = labels
        return {node for node in affected if previous[node] != labels[node]}

    def add_assumption(self, name: str, active: bool = True) -> set[str]:
        """Add an independent assumption; existing assumptions update context."""
        self._name(name)
        if not isinstance(active, bool):
            raise ValueError("active must be a boolean")
        if name in self._assumptions:
            return self.set_active(name, active)
        was_known = name in self._labels
        self._labels.setdefault(name, set())
        self._assumptions[name] = active
        try:
            changed = self._recompute(self._descendants({name}))
        except EnvironmentOverflow:
            del self._assumptions[name]
            if not was_known:
                del self._labels[name]
            raise
        self._revision += 1
        return changed

    def set_active(self, name: str, active: bool) -> set[str]:
        """Change context and return nodes whose supported truth value changed.

        Symbolic labels are retained exactly; no rule enumeration is required.
        """
        if name not in self._assumptions:
            raise KeyError(f"unknown assumption: {name!r}")
        if not isinstance(active, bool):
            raise ValueError("active must be a boolean")
        if self._assumptions[name] == active:
            return set()
        affected = self._descendants({name})
        before = {node: self.is_supported(node) for node in affected}
        self._assumptions[name] = active
        self._revision += 1
        return {node for node in affected if before[node] != self.is_supported(node)}

    def add_rule(self, conclusion: str, premises: tuple[str, ...]) -> set[str]:
        """Add a conjunctive justification; empty premises establish a fact."""
        self._name(conclusion)
        if isinstance(premises, str):
            raise ValueError("premises must be a sequence of node names")
        premises = tuple(dict.fromkeys(self._name(p) for p in premises))
        if premises in self._rules.get(conclusion, ()):
            return set()
        old_nodes = self.nodes
        for node in (conclusion, *premises):
            self._labels.setdefault(node, set())
        self._rules[conclusion].append(premises)
        for premise in premises:
            self._children[premise].add(conclusion)
        try:
            changed = self._recompute(self._descendants({conclusion}))
        except EnvironmentOverflow:
            self._rules[conclusion].remove(premises)
            for premise in premises:
                if not any(premise in rule for rule in self._rules[conclusion]):
                    self._children[premise].discard(conclusion)
            for node in self.nodes - old_nodes:
                del self._labels[node]
            raise
        self._revision += 1
        return changed

    def add_nogood(self, assumptions: Iterable[str]) -> set[str]:
        """Declare an impossible assumption combination and remove its supports.

        An empty nogood is an unconditional contradiction and removes all labels.
        A node can still hold in a consistent subset of the active context even
        when the full active context contains a nogood.
        """
        if isinstance(assumptions, str):
            raise ValueError("nogood must be an iterable of assumption names")
        nogood = frozenset(self._name(name) for name in assumptions)
        unknown = nogood - self._assumptions.keys()
        if unknown:
            raise KeyError(f"unknown assumptions: {sorted(unknown)}")
        if any(existing <= nogood for existing in self._nogoods):
            return set()
        old_nogoods = self._nogoods.copy()
        self._nogoods = self._minimal((*self._nogoods, nogood))
        roots = {node for node, label in self._labels.items() if any(nogood <= e for e in label)}
        try:
            changed = self._recompute(self._descendants(roots))
        except EnvironmentOverflow:
            self._nogoods = old_nogoods
            raise
        self._revision += 1
        return changed

    def supports(self, node: str) -> tuple[Environment, ...]:
        """Return deterministic minimal symbolic supports; unknown nodes are empty."""
        return tuple(sorted(self._labels.get(node, ()), key=lambda e: (len(e), tuple(sorted(e)))))

    def is_supported(self, node: str) -> bool:
        """Whether any consistent support is entirely active in this context."""
        return any(
            all(self._assumptions[name] for name in environment)
            for environment in self.supports(node)
        )

    def explain(self, node: str) -> dict:
        """Return JSON-safe support evidence and justification structure."""
        supports = self.supports(node)
        return {
            "node": node,
            "supported": self.is_supported(node),
            "supports": [sorted(environment) for environment in supports],
            "active_supports": [
                sorted(e) for e in supports if all(self._assumptions[a] for a in e)
            ],
            "inactive_assumptions": sorted(
                {a for e in supports for a in e if not self._assumptions[a]}
            ),
            "justifications": [list(rule) for rule in self._rules.get(node, ())],
            "revision": self._revision,
        }
