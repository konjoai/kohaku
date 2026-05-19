"""Single-shot episodic role binding: who / what / when / where → composite HV.

Each episode is a bundle of role-value bound pairs::

    composite = binarize(bind(R_who, who_hv) + bind(R_what, what_hv) + ...)

Role HVs are fixed, deterministically seeded random vectors so any two
EpisodeStore instances over the same ``dims`` share the same role space.

Retrieval from a partial cue::

    store.query_episode(what=action_hv)  # matches episodes containing that action

Unbinding: bind is its own inverse for bipolar ±1, so
``bind(composite, R_who) ≈ who_hv``  (noisy). The store keeps the originals
for exact reconstruction via :meth:`unbind_role`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from kohaku._pure import DIMS, EpisodicMemory, HyperVector
from kohaku._query import query

_ROLE_SEEDS: Dict[str, int] = {
    "who": 0xB001_0001,
    "what": 0xB001_0002,
    "when": 0xB001_0003,
    "where": 0xB001_0004,
}

_ROLES = tuple(_ROLE_SEEDS.keys())


@dataclass
class EpisodeRoles:
    """Original role HVs retained for exact unbinding."""

    who: Optional[HyperVector] = None
    what: Optional[HyperVector] = None
    when: Optional[HyperVector] = None
    where: Optional[HyperVector] = None


@dataclass(frozen=True)
class EpisodeResult:
    entry_id: int
    label: str
    similarity: float
    roles: EpisodeRoles


class EpisodeStore:
    """HDC episodic store with who / what / when / where role binding.

    Episodes are composite HVs: bundle of bind(role_hv, value_hv) for each
    provided role. Any subset of roles can be used as a retrieval cue.
    """

    def __init__(self, dims: int = DIMS, capacity: int = 1000) -> None:
        if dims <= 0:
            raise ValueError("dims must be > 0")
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self._dims = dims
        self._memory: EpisodicMemory = EpisodicMemory(capacity)
        self._role_hvs: Dict[str, HyperVector] = {
            role: HyperVector.random(dims, seed=seed)
            for role, seed in _ROLE_SEEDS.items()
        }
        self._stored_roles: Dict[int, EpisodeRoles] = {}

    def store_episode(
        self,
        label: str,
        *,
        who: Optional[HyperVector] = None,
        what: Optional[HyperVector] = None,
        when: Optional[HyperVector] = None,
        where: Optional[HyperVector] = None,
    ) -> int:
        """Bind provided roles into a composite HV and store it.

        Returns the entry_id. Raises ``ValueError`` if no roles are provided.
        """
        provided = {"who": who, "what": what, "when": when, "where": where}
        bound = [
            self._role_hvs[r].bind(hv)
            for r, hv in provided.items()
            if hv is not None
        ]
        if not bound:
            raise ValueError("At least one role HV must be provided")
        composite = HyperVector.bundle_all(bound)
        entry_id = self._memory.store(composite, composite, label)
        self._stored_roles[entry_id] = EpisodeRoles(
            who=who, what=what, when=when, where=where
        )
        return entry_id

    def query_episode(
        self,
        *,
        who: Optional[HyperVector] = None,
        what: Optional[HyperVector] = None,
        when: Optional[HyperVector] = None,
        where: Optional[HyperVector] = None,
        top_k: int = 5,
    ) -> List[EpisodeResult]:
        """Retrieve episodes matching the provided partial cue.

        Any subset of roles may be supplied; the query composite is built from
        the provided roles only. Raises ``ValueError`` if no roles are given.
        """
        provided = {"who": who, "what": what, "when": when, "where": where}
        bound = [
            self._role_hvs[r].bind(hv)
            for r, hv in provided.items()
            if hv is not None
        ]
        if not bound:
            raise ValueError("At least one role HV must be provided for query")
        query_hv = HyperVector.bundle_all(bound)
        raw = query(self._memory, query_hv, top_k)
        return [
            EpisodeResult(
                entry_id=r.entry_id,
                label=r.label,
                similarity=r.similarity,
                roles=self._stored_roles.get(r.entry_id, EpisodeRoles()),
            )
            for r in raw
        ]

    def unbind_role(self, entry_id: int, role: str) -> Optional[HyperVector]:
        """Return the original HV for a role from a stored episode.

        Returns ``None`` if the entry_id is unknown or the role was not
        provided when the episode was stored.  Raises ``ValueError`` for an
        unrecognised role name.
        """
        if role not in _ROLES:
            raise ValueError(f"Unknown role {role!r}; valid roles: {_ROLES}")
        stored = self._stored_roles.get(entry_id)
        if stored is None:
            return None
        return getattr(stored, role)

    def __len__(self) -> int:
        return len(self._memory)
