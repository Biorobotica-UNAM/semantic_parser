"""
primitives/schema.py

Enriched Conceptual Dependency (CD) primitive schema for service robots.

Design notes
────────────
• Every primitive is a plain dataclass — no business logic, only data.
  Business logic lives in the pipeline handlers.
• Dataclasses with slots=True are faster and prevent accidental attribute
  creation (Clean Code: no surprise side-effects).
• Each primitive carries a `subtype` field that lets the CLIPS planner
  dispatch on finer-grained intent without parsing strings.
• The `to_dict()` method is the only serialization surface — JSON for ROS2.

Primitive inventory
────────────────────
  PTRANS       — robot moves itself (physical transfer of self)
  ATRANS       — robot transfers an object (give / place / bring)
  GRASP        — robot picks up an object
  ACCOMPANY    — robot moves together with a person (escort / follow / guide)
  ATTEND       — robot perceives something (find object or person)
  QUERY        — robot answers a perceptual or factual question
  MTRANS       — robot communicates a result to someone
  SPEAK        — robot says a fixed or self-referential statement
  SOCIAL       — robot performs a social act (greet / introduce)
  WAIT         — robot pauses until a condition is met

BioRobotics Laboratory, Facultad de Ingeniería, UNAM
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

# ──────────────────────────────────────────────────────────────
# Base
# ──────────────────────────────────────────────────────────────


@dataclass
class CDPrimitive:
    """Base class for all CD primitives."""

    primitive: str
    actor: str = "robot"

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


# ──────────────────────────────────────────────────────────────
# Person descriptor — reused by ATTEND, ACCOMPANY, QUERY
# ──────────────────────────────────────────────────────────────


@dataclass
class PersonDescriptor:
    """
    Structured description of a person in a command.

    Examples
    ────────
    "the sitting person"         → PersonDescriptor(pose="sitting")
    "the person wearing a blue shirt"
                                 → PersonDescriptor(clothing_color="blue",
                                                    clothing_item="shirt")
    "the person raising their right arm"
                                 → PersonDescriptor(gesture="raising",
                                                    side="right",
                                                    body_part="arm")
    "Angel"                      → PersonDescriptor(name="Angel")
    """

    name: Optional[str] = None
    pose: Optional[str] = None  # sitting, standing, lying, waving
    gesture: Optional[str] = None  # raising, pointing
    side: Optional[str] = None  # left, right
    body_part: Optional[str] = None  # arm, hand
    clothing_color: Optional[str] = None
    clothing_item: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

    @property
    def is_named(self) -> bool:
        return self.name is not None

    @property
    def is_generic(self) -> bool:
        return not self.is_named


# ──────────────────────────────────────────────────────────────
# PTRANS — robot navigates to a location
# ──────────────────────────────────────────────────────────────


@dataclass
class PTRANSPrimitive(CDPrimitive):
    """
    Physical transfer of the robot itself.

    Subtypes
    ────────
    navigate   — go to a place
    open/close — act on a door/container at a location
    """

    primitive: str = "PTRANS"
    subtype: str = "navigate"  # navigate | open | close
    destination: str = "unknown"
    source: str = "unknown"


# ──────────────────────────────────────────────────────────────
# ATRANS — robot transfers an object
# ──────────────────────────────────────────────────────────────


@dataclass
class ATRANSPrimitive(CDPrimitive):
    """
    Abstract transfer of an object from one location/owner to another.

    Subtypes
    ────────
    bring      — carry to destination (implicit source = current loc)
    place      — put down at destination
    give       — hand over to a person
    """

    primitive: str = "ATRANS"
    subtype: str = "bring"  # bring | place | give
    object: str = "unknown"
    object_category: Optional[str] = None  # fruit, drink, toy, …
    source: str = "unknown"
    destination: str = "unknown"


# ──────────────────────────────────────────────────────────────
# GRASP — robot picks up an object
# ──────────────────────────────────────────────────────────────


@dataclass
class GRASPPrimitive(CDPrimitive):
    """
    Robot grasps/retrieves an object.

    Subtypes
    ────────
    pick_up    — standard grasp from a surface
    fetch      — navigate-then-grasp (compound)
    """

    primitive: str = "GRASP"
    subtype: str = "pick_up"  # pick_up | fetch
    object: str = "unknown"
    object_category: Optional[str] = None
    location: str = "unknown"


# ──────────────────────────────────────────────────────────────
# ACCOMPANY — robot moves with a person
# ──────────────────────────────────────────────────────────────


@dataclass
class ACCOMPANYPrimitive(CDPrimitive):
    """
    Robot accompanies a person from one place to another.

    Subtypes
    ────────
    escort     — robot leads the person to destination
    follow     — robot follows a person (destination may be unknown)
    guide      — synonym for escort in GPSR context
    lead       — explicit "lead" verb
    """

    primitive: str = "ACCOMPANY"
    subtype: str = "escort"  # escort | follow | guide | lead
    person: PersonDescriptor = field(default_factory=PersonDescriptor)
    source: str = "unknown"
    destination: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["person"] = self.person.to_dict()
        return d


# ──────────────────────────────────────────────────────────────
# ATTEND — robot perceives something in the environment
# ──────────────────────────────────────────────────────────────


@dataclass
class ATTENDPrimitive(CDPrimitive):
    """
    Robot directs attention to find/identify something.

    Subtypes
    ────────
    find_object    — locate a physical object
    find_person    — locate a specific person or person type
    count_objects  — count objects of a category
    count_persons  — count persons matching a descriptor
    identify       — identify a property (pose, name, gesture…)
    """

    primitive: str = "ATTEND"
    subtype: str = "find_object"
    object: str = "unknown"
    object_category: Optional[str] = None
    person: Optional[PersonDescriptor] = None
    location: str = "unknown"
    # For queries: what property to perceive
    query_attribute: Optional[str] = None  # pose, name, color, count…
    query_filter: Optional[str] = None  # e.g. "yellow_jacket"

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        if self.person:
            d["person"] = self.person.to_dict()
        return d


# ──────────────────────────────────────────────────────────────
# QUERY — robot must perceive then report
# (expands to ATTEND + MTRANS in the pipeline)
# ──────────────────────────────────────────────────────────────


@dataclass
class QUERYPrimitive(CDPrimitive):
    """
    Intermediate primitive: robot must perceive and then communicate.

    This primitive is internal to the pipeline and is always expanded
    into [ATTEND, MTRANS] by QueryExpansionHandler before output.

    Query types
    ────────────
    count      — "how many X in Y"
    property   — "what is the [superlative] X on Y"
    attribute  — "what is the [pose/name/gesture] of X"
    identity   — "who is in Y"
    """

    primitive: str = "QUERY"
    query_type: str = "count"  # count | property | attribute | identity
    target: str = "unknown"  # normalized noun (e.g. "people")
    target_category: Optional[str] = None
    qualifier: Optional[str] = None  # superlative: "heaviest", "smallest"…
    query_attribute: Optional[str] = None  # pose, name, gesture, color…
    person_filter: Optional[PersonDescriptor] = None  # wearing blue shirt…
    location: str = "unknown"
    recipient: str = "user"

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        if self.person_filter:
            d["person_filter"] = self.person_filter.to_dict()
        return d


# ──────────────────────────────────────────────────────────────
# MTRANS — robot communicates information
# ──────────────────────────────────────────────────────────────


@dataclass
class MTRANSPrimitive(CDPrimitive):
    """
    Mental transfer: robot reports a perceived result to a person.

    Always paired with a preceding ATTEND in the expanded plan.
    """

    primitive: str = "MTRANS"
    content: str = "unknown"
    recipient: str = "user"
    location: str = "unknown"


# ──────────────────────────────────────────────────────────────
# SPEAK — robot says something from internal state
# ──────────────────────────────────────────────────────────────


@dataclass
class SPEAKPrimitive(CDPrimitive):
    """
    Robot produces speech from internal knowledge (not from perception).

    Subtypes
    ────────
    affiliation   — say team name / affiliation / country
    time_info     — say time / date / day
    self_info     — say something about yourself
    answer_quiz   — answer a question / quiz
    """

    primitive: str = "SPEAK"
    subtype: str = "affiliation"  # affiliation | time_info | self_info | answer_quiz
    content: str = "unknown"
    recipient: str = "user"
    location: str = "unknown"


# ──────────────────────────────────────────────────────────────
# SOCIAL — robot performs a social gesture
# ──────────────────────────────────────────────────────────────


@dataclass
class SOCIALPrimitive(CDPrimitive):
    """
    Robot performs a social act directed at a person.

    Subtypes
    ────────
    greet         — say hello / wave
    introduce     — introduce itself
    salute        — formal salute gesture
    meet          — locate and approach a named person
    """

    primitive: str = "SOCIAL"
    subtype: str = "greet"  # greet | introduce | salute | meet
    person: PersonDescriptor = field(default_factory=PersonDescriptor)
    location: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["person"] = self.person.to_dict()
        return d


# ──────────────────────────────────────────────────────────────
# WAIT
# ──────────────────────────────────────────────────────────────


@dataclass
class WAITPrimitive(CDPrimitive):
    """Robot waits until a condition is met."""

    primitive: str = "WAIT"
    condition: str = "unknown"
