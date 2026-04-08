# ROS2 Semantic Parser Node — Conceptual Dependency

This repository contains a **ROS2 service node** that translates natural language instructions into a **formal semantic representation** based on **Schank's Conceptual Dependency (CD) theory**.

This project is developed as part of ongoing research at the  
**BioRobotics Laboratory, Facultad de Ingeniería, UNAM**  
https://biorobotics.fi-p.unam.mx/

The node is designed to serve as a **semantic front-end** for service robots, separating meaning representation from planning, reasoning, and execution.

---

## Overview

Natural language understanding in robotics benefits from a clear separation between:

- **Semantic interpretation**
- **Reasoning and planning**
- **Action execution**

This project focuses *exclusively* on semantic interpretation.

The node receives a natural language command and produces a **canonical, minimal Conceptual Dependency representation** that can be consumed by downstream symbolic or hybrid systems such as CLIPS.

---

## Conceptual Dependency (CD)

Conceptual Dependency (CD), introduced by Roger Schank, is a theory of meaning representation that models semantics using a **closed set of primitive actions and state changes**, independent of linguistic surface form.

This parser generates CD representations using the following fixed set of primitives:

| Primitive | Meaning | Example command |
|-----------|---------|-----------------|
| `PTRANS` | Physical movement of robot or object | *"Go to the kitchen"* |
| `ATRANS` | Transfer of possession or location | *"Bring me the cup"*, *"Put it on the table"* |
| `GRASP` | Physically take an object | *"Grasp the spoon"*, *"Pick it up"* |
| `ATTEND` | Perceive or search for something | *"Find a snack"*, *"Look for the person"* |
| `MTRANS` | Transfer information (say, report) | *"Tell me the name"* |
| `MBUILD` | Construct or infer internal knowledge | *"Count the objects on the shelf"* |
| `SPEAK` | Produce speech without specific content | *"Say hello"* |
| `SOCIAL` | Social interaction with a person | *"Meet Angel at the shelf"* |
| `ACCOMPANY` | Escort or guide a person | *"Escort the person to the living room"* |
| `WAIT` | Pause until a condition is met | *"Wait for the person"* |
| `QUERY` | Perceptual question (expands to ATTEND + MTRANS) | *"Tell me what is the heaviest object"* |
| `DO` | Fallback for unmapped verbs | — |

### Core Principles

- Literal, least-inferential interpretation
- No invention of actors, objects, or locations
- No goal, intention, or plan inference
- Physical primitives preferred over cognitive ones
- Output is semantic representation only — not an action plan

---

## System Architecture

The parser is implemented as a **ROS2 service node** written in Python.

### Data Flow

```
Natural language command
        ↓
  SpacySRL.parse()          spacy_srl.py
  Dependency tree parsing
  Anaphora resolution
        ↓
  map_to_primitive()        verb_to_cd.json
  Verb → CD primitive
        ↓
  expand_cd_actions()
  QUERY → [ATTEND, MTRANS]
        ↓
  JSON list of CD primitives → response.response
```

### Key Features

- **Multi-clause segmentation** — handles sequences like *"go to X then find Y and grasp it"*
- **Anaphora resolution** — resolves *"it"*, *"them"*, *"they"* to the last mentioned object
- **Perceptual query expansion** — *"Tell me what is..."* → `ATTEND` + `MTRANS`
- **Nominal query detection** — *"Tell me the pose of..."* treated as perceptual query
- **Complex NP extraction** — handles descriptors like *"the person raising their left arm in the kitchen"*
- **Dative resolution** — *"give it to me"* → `to: "user"`

---

## ROS2 Interface

### Service Name
```
/semantic_parser
```

### Service Type
```
semantic_parser_interfaces/srv/Parser
```

### Request
```
string command
```

### Response
```
string response   # JSON array of CD primitives
```

### Example

```bash
ros2 service call /semantic_parser/semantic_parser \
  semantic_parser_interfaces/srv/Parser \
  "{command: Go to the bathroom then find a snack and grasp it and put it on the bookshelf}"
```

```json
[
  {"primitive": "PTRANS", "actor": "robot", "object": "unknown", "from": "unknown", "to": "the bathroom"},
  {"primitive": "ATTEND", "actor": "robot", "object": "a snack"},
  {"primitive": "GRASP",  "actor": "robot", "object": "a snack"},
  {"primitive": "ATRANS", "actor": "robot", "object": "a snack", "from": "unknown", "to": "the bookshelf"}
]
```

---

## Node Details

- Node name: `semantic_parser_node`
- Language: Python 3.12
- NLP backend: spaCy `en_core_web_trf` (transformer-based dependency parser)
- Verb mapping: `verb_to_cd.json` (extensible dictionary)

The node performs **no semantic validation** and **no error correction** by design.

---

## NLP Backend

The semantic parser uses **spaCy** with the `en_core_web_trf` transformer model (~110M parameters, RoBERTa-based) for dependency parsing.

This replaces a previous LLM-based approach (Qwen via Ollama) with a deterministic, rule-based pipeline that is:

- **~50x faster** (~50 ms vs 2–5 s per command)
- **~10x lighter** (~500 MB RAM vs ~4–6 GB)
- **Fully offline** — no internet connection required
- **Deterministic** — same input always produces the same output

The pipeline consists of three stages:

1. **SpacySRL** (`spacy_srl.py`) — dependency tree parsing, clause segmentation, anaphora resolution
2. **Verb mapper** (`verb_to_cd.json`) — maps verb lemmas to CD primitives
3. **Expander** (`semantic_node.py`) — expands `QUERY` into `[ATTEND, MTRANS]`, normalizes recipients

---

## Requirements

- Ubuntu 22.04 or 24.04
- ROS2 Jazzy
- Python 3.10+
- spaCy

---

## Installation

```bash
pip install spacy
python -m spacy download en_core_web_trf
```

Then build the ROS2 package:

```bash
cd ~/ros2_ws
colcon build --packages-select semantic_parser
source install/setup.bash
```

---

## Running

```bash
ros2 launch semantic_parser semantic_parser.launch.xml
```

---

## Extending the Verb Dictionary

To add support for new verbs, edit `semantic_parser/verb_to_cd.json`:

```json
{
  "hand":    "ATRANS",
  "carry":   "ATRANS",
  "your_new_verb": "PRIMITIVE"
}
```

Valid primitives: `PTRANS`, `ATRANS`, `GRASP`, `ATTEND`, `MTRANS`, `MBUILD`, `SPEAK`, `SOCIAL`, `ACCOMPANY`, `WAIT`, `QUERY`, `DO`.
