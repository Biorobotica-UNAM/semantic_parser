# ROS2 Semantic Parser Node — Conceptual Dependency

This repository contains a **ROS2 service node** that translates natural language instructions into a **formal semantic representation** based on **Schank’s Conceptual Dependency (CD) theory**.

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

The node receives a natural language command and produces a **canonical, minimal Conceptual Dependency representation** that can be consumed by downstream symbolic or hybrid systems.

---

## Conceptual Dependency (CD)

Conceptual Dependency (CD), introduced by Roger Schank, is a theory of meaning representation that models semantics using a **closed set of primitive actions and state changes**, independent of linguistic surface form.

This parser generates CD representations using the following fixed set of primitives:

PTRANS
ATRANS
GRASP
ATTEND
MOVE
PROPEL
SPEAK
MTRANS
MBUILD
INGEST
STATE
DO



### Core Principles

- Literal, least-inferential interpretation
- No invention of actors, objects, or locations
- No goal, intention, or plan inference
- Physical primitives preferred over cognitive ones
- Output is semantic representation only — not an action plan

The complete generation rules are enforced via a strict system prompt used by the language model.

---

## System Architecture

The parser is implemented as a **ROS2 service node** written in Python.

### Data Flow

1. A ROS2 client sends a natural language string
2. The node calls a locally hosted LLM via **Ollama**
3. The model generates a Conceptual Dependency representation
4. The result is returned as a string response

---

## ROS2 Interface

### Service Name
/semantic_parser


### Service Type
semantic_parser_interfaces/srv/Parser


### Request
string command

### Response
string response


---

## Node Details

- Node name: `semantic_parser_node`
- Language: Python
- HTTP client: `httpx`
- LLM backend: Ollama (local inference)
- Optional model warm-up on startup

The node performs **no semantic validation** and **no error correction** by design.

---

## Language Model

The backend model is a **Qwen-2.5 14B Instruct**-class model running locally via Ollama.

A custom model named:

cd-qwen-big


is created using a `Modelfile` that defines a highly restrictive system prompt enforcing Conceptual Dependency generation rules.

The model:

- Outputs only CD primitives
- Produces no explanations or meta-comments
- Uses canonical verb-to-primitive mappings
- Generates machine-readable semantic structures

---

## Requirements

- Ubuntu 22.04 or 24.04
- ROS2 (Jazzy)
- Python 3.10+
- Ollama
- ≥16 GB RAM recommended

---

## Installation

A helper script is provided to install Ollama, Python dependencies, and create the required LLM model.

```bash
./install_ollama.sh

