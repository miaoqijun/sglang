"""Prompt templates for the generative-agents benchmark.

Each entry is a list of segments compatible with the
`/register_prompt_template` endpoint:

    {"kind": "fixed", "text": "..."}        # constant prefix/suffix/separator
    {"kind": "var",   "var_name": "..."}    # client-supplied substring

At request time, send `template_id` + `template_vars` to /generate; the
TokenizerManager will expand the template, and the TemplateAwareChunkCache
will key its KV chunks by (template_id, segment_index, segment_kind).

The fixed segments are extracted verbatim from agent_functions.py so the
expansion produces the same prompt the original benchmark sends.
"""

from __future__ import annotations

from typing import Any, Dict, List

# ---- poignancy_event ----
POIGNANCY_EVENT_TEMPLATE: Dict[str, Any] = {
    "template_id": "poignancy_event",
    "segments": [
        {"kind": "fixed", "text": "Here is a brief description of "},
        {"kind": "var", "var_name": "persona_name"},
        {"kind": "fixed", "text": ".\n"},
        {"kind": "var", "var_name": "persona_iss"},
        {
            "kind": "fixed",
            "text": (
                "\nOn the scale of 1 to 10, where 1 is purely mundane "
                "(e.g., brushing teeth, making bed) and 10 is extremely poignant "
                "(e.g., a break up, college acceptance), rate the likely "
                "poignancy of the following event for"
            ),
        },
        {"kind": "var", "var_name": "persona_name"},
        {"kind": "fixed", "text": ".\n\nEvent: "},
        {"kind": "var", "var_name": "event"},
        {"kind": "fixed", "text": "Rate (return a number between 1 to 10):"},
    ],
    "max_tokens": 2,
    "stop": None,
}


# ---- generate_event_triple ----
_TRIPLE_HEADER = (
    "Task: Turn the input into (subject, predicate, object).\n"
    "Input: Sam Johnson is eating breakfast.\n"
    "Output: (Dolores Murphy, eat, breakfast)\n"
    "---\n"
    "Input: Joon Park is brewing coffee.\n"
    "Output: (Joon Park, brew, coffee)\n"
    "---\n"
    "Input: Jane Cook is sleeping.\n"
    "Output: (Jane Cook, is, sleep)\n"
    "---\n"
    "Input: Michael Bernstein is writing email on a computer.\n"
    "Output: (Michael Bernstein, write, email)\n"
    "---\n"
    "Input: Percy Liang is teaching students in a classroom.\n"
    "Output: (Percy Liang, teach, students)\n"
    "---\n"
    "Input: Merrie Morris is running on a treadmill.\n"
    "Output: (Merrie Morris, run, treadmill)\n"
    "---"
)

GENERATE_EVENT_TRIPLE_TEMPLATE: Dict[str, Any] = {
    "template_id": "generate_event_triple",
    "segments": [
        {"kind": "fixed", "text": _TRIPLE_HEADER},
        {"kind": "var", "var_name": "persona_name"},
        {"kind": "fixed", "text": "is"},
        {"kind": "var", "var_name": "action"},
        {"kind": "fixed", "text": ".\n("},
        {"kind": "var", "var_name": "persona_name"},
        {"kind": "fixed", "text": ","},
    ],
    "max_tokens": 20,
    "stop": ")",
}


# ---- generate_pronunciatio ----
GENERATE_PRONUNCIATIO_TEMPLATE: Dict[str, Any] = {
    "template_id": "generate_pronunciatio",
    "segments": [
        {
            "kind": "fixed",
            "text": (
                "Convert an action description to an emoji "
                "(important: use two or less emojis).\nAction description: "
            ),
        },
        {"kind": "var", "var_name": "action"},
        {"kind": "fixed", "text": ".\nEmoji:"},
    ],
    "max_tokens": 6,
    "stop": None,
}


# ---- action_location_sector ----
_SECTOR_HEADER = (
    "Task -- choose an appropriate area  from the area options for a task at hand.\n"
    "Sam Kim lives in {Sam Kim's house} that has Sam Kim's room, bathroom, kitchen.\n"
    "Sam Kim is currently in {Sam Kim's house} that has Sam Kim's room, bathroom, kitchen.\n"
    "Area options: {Sam Kim's house, The Rose and Crown Pub, Hobbs Cafe, Oak Hill College, "
    "Johnson Park, Harvey Oak Supply Store, The Willows Market and Pharmacy}.\n"
    "* Stay in the current area if the activity can be done there. Only go out if the "
    "activity needs to take place in another place.\n"
    "* Must be one of the \"Area options,\" verbatim.\n"
    "For taking a walk, Sam Kim should go to the following area: {Johnson Park}\n"
    "---\n"
    "Jane Anderson lives in {Oak Hill College Student Dormatory} that has Jane Anderson's room.\n"
    "Jane Anderson is currently in {Oak Hill College} that has a classroom, library\n"
    "Area options: {Oak Hill College Student Dormatory, The Rose and Crown Pub, Hobbs Cafe, "
    "Oak Hill College, Johnson Park, Harvey Oak Supply Store, The Willows Market and Pharmacy}.\n"
    "* Stay in the current area if the activity can be done there. Only go out if the "
    "activity needs to take place in another place.\n"
    "* Must be one of the \"Area options,\" verbatim.\n"
    "For eating dinner, Jane Anderson should go to the following area: {Hobbs Cafe}\n"
    "---"
)

_SECTOR_RULES = (
    "* Stay in the current area if the activity can be done there. Only go out if the "
    "activity needs to take place in another place.\n"
    "* Must be one of the \"Area options,\" verbatim.\n"
)

ACTION_LOCATION_SECTOR_TEMPLATE: Dict[str, Any] = {
    "template_id": "action_location_sector",
    "segments": [
        {"kind": "fixed", "text": _SECTOR_HEADER},
        {"kind": "var", "var_name": "persona_name"},
        {"kind": "fixed", "text": " lives in "},
        {"kind": "var", "var_name": "living_sector"},
        {"kind": "fixed", "text": " that has "},
        {"kind": "var", "var_name": "living_sector_areas"},
        {"kind": "fixed", "text": ".\n"},
        {"kind": "var", "var_name": "persona_name"},
        {"kind": "fixed", "text": " is currently in "},
        {"kind": "var", "var_name": "current_sector"},
        {"kind": "fixed", "text": " that has "},
        {"kind": "var", "var_name": "current_sector_areas"},
        {"kind": "fixed", "text": ".\n"},
        {"kind": "var", "var_name": "daily_plan"},
        {"kind": "fixed", "text": ".\nArea options: "},
        {"kind": "var", "var_name": "sector_options"},
        {"kind": "fixed", "text": ".\n" + _SECTOR_RULES},
        {"kind": "var", "var_name": "persona_name"},
        {"kind": "fixed", "text": " is "},
        {"kind": "var", "var_name": "current_action"},
        {"kind": "fixed", "text": ". For "},
        {"kind": "var", "var_name": "next_action"},
        {"kind": "fixed", "text": ", "},
        {"kind": "var", "var_name": "persona_name"},
        {"kind": "fixed", "text": " should go to the following area: {"},
    ],
    "max_tokens": 10,
    "stop": "}",
}


# ---- action_location_object ----
_OBJECT_HEADER = (
    "\n"
    "Jane Anderson is in kitchen in Jane Anderson's house.\n"
    "Jane Anderson is going to Jane Anderson's house that has the following areas: "
    "{kitchen,  bedroom, bathroom}\n"
    "Stay in the current area if the activity can be done there. Never go into other "
    "people's rooms unless necessary.\n"
    "For cooking, Jane Anderson should go to the following area in Jane Anderson's house:\n"
    "Answer: {kitchen}\n"
    "---\n"
    "Tom Watson is in common room in Tom Watson's apartment.\n"
    "Tom Watson is going to Hobbs Cafe that has the following areas: {cafe}\n"
    "Stay in the current area if the activity can be done there. Never go into other "
    "people's rooms unless necessary.\n"
    "For getting coffee, Tom Watson should go to the following area in Hobbs Cafe:\n"
    "Answer: {cafe}\n"
    "---"
)

_OBJECT_RULES = (
    "* Stay in the current area if the activity can be done there.\n"
    "* NEVER go into other people's rooms unless necessary."
)

ACTION_LOCATION_OBJECT_TEMPLATE: Dict[str, Any] = {
    "template_id": "action_location_object",
    "segments": [
        {"kind": "fixed", "text": _OBJECT_HEADER},
        {"kind": "var", "var_name": "persona_name"},
        {"kind": "fixed", "text": " is going to "},
        {"kind": "var", "var_name": "target_sector"},
        {"kind": "fixed", "text": " that has the following areas: {"},
        {"kind": "var", "var_name": "target_sector_areas"},
        {"kind": "fixed", "text": "}\n" + _OBJECT_RULES},
        {"kind": "var", "var_name": "persona_name"},
        {"kind": "fixed", "text": " is "},
        {"kind": "var", "var_name": "current_action"},
        {"kind": "fixed", "text": ". For "},
        {"kind": "var", "var_name": "next_action"},
        {"kind": "fixed", "text": ", "},
        {"kind": "var", "var_name": "persona_name"},
        {"kind": "fixed", "text": "should go to the following area in "},
        {"kind": "var", "var_name": "target_sector"},
        {"kind": "fixed", "text": " (MUST pick one of {"},
        {"kind": "var", "var_name": "target_sector_areas"},
        {"kind": "fixed", "text": "}):\nAnswer: {"},
    ],
    "max_tokens": 5,
    "stop": "}",
}


ALL_TEMPLATES: List[Dict[str, Any]] = [
    POIGNANCY_EVENT_TEMPLATE,
    GENERATE_EVENT_TRIPLE_TEMPLATE,
    GENERATE_PRONUNCIATIO_TEMPLATE,
    ACTION_LOCATION_SECTOR_TEMPLATE,
    ACTION_LOCATION_OBJECT_TEMPLATE,
]
