"""Tests for general/config.py constants and prompt_texts.py."""
import pytest


# ---------------------------------------------------------------------------
# 15. test_config_constants_integrity
# ---------------------------------------------------------------------------

def test_config_constants_integrity():
    from general.config import (
        DATA_FILES_DIR,
        DEFAULT_STT_MODEL,
        FILE_MEMORY_DIR,
        MAIN_DIR,
        OLLAMA_MODELS,
        OPENAI_MODELS,
        RECURSION_LIMIT,
        SUPPORTED_FILETYPES,
        SUPPORTED_STT_MODELS,
    )

    # 1. Non-empty strings.
    assert isinstance(FILE_MEMORY_DIR, str) and FILE_MEMORY_DIR
    assert isinstance(DATA_FILES_DIR, str) and DATA_FILES_DIR

    # 2. Both dirs are subpaths of MAIN_DIR.
    assert FILE_MEMORY_DIR.startswith(MAIN_DIR)
    assert DATA_FILES_DIR.startswith(MAIN_DIR)

    # 3. SUPPORTED_FILETYPES is tuple of strings starting with ".".
    assert isinstance(SUPPORTED_FILETYPES, tuple)
    for ft in SUPPORTED_FILETYPES:
        assert isinstance(ft, str)
        assert ft.startswith(".")

    # 4. Model lists non-empty and non-overlapping.
    assert len(OPENAI_MODELS) > 0
    assert len(OLLAMA_MODELS) > 0
    assert not set(OPENAI_MODELS) & set(OLLAMA_MODELS)

    # 5. RECURSION_LIMIT > 0.
    assert RECURSION_LIMIT > 0

    # 6. DEFAULT_STT_MODEL is in SUPPORTED_STT_MODELS.
    assert DEFAULT_STT_MODEL in SUPPORTED_STT_MODELS


# ---------------------------------------------------------------------------
# 16. test_prompt_texts_pipeline
# ---------------------------------------------------------------------------

def test_prompt_texts_pipeline():
    from langchain_core.prompts import ChatPromptTemplate

    from prompt_texts import main_system_prompt, summary_prompt
    from tools import DEFINED_TOOLS

    # 1. summary_prompt("eng") → ChatPromptTemplate; contains "English".
    prompt = summary_prompt("eng")
    assert isinstance(prompt, ChatPromptTemplate)
    assert "English" in str(prompt)

    # 2. summary_prompt("fr") → contains "French".
    prompt = summary_prompt("fr")
    assert "French" in str(prompt)

    # 3. summary_prompt("xyz") (unsupported) → falls back to "English".
    prompt = summary_prompt("xyz")
    assert isinstance(prompt, ChatPromptTemplate)
    assert "English" in str(prompt)

    # 4. main_system_prompt() → non-empty string containing every tool name.
    system = main_system_prompt()
    assert isinstance(system, str) and system
    for t in DEFINED_TOOLS:
        assert t.name in system

    # 5. Each tool name appears exactly once.
    for t in DEFINED_TOOLS:
        assert system.count(t.name) == 1
