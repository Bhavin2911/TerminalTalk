import pytest
from TerminalTalk_v3 import get_answer, QA_DATA

def test_exact_match():
    q = "what is your name?"
    ans = get_answer(q)
    assert ans in QA_DATA[q]

def test_semantic_match():
    q = "tell me your name"
    ans = get_answer(q)
    assert "TerminalTalk" in ans

def test_unknown_question():
    q = "why do aliens love pizza?"
    ans = get_answer(q)
    assert "don't recognize" in ans.lower()
