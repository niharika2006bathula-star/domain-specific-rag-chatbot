from types import SimpleNamespace

from document_loader import DocumentChunk
from rag_pipeline import answer_question
from vector_store import build_vector_store


class FakeGroqClient:
    def __init__(
        self,
        answer: str,
        verification: str = '{"fully_supported": false, "supports": []}',
    ):
        self.answer = answer
        self.verification = verification
        self.calls = []

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if "you verify whether" in kwargs["messages"][0]["content"].lower():
            content = self.verification
        else:
            content = self.answer
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=content),
                )
            ]
        )


class FailingGroqClient:
    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        raise RuntimeError("Model unavailable")


def test_returning_correct_answer_structure():
    store = build_vector_store(
        [
            DocumentChunk("The office has blue windows and a white desk.", "office_policy.pdf", 1, 0),
            DocumentChunk("Employees must submit forms by Friday.", "office_policy.pdf", 2, 0),
        ]
    )

    result = answer_question(
        "What color are the office windows?",
        store,
        relevance_threshold=0.0,
        groq_client=FakeGroqClient(
            "The office windows are blue.",
            '{"fully_supported": true, "supports": ['
            '{"passage_id": 1, "quote": "The office has blue windows"}]}',
        ),
    )

    assert result["fallback"] is False
    assert "source_passages" in result
    assert result["source_passages"][0]["source"] == "office_policy.pdf"
    assert result["answer"] == "The office windows are blue."


def test_returns_fallback_when_evidence_is_insufficient():
    store = build_vector_store(
        [
            DocumentChunk("The company is based in Berlin.", "employee_handbook.pdf", 1, 0),
        ]
    )

    result = answer_question(
        "What color are the office windows?",
        store,
        relevance_threshold=10.0,
        groq_client=FakeGroqClient("This should not be used."),
    )

    assert result["fallback"] is True
    assert "could not find this information" in result["answer"].lower()
    assert result["source_passages"] == []


def test_handles_llm_api_errors_without_inventing_answers():
    store = build_vector_store(
        [
            DocumentChunk("The office has blue windows and a white desk.", "office_policy.pdf", 1, 0),
        ]
    )

    result = answer_question(
        "What color are the office windows?",
        store,
        relevance_threshold=0.0,
        groq_client=FailingGroqClient(),
    )

    assert result["fallback"] is True
    assert "language model service is unavailable" in result["answer"].lower()


class MockVectorStore:
    def __init__(self, passages):
        self.passages = passages

    def retrieve(self, question, top_k=4):
        return self.passages


def make_passage(text, source, page_number, score=0.9):
    return {
        "text": text,
        "source": source,
        "page_number": page_number,
        "score": score,
    }


def test_cites_only_retrieved_page_supporting_generated_code():
    code = "def square(number):\n    return number * number"
    store = MockVectorStore(
        [
            make_passage(f"Use this function:\n{code}", "guide.pdf", 2),
            make_passage("The program was updated in 2024.", "guide.pdf", 5),
            make_passage("The appendix lists contributors.", "guide.pdf", 7),
            make_passage("The license is included at the end.", "guide.pdf", 12),
        ]
    )
    client = FakeGroqClient(
        f"```python\n{code}\n```",
        '{"fully_supported": true, "supports": ['
        '{"passage_id": 1, "quote": "def square(number): '
        'return number * number"}]}',
    )

    result = answer_question(
        "Write the square function.",
        store,
        relevance_threshold=0.0,
        groq_client=client,
    )

    assert result["fallback"] is False
    assert result["source_passages"] == [
        {"source": "guide.pdf", "page_number": 2}
    ]


def test_deduplicates_citations_for_same_filename_and_page():
    store = MockVectorStore(
        [
            make_passage("The meeting starts at 9 AM.", "schedule.pdf", 3),
            make_passage("Please arrive by 9 AM for the meeting.", "schedule.pdf", 3),
            make_passage("The meeting starts at 9 AM.", "other.pdf", 3),
        ]
    )
    client = FakeGroqClient(
        "The meeting starts at 9 AM.",
        '{"fully_supported": true, "supports": ['
        '{"passage_id": 1, "quote": "The meeting starts at 9 AM."},'
        '{"passage_id": 2, "quote": "arrive by 9 AM for the meeting"},'
        '{"passage_id": 3, "quote": "The meeting starts at 9 AM."}]}',
    )

    result = answer_question(
        "When does the meeting start?",
        store,
        relevance_threshold=0.0,
        groq_client=client,
    )

    assert result["source_passages"] == [
        {"source": "schedule.pdf", "page_number": 3},
        {"source": "other.pdf", "page_number": 3},
    ]


def test_retains_multiple_pages_that_support_the_answer():
    store = MockVectorStore(
        [
            make_passage("Initialize total to zero before processing items.", "manual.pdf", 2),
            make_passage("Add each item's value to total.", "manual.pdf", 5),
            make_passage("The application was released in June.", "manual.pdf", 8),
        ]
    )
    client = FakeGroqClient(
        "Initialize total to zero, then add each item's value.",
        '{"fully_supported": true, "supports": ['
        '{"passage_id": 1, "quote": "Initialize total to zero"},'
        '{"passage_id": 2, "quote": "Add each item\'s value to total"}]}',
    )

    result = answer_question(
        "How should I calculate the total?",
        store,
        relevance_threshold=0.0,
        groq_client=client,
    )

    assert result["source_passages"] == [
        {"source": "manual.pdf", "page_number": 2},
        {"source": "manual.pdf", "page_number": 5},
    ]


def test_uses_fallback_when_no_passage_supports_the_answer():
    store = MockVectorStore(
        [make_passage("The manual describes the login process.", "manual.pdf", 1)]
    )
    client = FakeGroqClient(
        "The password must contain twelve characters.",
        '{"fully_supported": false, "supports": []}',
    )

    result = answer_question(
        "What is the password length?",
        store,
        relevance_threshold=0.0,
        groq_client=client,
    )

    assert result["fallback"] is True
    assert "could not find this information" in result["answer"].lower()
    assert result["source_passages"] == []


def test_omits_citations_when_verification_quote_is_not_in_passage():
    store = MockVectorStore(
        [make_passage("The manual describes the login process.", "manual.pdf", 1)]
    )
    client = FakeGroqClient(
        "The password must contain twelve characters.",
        '{"fully_supported": true, "supports": ['
        '{"passage_id": 1, "quote": "Password length is twelve characters"}]}',
    )

    result = answer_question(
        "What is the password length?",
        store,
        relevance_threshold=0.0,
        groq_client=client,
    )

    assert result["fallback"] is True
    assert result["source_passages"] == []
