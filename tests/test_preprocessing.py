import pytest
from src.data.preprocessing import clean_text, preprocess, batch_preprocess, remove_stopwords, tokenize


class TestCleanText:
    def test_lowercases_text(self):
        assert clean_text("HELLO World") == "hello world"

    def test_removes_urls(self):
        result = clean_text("visit https://example.com for more info")
        assert "http" not in result
        assert "example" not in result

    def test_removes_punctuation(self):
        result = clean_text("Hello, World!")
        assert "," not in result
        assert "!" not in result

    def test_removes_numbers(self):
        result = clean_text("I have 3 cats and 2 dogs")
        assert "3" not in result
        assert "2" not in result

    def test_strips_extra_whitespace(self):
        result = clean_text("  hello   world  ")
        assert result == "hello world"

    def test_removes_html_tags(self):
        result = clean_text("<p>Hello <b>World</b></p>")
        assert "<" not in result
        assert ">" not in result


class TestTokenize:
    def test_splits_on_spaces(self):
        assert tokenize("hello world") == ["hello", "world"]

    def test_empty_string_returns_empty_list(self):
        assert tokenize("") == []


class TestRemoveStopwords:
    def test_removes_common_stopwords(self):
        tokens = ["i", "love", "this", "movie", "very", "much"]
        result = remove_stopwords(tokens)
        assert "i" not in result
        assert "this" not in result
        assert "love" in result
        assert "movie" in result

    def test_empty_list_returns_empty(self):
        assert remove_stopwords([]) == []


class TestPreprocess:
    def test_returns_string(self):
        assert isinstance(preprocess("This is a great movie!"), str)

    def test_removes_noise(self):
        result = preprocess("The 123 <b>best</b> movie! https://imdb.com")
        assert "<b>" not in result
        assert "123" not in result
        assert "http" not in result

    def test_content_words_preserved(self):
        result = preprocess("This movie is absolutely fantastic and brilliant")
        assert "fantastic" in result
        assert "brilliant" in result

    def test_stem_flag(self):
        result_stemmed = preprocess("running runners run", stem=True)
        result_unstemmed = preprocess("running runners run", stem=False)
        assert len(result_stemmed.split()) <= len(result_unstemmed.split())


class TestBatchPreprocess:
    def test_returns_list(self):
        result = batch_preprocess(["hello world", "foo bar"])
        assert isinstance(result, list)
        assert len(result) == 2

    def test_empty_list(self):
        assert batch_preprocess([]) == []
