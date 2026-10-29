import pytest

from lyricmood.core.lyrics import stats, truncate
from lyricmood.core.text import clean_title, normalize, same_name


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Nemo - Remastered 2004", "Nemo"),
        ("Song - 2011 Remaster", "Song"),
        ("Song (feat. X & Y)", "Song"),
        ("Love Me (Radio Edit)", "Love Me"),
        ("Track [Bonus Track]", "Track"),
        # Different recordings with possibly different lyrics are kept.
        ("Nemo - Orchestral Version", "Nemo - Orchestral Version"),
        ("Carnival of Rust (Live)", "Carnival of Rust (Live)"),
    ],
)
def test_clean_title(title, expected):
    assert clean_title(title) == expected


def test_normalize_keeps_finnish_letters():
    assert normalize("Vapaus Käteen JÄÄ!") == "vapaus käteen jää"


def test_same_name():
    assert same_name("Carnival Of Rust", "carnival of rust")
    assert same_name("Poets of the Fall", "Poets Of The Fall")
    assert not same_name("Carnival of Rust (Live)", "Carnival of Rust")
    assert not same_name("Nightwish", "Within Temptation")


LYRICS = "one two\nthree four\n\nchorus line\nchorus line\n\nfive six\nchorus line"


def test_truncate_short_text_untouched():
    assert truncate(LYRICS, 1000) == (LYRICS, False)


def test_truncate_cuts_at_stanza_boundary():
    text, truncated = truncate(LYRICS, 45)
    assert truncated
    assert text == "one two\nthree four\n\nchorus line\nchorus line"


def test_truncate_single_huge_stanza_falls_back_to_lines():
    text, truncated = truncate("aaaa\nbbbb\ncccc\ndddd", 10)
    assert (text, truncated) == ("aaaa\nbbbb", True)


def test_stats():
    s = stats(LYRICS)
    assert s.lines == 6
    assert s.stanzas == 3
    assert s.words == 12
    assert s.repeated_lines == ["chorus line"]
