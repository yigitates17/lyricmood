"""Compact output shapes reused across tools. Kept small on purpose: every
field here costs tokens in every tool result the LLM reads."""

from pydantic import BaseModel

from lyricmood.core.models import Track


class TrackOut(BaseModel):
    id: str
    title: str
    artists: str
    album: str | None = None
    year: str | None = None
    duration: str
    isrc: str | None = None

    @classmethod
    def of(cls, track: Track) -> "TrackOut":
        return cls(
            id=track.spotify_id,
            title=track.title,
            artists=", ".join(track.artists),
            album=track.album,
            year=(track.release_date or "")[:4] or None,
            duration=format_ms(track.duration_ms),
            isrc=track.isrc,
        )


TRACK_ARG = (
    "The song: a Spotify track ID/URL/URI (best: use `id` from search_track) or free text "
    'like "Carnival of Rust Poets of the Fall".'
)


def format_ms(ms: int) -> str:
    seconds = round(ms / 1000)
    return f"{seconds // 60}:{seconds % 60:02d}"
