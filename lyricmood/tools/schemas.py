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
        seconds = round(track.duration_ms / 1000)
        return cls(
            id=track.spotify_id,
            title=track.title,
            artists=", ".join(track.artists),
            album=track.album,
            year=(track.release_date or "")[:4] or None,
            duration=f"{seconds // 60}:{seconds % 60:02d}",
            isrc=track.isrc,
        )
