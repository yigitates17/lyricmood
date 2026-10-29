"""Domain models shared by every API client and tool."""

from pydantic import BaseModel


class Track(BaseModel):
    """A track as resolved on Spotify, the canonical identity for caching."""

    spotify_id: str
    title: str
    artists: list[str]
    artist_ids: list[str] = []
    album: str | None = None
    release_date: str | None = None
    duration_ms: int
    isrc: str | None = None
    explicit: bool = False

    @property
    def primary_artist(self) -> str:
        return self.artists[0] if self.artists else ""

    @property
    def url(self) -> str:
        return f"https://open.spotify.com/track/{self.spotify_id}"

    @property
    def uri(self) -> str:
        return f"spotify:track:{self.spotify_id}"

    @classmethod
    def from_spotify(cls, data: dict) -> "Track":
        album = data.get("album") or {}
        artists = data.get("artists") or []
        return cls(
            spotify_id=data["id"],
            title=data["name"],
            artists=[a["name"] for a in artists],
            artist_ids=[a["id"] for a in artists if a.get("id")],
            album=album.get("name"),
            release_date=album.get("release_date"),
            duration_ms=data.get("duration_ms") or 0,
            # external_ids was removed in Feb 2026 and restored in March 2026;
            # treat it as optional in case it disappears again.
            isrc=(data.get("external_ids") or {}).get("isrc"),
            explicit=bool(data.get("explicit")),
        )
