from collections import Counter

from pydantic import BaseModel, Field

from lyricmood.context import AppContext
from lyricmood.core.errors import InvalidInput
from lyricmood.core.spotify.client import parse_track_id
from lyricmood.registry import tool
from lyricmood.tools.library_tools import PLAYLIST_ARG, PlaylistOut, playlist_id_or_fail

TRACK_LIST_ARG = (
    "Spotify track IDs, URLs or URIs (use `id` from search_track or "
    "build_mood_playlist). Song names are not accepted here."
)


def track_ids_or_fail(refs: list[str]) -> list[str]:
    ids, bad = [], []
    for ref in refs:
        (ids.append(tid) if (tid := parse_track_id(ref)) else bad.append(ref))
    if bad:
        raise InvalidInput(
            f"Not Spotify track IDs: {', '.join(map(repr, bad[:5]))}. "
            "Look them up with search_track first and pass the `id`."
        )
    return ids


class CreatePlaylistInput(BaseModel):
    name: str = Field(min_length=1, max_length=100, description="Playlist name.")
    tracks: list[str] = Field(default=[], max_length=500, description=TRACK_LIST_ARG)
    description: str = Field(default="", max_length=300, description="Optional description.")
    public: bool = Field(default=False, description="False (default) keeps it private to the user.")


class PlaylistWriteOutput(BaseModel):
    playlist: PlaylistOut
    added: int = 0
    removed: int = 0
    reordered: bool = False
    # Tracks skipped because they were already in the playlist.
    already_there: int | None = None


@tool
async def create_spotify_playlist(
    ctx: AppContext, args: CreatePlaylistInput
) -> PlaylistWriteOutput:
    """Create a new playlist in the user's Spotify account (private unless
    `public` is true) and add the given tracks in order. This changes the
    user's account: only call it after the user has agreed to the name and
    the track list."""
    ids = track_ids_or_fail(args.tracks)
    playlist = await ctx.spotify.create_playlist(args.name, args.description, args.public)
    if ids:
        await ctx.spotify.add_to_playlist(playlist.id, ids)
    playlist.total = len(ids)
    return PlaylistWriteOutput(
        playlist=PlaylistOut.of(playlist, await ctx.spotify.user_id()), added=len(ids)
    )


class UpdatePlaylistInput(BaseModel):
    playlist_id: str = Field(min_length=1, description=PLAYLIST_ARG)
    add: list[str] = Field(
        default=[], max_length=500, description=f"Tracks to append. {TRACK_LIST_ARG}"
    )
    remove: list[str] = Field(
        default=[],
        max_length=500,
        description=f"Tracks to remove (every occurrence). {TRACK_LIST_ARG}",
    )
    reorder: list[str] | None = Field(
        default=None,
        description="The complete new order: every track ID the playlist will contain "
        "after add/remove, each exactly once. Get the current list from "
        "get_playlist_tracks.",
    )


@tool
async def update_spotify_playlist(
    ctx: AppContext, args: UpdatePlaylistInput
) -> PlaylistWriteOutput:
    """Change one of the user's own playlists: remove tracks, append tracks
    (ones already in it are skipped), and/or set a completely new order, in
    that sequence. This changes the user's account: confirm with the user
    first."""
    pid = playlist_id_or_fail(args.playlist_id)
    add, remove = track_ids_or_fail(args.add), track_ids_or_fail(args.remove)
    reorder = track_ids_or_fail(args.reorder) if args.reorder is not None else None
    if not (add or remove or reorder):
        raise InvalidInput("Nothing to do: give `add`, `remove` and/or `reorder`.")

    spotify = ctx.spotify
    playlist = await spotify.playlist(pid)
    user = await spotify.user_id()
    if playlist.owner_id != user and not playlist.collaborative:
        raise InvalidInput("Only your own or collaborative playlists can be changed.")

    current = await spotify.playlist_tracks(pid, limit=10_000)
    current_ids = [t.spotify_id for t in current]
    if reorder is not None and playlist.total is not None and playlist.total != len(current):
        # Replacing the order rewrites the whole playlist; podcast episodes
        # and local files aren't returned as tracks and would be lost.
        raise InvalidInput(
            "This playlist contains podcast episodes or local files, so it can't be "
            "reordered safely. Add or remove songs instead."
        )

    # Work out the end state and validate it before writing anything, so a
    # bad `reorder` never leaves the playlist half-changed.
    removing = set(remove) & set(current_ids)
    remaining = [i for i in current_ids if i not in removing]
    new = [i for i in dict.fromkeys(add) if i not in set(remaining)]
    final = remaining + new
    if reorder is not None and Counter(reorder) != Counter(final):
        missing = [i for i in final if i not in reorder][:5]
        extra = [i for i in reorder if i not in final][:5]
        raise InvalidInput(
            "`reorder` must list exactly the playlist's tracks after add/remove "
            f"(missing: {missing or 'none'}, unexpected: {extra or 'none'}). "
            "Nothing was changed."
        )

    if removing:
        await spotify.remove_from_playlist(pid, list(removing))
    if new:
        await spotify.add_to_playlist(pid, new)
    reordered = reorder is not None and reorder != final
    if reordered:
        await spotify.replace_playlist(pid, reorder)

    playlist.total = len(final)
    return PlaylistWriteOutput(
        playlist=PlaylistOut.of(playlist, user),
        added=len(new),
        removed=len(removing),
        reordered=reordered,
        already_there=(len(set(add)) - len(new)) or None,
    )
