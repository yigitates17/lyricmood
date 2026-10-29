# lyricmood

A personal, lyrics-aware music assistant. Ask it why you like a song, what
the lyrics actually say (including in languages you don't speak well), which
songs feel like this one *lyrically*, or for a playlist that moves from sad to
hopeful.

lyricmood has no chat UI of its own. It's a set of tools you can use from any
LLM client:

- **MCP server**: use Claude Code, Codex, Gemini CLI or Claude Desktop as
  the chat interface. See [Use it from an MCP client](#use-it-from-an-mcp-client).
- **Terminal chat** (`lyricmood chat`, coming in milestone 5): a chat in
  your terminal with a swappable LLM backend. The first backend is your own
  Claude Code (no API key). An API backend (LiteLLM: OpenRouter, Gemini,
  OpenAI, Anthropic) can be added later.

Both read the same tool registry, so every tool is defined once.

> **Status: milestones 1–4 and 6 of 7 done; 5 (terminal chat) is next.** Spotify search, lyrics (LRCLIB), audio
> features (ReccoBeats), Last.fm tags, artist top tracks, the song you're
> playing now, lyrics-based similar songs, artist comparisons, your top and
> liked songs, mood playlists and playlist editing, all available to MCP
> clients. See [Roadmap](#roadmap).

## Why this exists

Plenty of Spotify MCP servers exist (playback control, playlist CRUD, a few
that fetch lyrics for one track). None of them combine lyrics, mood
features and tags to answer *"what is this song about, and why do I like
it?"*, or rank songs by lyrical similarity. Existing lyrics-similarity
projects are mostly notebooks over static Kaggle datasets. lyricmood is built
around three things they lack:

- understanding lyrics for a non-native speaker
- lyric-based discovery that only embeds tracks when they come up as
  candidates, never a whole catalog
- working with whatever LLM you already use

## Setup

Requires [uv](https://docs.astral.sh/uv/). uv installs Python 3.12 itself if
you don't have it.

```bash
git clone https://github.com/yigitates17/lyricmood.git
cd lyricmood
uv sync --extra embeddings   # or plain `uv sync`, see below
cp .env.example .env
```

**The `embeddings` extra** adds what `find_similar_by_lyrics` needs:
ONNX Runtime, Hugging Face `tokenizers`, and LanceDB. That's about 385 MB
on top of the base install's ~215 MB. The first similarity search also
downloads the lyrics model,
[`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small),
once: about 480 MB, into `~/.lyricmood/models`. It runs on CPU through ONNX,
so PyTorch is not needed. Leave the extra out if you only want lyrics
explanations, profiles and playlists. Everything else works, and the
similarity tool will say how to enable it.

### Spotify (required)

Spotify's Web API now requires the app owner to have **Premium**, and
Development Mode apps are limited to 5 users. Both are fine for personal use.

1. Open the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
   and log in with your Premium account. Accept the developer terms if asked.
2. Click **Create app** and fill in:
   - **App name**: `lyricmood` (anything works)
   - **App description**: `Personal lyrics-aware music assistant`
   - **Redirect URIs**: enter `http://127.0.0.1:8888/callback` and click
     **Add**. Use `127.0.0.1`, not `localhost`; Spotify rejects `localhost`.
   - **Which API/SDKs are you planning to use?**: tick **Web API**.
   - Accept the terms and click **Save**.
3. Open the app, go to **Settings**, and copy the **Client ID** into `.env`:
   ```
   SPOTIFY_CLIENT_ID=your_client_id_here
   ```
   You don't need the Client Secret. lyricmood logs in with PKCE, which is
   designed for apps that can't keep a secret.
4. Log in:
   ```bash
   uv run lyricmood auth spotify
   ```
   Your browser opens Spotify's consent page. Click **Agree**. The browser
   then shows "Authentication successful" and the terminal prints
   `Spotify: logged in`. The token is saved to `~/.lyricmood/spotify_token.json`
   and refreshes automatically.

If it doesn't work:

- **`INVALID_CLIENT: Invalid redirect URI`**: the URI in `.env` must match
  the dashboard character for character, including `/callback` and no
  trailing slash.
- **`Address already in use`**: something else is using port 8888. Pick
  another port, say 8899, and change it in **both** the dashboard and
  `SPOTIFY_REDIRECT_URI`.
- **403 "user not registered"**: in the dashboard, go to **User Management**
  and add the email of your Spotify account.
- **"Spotify session expired"** after several months: Spotify refresh tokens
  now last six months. Run `uv run lyricmood auth spotify` again.

The login asks for all the permissions lyricmood will eventually need (top
tracks, saved tracks, reading and editing playlists, the currently playing
song), so later milestones won't make you log in again.

### Last.fm (recommended)

Last.fm provides mood and genre tags, artist top tracks and, later, similar
songs. Without a key, `get_track_profile` leaves out tags and
`get_artist_tracks` falls back to Spotify's search order.

1. Go to https://www.last.fm/api/account/create (log in or create a free
   account).
2. Fill in **Application name** (e.g. `lyricmood`). The other fields can
   stay empty.
3. Copy the **API key** (not the shared secret) into `.env`:
   ```
   LASTFM_API_KEY=your_key_here
   ```

### LRCLIB and ReccoBeats

Neither needs a key. lyricmood sends a User-Agent that identifies it, as
LRCLIB asks.

### YouTube

This comes with milestone 7.

## Usage

```bash
uv run lyricmood status                         # keys, login state, cache size
uv run lyricmood tools                          # list tools
uv run lyricmood tools --schema                 # ...with JSON input schemas
uv run lyricmood call search_track query="nemo nightwish"
uv run lyricmood call search_track query="artist:Nightwish" limit=10
uv run lyricmood call search_track query="https://open.spotify.com/track/..."
uv run lyricmood call get_track_profile track="Carnival of Rust Poets of the Fall"
uv run lyricmood call get_lyrics track=2DtPEKs3Pj7E06EoWW4vw8
uv run lyricmood call get_artist_tracks artist="Haloo Helsinki!" limit=5
uv run lyricmood call now_playing                # play something on Spotify first
uv run lyricmood call now_playing with_profile=false
uv run lyricmood call find_similar_by_lyrics track=2DtPEKs3Pj7E06EoWW4vw8 k=8
uv run lyricmood call find_similar_by_lyrics track=2DtPEKs3Pj7E06EoWW4vw8 mood_weight=0.4
uv run lyricmood call find_similar_by_lyrics track="Nemo Nightwish" expand_pool=false
uv run lyricmood call analyze_artists --json '{"artists": ["Poets of the Fall", "Nightwish"]}'
uv run lyricmood call get_my_top_tracks time_range=short_term limit=10
uv run lyricmood call get_my_playlists
uv run lyricmood call build_mood_playlist --json '{"mood_text": "heartbreak, lonely nights", "valence": 0.2, "energy": 0.35, "seed_artists": ["Nightwish"], "size": 10}'
uv run lyricmood call build_mood_playlist --json '{"mood_text": "-", "size": 10, "arc": [{"mood_text": "despair, loss", "valence": 0.15}, {"mood_text": "hope, light after the storm", "valence": 0.75}]}'
```

`call` runs any tool directly and prints exactly the JSON an LLM would get.
It's the quickest way to test a tool without an LLM.

Your data (SQLite cache, Spotify token, lyrics vectors, the model) lives in
`~/.lyricmood`. It sits outside the repo on purpose: MCP clients start the
server from whatever directory they like, and they all need to share one
cache. Set `LYRICMOOD_DATA_DIR` to move it.

## Use it from an MCP client

The MCP server (`lyricmood-mcp`) exposes every tool over stdio. It finds
`.env` and `~/.lyricmood` by itself, so clients need no environment
variables. Before registering it:

```bash
cd ~/lyricmood
uv sync --extra embeddings
uv run lyricmood auth spotify      # the server can't open a login prompt itself
uv run lyricmood download-model    # so the first similarity search doesn't
                                   # spend a tool call downloading 480 MB
```

Every example below points at the server binary inside the project's
virtualenv. Replace `$HOME/lyricmood` if you cloned somewhere else. A direct
path works no matter which directory the client starts in, needs no extra
flags, and starts in about half a second. Re-run `uv sync --extra embeddings`
after pulling updates.

**Claude Code**

```bash
claude mcp add -s user lyricmood -- "$HOME/lyricmood/.venv/bin/lyricmood-mcp"
claude mcp list                    # should show: lyricmood ... ✓ Connected
```

`-s user` makes it available in every project. Inside Claude Code, `/mcp`
lists the tools. Try *"What is Carnival of Rust by Poets of the Fall
about?"* or *"Explain the song I'm playing right now."*

**Codex CLI**

```bash
codex mcp add lyricmood -- "$HOME/lyricmood/.venv/bin/lyricmood-mcp"
```

Codex stops tool calls after 60 s by default, and a song's first
`find_similar_by_lyrics` can take about a minute. Raise the limit in
`~/.codex/config.toml`:

```toml
[mcp_servers.lyricmood]
command = "/Users/you/lyricmood/.venv/bin/lyricmood-mcp"
tool_timeout_sec = 180
```

**Gemini CLI**

```bash
gemini mcp add -s user lyricmood "$HOME/lyricmood/.venv/bin/lyricmood-mcp"
```

Use `-s user`. Gemini refuses project-scoped config in folders it doesn't
trust.

**Claude Desktop**

Add this to `~/Library/Application Support/Claude/claude_desktop_config.json`
and restart the app:

```json
{
  "mcpServers": {
    "lyricmood": {
      "command": "/Users/you/lyricmood/.venv/bin/lyricmood-mcp"
    }
  }
}
```

The desktop app has been reported to cancel tool calls after about 60 s.
If a first similarity search fails there, run it once from the terminal
(`uv run lyricmood call find_similar_by_lyrics track=...`) to fill the
cache, then ask again.

The server also sends the client short instructions: the user may not be a
native speaker, so explain idioms and translate key lines; "this song"
means `now_playing`; and mention missing data instead of guessing.

## Tools

| Tool | Milestone | Status |
|---|---|---|
| `search_track(query, limit)` | 1 | ✅ |
| `get_lyrics(track)` | 2 | ✅ |
| `get_track_profile(track)` | 2 | ✅ |
| `get_artist_tracks(artist, limit)` | 2 | ✅ |
| `now_playing(with_profile)` | 2 | ✅ |
| `find_similar_by_lyrics(track, k, expand_pool, mood_weight)` | 3 | ✅ |
| `analyze_artists(artists, tracks_per_artist)` | 3 | ✅ |
| `get_my_top_tracks(time_range, limit)` | 6 | ✅ |
| `get_saved_tracks(limit, offset)` | 6 | ✅ |
| `get_my_playlists(limit, offset)` | 6 | ✅ (added: needed to find playlist IDs) |
| `get_playlist_tracks(playlist, limit)` | 6 | ✅ (added: needed before reordering) |
| `build_mood_playlist(mood_text, valence, energy, seed_artists, size, arc, …)` | 6 | ✅ |
| `create_spotify_playlist(name, tracks, description, public)` | 6 | ✅ |
| `update_spotify_playlist(playlist_id, add, remove, reorder)` | 6 | ✅ |
| `convert_playlist` | 7 | planned |

Every `track` argument takes a Spotify ID, URL or URI, or free text, which
resolves to the top search hit.

**`get_track_profile`** returns, in one call:

- the lyrics (cut at a stanza boundary past 2,500 characters) and simple
  stats. `repeated_lines` usually marks the chorus.
- ReccoBeats audio features: valence (musical positivity), energy,
  danceability, acousticness, tempo and key.
- Last.fm tags. Last.fm rarely tags individual songs, so the artist's tags
  are often used instead; `tags_source` says which.

Each source can fail on its own (ReccoBeats in particular is flaky). The
profile still comes back, with a `missing` field saying what's absent and
why. The LLM interprets the lyrics; the code only gathers facts.

**Lyrics matching**: artist + title decide the match; duration only breaks
ties. lyricmood tries LRCLIB `/api/get` with artist, title and duration
first. If that misses, it searches and takes the entry whose artist and
title match (fuzzy), choosing the closest length and preferring entries
with lyrics over ones flagged instrumental. Length never rejects a match,
because plain lyrics are the same across remasters, compilations and radio
edits. Length only matters for synced (timestamped) lyrics, which lyricmood
doesn't use. Version suffixes ("- Remastered 2004", "(feat. X)") are
stripped before lookup. "Live" and "Orchestral Version" are kept, because
their lyrics can differ or be missing. Lyrics are cached forever; misses are
retried after 14 days.

**`find_similar_by_lyrics`** finds songs whose lyrics are about similar
things, across languages:

1. It gathers about 40 candidates from Last.fm (similar tracks, the seed's
   top tags, similar artists' hits) and ReccoBeats recommendations, taking
   from each source in turn so no single one dominates.
2. It resolves each candidate to Spotify and fetches its lyrics.
3. It embeds the lyrics, one vector per stanza, with repeated choruses
   counted once and the song vector being their average. Vectors are
   stored in LanceDB, so a song is only ever embedded once.
4. It ranks the candidates by similarity to the seed.

Each result includes `matching_lines`: the two passages, one from each
song, that matched best. `mood_weight` (0–1) also favors songs with
similar valence and energy. `expand_pool=false` skips the network and ranks
only songs analyzed in earlier searches.

Scores are relative within one search (the best match is 1.0), not
absolute. The first search for a song takes about a minute (mostly
LRCLIB, which is rate-limited politely); repeats take under a second.

Known limitation: the model groups lyrics partly by language. An English
song about loss can score close to an English party song. In a mixed
English/Finnish test, the nearest neighbour was always the right topic,
but scores aren't comparable across languages. See
[DESIGN §8.2](docs/DESIGN.md).

**`build_mood_playlist`** proposes an ordered list. It doesn't save
anything.

- **Where songs come from:** the seed artists' top songs plus three similar
  artists each, or your own top and liked songs when no artists are given.
  The pool is about 60 songs (`LYRICMOOD_MOOD_POOL_SIZE`).
- **How each song is scored:** half on lyrics, comparing `mood_text`
  against each song's stanzas, and half on music, how close its valence
  and energy are to the targets. `feature_weight` changes that split.
  Claude chooses the targets from your words, for example "sad" ≈ valence
  0.2. A song missing lyrics or features counts as typical on that side,
  so missing data neither helps nor hurts it.
- **`arc`** gives 2–5 stages, for example heartbroken → hopeful. Songs
  move through them gradually, with targets blended position by position.
- **`max_per_artist`** keeps variety (default 3).
- **Mixed languages:** give the mood in each language on separate lines,
  because the model favours lyrics in the query's language.

**Playlist writes** (`create_spotify_playlist`, `update_spotify_playlist`)
change your Spotify account. New playlists are private by default, and
their docstrings tell the LLM to confirm with you first. Your MCP client
also asks before running a tool unless you've allowed it.
- `update_spotify_playlist` works out the end state and checks it before
  writing anything, so a bad `reorder` changes nothing.
- It refuses to reorder playlists that contain podcast episodes or local
  files, which a rewrite would drop.
- It skips songs that are already in the playlist.
- Spotify only lets apps read and edit your own (or collaborative)
  playlists.

**`analyze_artists`** samples each artist's top songs and returns:

- lyric excerpts (opening stanza plus chorus, about 500 characters each)
- average valence, energy, danceability and acousticness
- Last.fm tags, plus the tags the artists share

The LLM reads the excerpts to find the common themes behind "why do I like
these?". It doesn't need the embeddings extra.

### Adding a tool

Put a module in `lyricmood/tools/`. Modules there are discovered
automatically and show up in the MCP server, the chat and the CLI:

```python
from pydantic import BaseModel
from lyricmood.context import AppContext
from lyricmood.registry import tool


class EchoInput(BaseModel):
    text: str


class EchoOutput(BaseModel):
    text: str


@tool
async def echo(ctx: AppContext, args: EchoInput) -> EchoOutput:
    """The docstring is the description the LLM sees."""
    return EchoOutput(text=args.text)
```

API clients live in `lyricmood/core/` and must not contain LLM logic. Tools
get those clients from `ctx`.

## Development

```bash
uv sync --extra embeddings
uv run pytest            # offline: HTTP is mocked with respx + JSON fixtures
uv run ruff check . && uv run ruff format --check .
```

Similarity tests use a fake word-overlap embedder, so they're instant and
need no model. `tests/test_model_quality.py` checks the real model on a
small invented English/Finnish topic set. It runs only when the model is
already downloaded, so CI skips it.

CI runs the same commands and never calls a real API. LRCLIB, Last.fm and
ReccoBeats fixtures in `tests/fixtures/` are recorded responses with images
and country lists removed. All lyrics in them are replaced with invented
text, because test data must never contain copyrighted lyrics. The Spotify
fixtures are hand-built from Spotify's documented response shapes.

## Spotify API constraints (2026)

Spotify's February 2026 changes for Development Mode apps shape the design:

- **Gone**: recommendations, audio features, related artists (removed in
  2024), artist top tracks, and batch `GET /tracks?ids=`. lyricmood gets
  audio features from ReccoBeats and similar tracks and artists from Last.fm.
- **Search** returns at most 10 results per page. lyricmood paginates when
  asked for more.
- **ISRC** (`external_ids`) was removed and then restored in March 2026.
  lyricmood treats it as optional.
- **Playlist contents** are only returned for playlists you own.
  `convert_playlist` can't read other people's Spotify playlists.

[docs/DESIGN.md](docs/DESIGN.md) explains the architecture, the reasoning
behind it, and open questions.

## Roadmap

1. ✅ Skeleton, config, SQLite cache, Spotify OAuth, `search_track`
2. ✅ LRCLIB, Last.fm and ReccoBeats clients, `get_lyrics`, `get_track_profile`, `get_artist_tracks`, `now_playing`
3. ✅ Embeddings (multilingual-e5-small via ONNX) + LanceDB, `find_similar_by_lyrics`, `analyze_artists`
4. ✅ MCP server exposing all tools (`lyricmood-mcp`)
5. Terminal chat (`lyricmood chat`) with a swappable backend, starting with Claude Code, plus `lyricmood watch`, which summarizes each new song as it starts playing
6. ✅ Top/liked songs, playlist create/update/read, `build_mood_playlist` (done before 5)
7. YouTube client, `convert_playlist`

## License

MIT
