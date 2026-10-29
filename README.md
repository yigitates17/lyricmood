# lyricmood

A personal, lyrics-aware music assistant. Ask it why you like a song, what
the lyrics actually say (including in languages you don't speak well), which
songs feel like this one *lyrically*, or for a playlist that moves from sad to
hopeful.

lyricmood has no chat UI of its own. It's a set of tools you can use from any
LLM client:

- **MCP server**: use Claude Code, Codex, Gemini CLI or Claude Desktop as
  the chat interface. *(milestone 4)*
- **Terminal chat**: a small agent loop over LiteLLM, so any model from
  OpenRouter, Gemini, OpenAI or Anthropic works. *(milestone 5)*

Both read the same tool registry, so every tool is defined once.

> **Status: milestone 1 of 7.** Project skeleton, config, SQLite cache,
> Spotify login, and the `search_track` tool. See [Roadmap](#roadmap).

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
uv sync
cp .env.example .env
```

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

### Last.fm, YouTube, LLM keys

These come with milestones 2, 7 and 5. Each will get its own section here.

## Usage

```bash
uv run lyricmood status                         # keys, login state, cache size
uv run lyricmood tools                          # list tools
uv run lyricmood tools --schema                 # ...with JSON input schemas
uv run lyricmood call search_track query="nemo nightwish"
uv run lyricmood call search_track query="artist:Nightwish" limit=10
uv run lyricmood call search_track query="https://open.spotify.com/track/..."
```

`call` runs any tool directly and prints exactly the JSON an LLM would get.
It's the quickest way to test a tool without an LLM.

Your data (SQLite cache, Spotify token, and later the vector store) lives in
`~/.lyricmood`. It sits outside the repo on purpose: MCP clients start the
server from whatever directory they like, and they all need to share one
cache. Set `LYRICMOOD_DATA_DIR` to move it.

## Tools

| Tool | Milestone | Status |
|---|---|---|
| `search_track(query, limit)` | 1 | ✅ |
| `get_artist_tracks`, `get_lyrics`, `get_track_profile`, `now_playing` | 2 | planned |
| `find_similar_by_lyrics`, `analyze_artists` | 3 | planned |
| `get_my_top_tracks`, `get_saved_tracks` | 4–6 | planned |
| `create_spotify_playlist`, `update_spotify_playlist`, `build_mood_playlist` | 6 | planned |
| `convert_playlist` | 7 | planned |

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
uv run pytest            # offline: HTTP is mocked with respx + JSON fixtures
uv run ruff check . && uv run ruff format --check .
```

CI runs the same commands and never calls a real API. The fixtures in
`tests/fixtures/` follow the response shapes in Spotify's documentation.
Test data must never contain copyrighted lyrics, so lyrics fixtures will be
invented text.

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
2. LRCLIB, Last.fm and ReccoBeats clients, `get_track_profile`, `now_playing`
3. Embeddings (multilingual-e5-small) + LanceDB, `find_similar_by_lyrics`, `analyze_artists`
4. MCP server exposing all tools
5. LiteLLM terminal chat, plus `lyricmood watch`, which summarizes each new song as it starts playing
6. Playlist create/update, `build_mood_playlist`
7. YouTube client, `convert_playlist`

## License

MIT
