# lyricmood design

This document records how lyricmood is built, why, and what is still
undecided. It's updated at every milestone. Decisions marked **open** need
the owner's input before the milestone that depends on them.

## 1. Goals

- Explain songs to a non-native speaker: what the lyrics say, the idioms and
  slang, and the overall theme, using lyrics plus mood and genre data.
- Answer "why do I like these?" across tracks or artists.
- Discover songs by *lyrical* similarity, not only by sound or genre.
- Build and edit playlists, including mood arcs (e.g. sad → hopeful).
- Work with any LLM: MCP clients, or any model through LiteLLM.

**Non-goals:** a web frontend, multiple users, pre-embedding a catalog,
audio analysis.

## 2. Prior art (researched 2026-10-29)

| Project | What it does | What it lacks for this use case |
|---|---|---|
| Many Spotify MCP servers ([Glama list](https://glama.ai/mcp/servers/integrations/spotify)) | Playback, search, playlist CRUD; some fetch lyrics for one track ([Tempo](https://glama.ai/mcp/servers/amarisaster/Tempo), [Spotify-OAuth-MCP](https://glama.ai/mcp/servers/@latiftplgu/Spotify-OAuth-MCP-server/blob/ff895f6ea0a3eeacaa8fbe5679b6afa0982a7745/README.md)) | No lyrics analysis across tracks, no lyrical similarity, no mood arcs |
| [J.O.H.A.N.N.](https://github.com/AidenEscamilla/Johann) | Saved tracks + Genius lyrics + GPT sentiment | Tied to OpenAI, Postgres, scraped lyrics, sentiment only |
| [Lyrics pair similarity](https://github.com/NotShrirang/Lyrics-Analysis-and-Music-Recommendation-with-Pair-Similarities) | TF-IDF over a Kaggle dataset | Static dataset, lexical only, English only |
| [Spot (HF Space)](https://huggingface.co/spaces/ngeb25/SpotChatbot/blob/main/app.py) | Llama chatbot over a Spotify dataset | Static dataset, no lyrics, no account access |

**Verdict:** the pieces exist separately, but nothing combines lyrics
understanding, lazy lyric embeddings, mood features and playlist actions
behind an LLM-agnostic tool layer. Worth building.

**Lesson from prior art:** raw lyric-embedding similarity has been reported
as weak on its own
([DZone](https://dzone.com/articles/harmonizing-ai-crafting-personalized-song-suggesti)).
That's why `find_similar_by_lyrics` blends in mood features and why §8.3
proposes chunked embeddings.

## 3. Architecture

```
 Claude Code / Codex / Gemini CLI / Claude Desktop      terminal (typer + rich)
                 │  MCP (stdio)                                 │
        lyricmood/mcp_server.py (M4)             lyricmood/chat.py (M5, LiteLLM loop)
                 └───────────────┬──────────────────────────────┘
                     lyricmood/registry.py   ← one ToolSpec per tool
                     lyricmood/tools/*.py    ← tool functions (auto-discovered)
                     lyricmood/context.py    ← AppContext: lazy clients + cache
                                 │
                     lyricmood/core/  (no LLM logic)
          http.py  cache.py  models.py  spotify/  lrclib  lastfm  reccobeats  youtube
          embeddings (M3)  ranking (M3/M6)
                                 │
                 ~/.lyricmood/{cache.sqlite3, spotify_token.json, vectors/}
```

### 3.1 Tool registry contract

A tool is `async def name(ctx: AppContext, args: InputModel) -> OutputModel`
decorated with `@tool`:

- The **name** comes from the function name, the **description** from the
  docstring. The docstring is written for the LLM.
- **Input and output models** are pydantic models read from the type hints.
  Input models stay flat with scalar or list fields, so the MCP adapter (M4)
  can turn them into plain keyword parameters. Nested input objects make
  some MCP clients and smaller models struggle.
- `registry.call_json` serializes with `exclude_none=True` and no whitespace.
  Every token in a tool result costs context in every turn.
- **Errors**: tools raise `LyricmoodError` subclasses whose messages tell the
  user what to do next ("Run `uv run lyricmood auth spotify`"). Adapters pass
  the message to the LLM as an error result, so the model can relay it.
- **Discovery**: every non-underscore module in `lyricmood/tools/` is
  imported by `load_tools()`. To add a feature later, drop in one file. If
  external plugins are ever needed, the next step is Python entry points
  (`lyricmood.tools` group). Not built now (YAGNI).

Shared output shapes live in `tools/schemas.py`. `TrackOut` is the compact
track: `id, title, artists, album, year, duration, isrc`.

### 3.2 Track identity

The canonical identity is the **Spotify track ID**. The ISRC is stored
alongside it and indexed, so tracks from other sources (YouTube, Last.fm)
can be matched later. Every tool that takes a track accepts one string
`track` and resolves it with `SpotifyClient.resolve`: a URL, URI or ID is
fetched directly, and free text resolves to the top search hit.

**Open (M3):** see §8.3. Candidates from Last.fm don't have Spotify IDs.
Resolving each one costs a Spotify search.

## 4. Caching (SQLite, `~/.lyricmood/cache.sqlite3`)

- `tracks(spotify_id PK, isrc, data, updated_at)` holds canonical tracks
  with no expiry. Metadata hardly changes.
- `kv(namespace, key, value, created_at, expires_at)` holds raw API results.
  Namespaces and TTLs:
  - `spotify:search`: 7 days. Stores track IDs that point into `tracks`.
  - `lrclib`: keyed by Spotify ID. Hits never expire; misses expire after
    14 days.
  - `reccobeats:features`: keyed by Spotify ID. Hits never expire; nulls
    expire after 14 days.
  - `reccobeats:recs`: 7 days.
  - `lastfm:<method>`: 30 days. Keyed by normalized artist/title.
- **Negative caching**: `get()` returns a `MISS` sentinel on a miss, so a
  stored `null` ("ReccoBeats has no features for this track", "no lyrics on
  LRCLIB") counts as a real answer. The upstream isn't asked again until the
  TTL runs out.
- **Concurrency**: WAL mode with a 5 s busy timeout. The MCP server and the
  terminal chat can run at the same time on one file. Calls are synchronous
  sqlite3 behind a lock, because local queries take microseconds and
  aiosqlite would be a dependency for no gain.
- Schema version is in `PRAGMA user_version`, ready for migrations.

## 5. HTTP layer (`core/http.py`)

- One `httpx.AsyncClient` and one token-bucket `RateLimiter` per API.
- tenacity retries transport errors, 429 and 5xx with exponential backoff
  and jitter. It honors `Retry-After` when that's longer than the backoff.
- **Fail fast on long waits**: a `Retry-After` above 60 s raises
  `RateLimited` instead of sleeping. Spotify can answer `QUOTA_EXCEEDED`
  with waits of hours, and a chat turn shouldn't hang on that.
- A 401 calls the client's `_on_unauthorized` hook once. Spotify uses it to
  force a token refresh.
- `_check_response` lets a client mark a response as retryable based on its
  body. Last.fm needs this, because it reports rate limits and temporary
  failures as JSON error codes (29, 8, 11, 16), sometimes with HTTP 200.

## 6. Spotify auth

- spotipy `SpotifyPKCE` is used **only** for login and refresh. API calls go
  through async httpx. spotipy is synchronous, and its endpoint wrappers
  have lagged behind Spotify's 2026 renames.
- The interactive login runs only from the CLI. The MCP server talks over
  stdin/stdout, so a prompt there would corrupt the protocol. Without a
  token it raises `AuthRequired`.
- All scopes for every milestone are requested up front: `user-top-read`,
  `user-library-read`, `playlist-read-private`,
  `playlist-read-collaborative`, `playlist-modify-private`,
  `playlist-modify-public`, `user-read-currently-playing` (for `now_playing`,
  §8.6). Adding a scope later means logging in again,
  because spotipy rejects a cached token whose scopes don't cover the
  requested set.
- Refresh tokens expire after six months (Spotify, July 2026). The error
  message tells the user to log in again.

## 7. Data source constraints

| Source | Constraint | Consequence |
|---|---|---|
| Spotify (Feb 2026 dev-mode changes, [changelog](https://developer.spotify.com/documentation/web-api/references/changes/february-2026)) | Removed: artist top tracks, batch `/tracks`, `/users/{id}/playlists`, `POST /users/{id}/playlists`, `/playlists/{id}/tracks` (renamed `/items`). Search max 10 per page. `external_ids` restored in March 2026. Playlist `items` only returned for **your own** playlists. Owner needs Premium; 5 users; quota shared across your apps | Top tracks come from Last.fm. Tracks are fetched one by one and cached forever. Playlists are created with `POST /me/playlists`. Other people's Spotify playlists can't be read |
| LRCLIB | No key. `/api/get` matches artist + title + duration (±2 s server-side); `album_name` turned out to be optional. `/api/search` returns at most 20 results. Answers 503 "ServerOverloaded" intermittently, even for songs it has | Album is not sent (Spotify album names like "Once (Remastered)" cause misses). The search fallback matches on artist + title only; duration just ranks the matches (§8.1). 503s are retried |
| ReccoBeats | No key, unreliable, sparse coverage. `/audio-features?ids=` takes several Spotify IDs; unknown IDs are silently left out, and results come in any order | Batches of 40, matched back by the Spotify URL in `href`. Nulls are cached. Profiles work without features |
| Last.fm | Free key, ~5 req/s guideline. Errors come as JSON codes. Numbers are sometimes strings. **Track-level tags are rare**: none of 6 sampled songs (English and Finnish) had any | Tags fall back to the artist's (`tags_source: "artist"`). Listener-specific tags ("seen live", "favorites", the artist's own name) and tags below count 5 are dropped |
| YouTube Data API | 10,000 units/day. `playlistItems.insert` = 50, **`search.list` = 100** | See §8.5. Matching Spotify → YouTube through search is the real bottleneck |

## 8. Milestone notes and open questions

### 8.1 M2 (done): lyrics, features, tags, artist tracks, now playing

- **`get_artist_tracks`**: ranks by Last.fm `artist.getTopTracks`
  (listeners). It asks for `limit + 10`, because some don't resolve and
  versions collapse. Each track is resolved with a Spotify search
  `artist:"X" track:"Y"` (`find_by_name`), which rejects hits whose artist
  or title fails a fuzzy match (rapidfuzz ratio ≥ 90 after normalization).
  Duplicates by cleaned title (the original and a remaster) collapse into
  one. Tracks that can't be found are listed in `not_on_spotify`. Without
  Last.fm, it falls back to Spotify search `artist:"X"`, filtered to the
  artist. Cold cost is one Last.fm call plus up to `limit + 10` Spotify
  searches (~3 s); after that it's served from the cache.
- **Title cleaning** (`core/text.clean_title`) strips remaster, feat.,
  radio edit, single/album version and bonus track suffixes. It keeps
  "Live", "Acoustic" and "Orchestral Version", because those can have
  different lyrics or none.
- **`get_lyrics`**: plain lyrics only (decided; synced lyrics are ignored).
  The text is cut to `LYRICMOOD_LYRICS_MAX_CHARS` (2,500) at a stanza
  boundary, or a line boundary if the first stanza alone is too long, with
  `truncated: true`. The full text stays in the cache. `match` says
  `exact` or `search`.
- **`get_track_profile`** (decided): computed stats only (lines, stanzas,
  words, unique-word ratio, the 3 most repeated lines), with the truncated
  lyrics included so the LLM does the interpretation. **Language detection
  was dropped**: the LLM reads the lyrics anyway, and a reliable
  multilingual detector would add a large dependency for nothing. The three
  sources are fetched in parallel. A `LyricmoodError` from one source turns
  into an entry in `missing` instead of failing the call, while bugs (other
  exceptions) still raise.
- **`now_playing`**: `GET /me/player/currently-playing`. HTTP 204 means
  `nothing_playing`, and episodes or ads mean `not_a_song`. With
  `with_profile` (the default) it embeds the full profile, saving the LLM a
  second call.
- **ReccoBeats recommendations** and **Last.fm similar tracks, similar
  artists and tag top tracks** are implemented in the clients but not yet
  exposed as tools. They feed `find_similar_by_lyrics` in M3.
- **Lyrics match on names, not duration** (decided; replaces the original
  ±2 s rule). Duration only matters for synced lyrics, whose timestamps
  must fit the recording, and we use plain lyrics only. Real case: LRCLIB
  had about 20 "Poets of the Fall – Carnival of Rust" entries between 221 s
  and 288 s (radio edit, compilations, live sessions). None was within ±2 s
  of Spotify's 263 s, yet all very likely share the same lyrics. The rule
  is now: fuzzy artist + title match, entries with lyrics beat
  instrumental-flagged ones, and the closest duration breaks ties.
  `/api/get` with duration is still tried first, because it's one precise
  call.

### 8.2 M3: embeddings

- Model: `intfloat/multilingual-e5-small` on CPU. Inputs need the
  `"passage: "` prefix for lyrics and `"query: "` for free-text mood
  queries. Max 512 tokens.
- **Proposal:** chunk lyrics by stanza, embed each chunk, and store the mean
  (plus the chunks, for "which part is similar"). Truncating to the first
  512 tokens would mostly embed verse 1.
- LanceDB table `lyrics_vectors(spotify_id, isrc, model, chunk_ix, vector)`.
  `model` is part of the key, so switching models re-embeds lazily instead of
  mixing vector spaces.

### 8.3 M3: candidate identity (**open**; deferred to M3)

For now, follow the original spec: resolve candidates to Spotify ID/ISRC.
The owner will revisit this when M3 starts.

`find_similar_by_lyrics` expands candidates through Last.fm (similar
tracks, tag top tracks) and ReccoBeats recommendations. Last.fm gives
artist + title with no Spotify ID. Resolving every candidate means one
Spotify search each, and the dev-mode quota is shared. Options:

- (a) Resolve every candidate to a Spotify ID before embedding. Matches the
  "keyed by Spotify ID/ISRC" rule exactly, but costs one search per
  candidate (50 candidates = 50 calls, cached afterwards).
- (b) Key candidates by normalized `artist|title` until they make the final
  top-k, then resolve only those. Cuts Spotify calls by about 5×, but the
  cache has a second key type.

Recommendation: (b), with the key stored next to the Spotify ID once
resolved.

### 8.4 M6: `build_mood_playlist` and arcs

- Mood targets: map `mood_text` to (valence, energy) plus a text embedding
  of the mood, and score `w1·cosine(lyrics, mood) + w2·feature distance`.
  ReccoBeats nulls fall back to lyrics only.
- Arc ordering: split the arc into N segments, give each its target point,
  and assign tracks greedily without reuse.

### 8.5 M7: `convert_playlist`

- **ISRC-first only works on the Spotify side.** YouTube videos don't expose
  ISRCs. YouTube → Spotify: fuzzy-match the video title or channel against
  a Spotify search (`isrc:` doesn't help, since the source has no ISRC).
  Spotify → YouTube: `search.list` at 100 units per track. With inserts at
  50, a 50-track playlist costs about 7,500 of the 10,000 daily units.
- **Matching rule (decided)**: match by name and prefer the closest
  length. Duration never rejects a match on its own, but unlike lyrics it
  is still worth checking. A wrong *recording* is
  something you hear: YouTube results for "Nightwish Nemo" include the
  official video with a long intro, live shows, sped-up and nightcore
  edits, covers and 1-hour loops, all with nearly the same name. So:
  rank by name similarity and pick the closest length. **Open:** whether
  large mismatches (more than ~15 s) should also be listed in the "please
  check" report. ytmusicapi's "songs" filter returns the album
  recordings, with clean artist/title/duration, which avoids most of this.
- Resumable transfers: keep a `transfers` table (source, target, item index,
  status, matched ID). Each run continues from the first unfinished item.
- **Decided:** Spotify → YouTube matching searches with
  [`ytmusicapi`](https://github.com/sigma67/ytmusicapi) (open source, MIT,
  unofficial, no quota). Playlist reads and writes still go through the
  official YouTube Data API. Without it, the quota allows about one playlist
  per day.
- Other people's Spotify playlists can't be read (§7). Source playlists must
  be your own.

### 8.6 `now_playing` and watch mode (accepted idea)

- `now_playing` tool (M2, once lyrics exist): returns the currently playing
  track and its profile. Uses `GET /me/player/currently-playing`.
- Automatic summaries need something that runs on its own, because MCP
  tools only run when the LLM calls them. Plan: a `lyricmood watch` command
  (M5, since it needs the LiteLLM layer). It polls currently-playing every
  ~10 s. When the track changes, it fetches lyrics and the profile, asks the
  configured model for a short summary (themes, notable idioms), and prints
  it. Every new song costs one LLM call. Summaries are cached per track, so
  replays are free.

## 9. Testing

- respx mocks httpx, and responses come from JSON fixtures in
  `tests/fixtures/`. No network in CI.
- M1 fixtures are hand-built from Spotify's documented response shapes,
  because there were no credentials at build time. They should be replaced
  with recorded responses (IDs only, no personal data) once logged in.
- Lyrics fixtures must be invented text. The repo is public.

## 10. Feature ideas (not committed)

- `explain_lyrics(track, target_language)`: an MCP *prompt* (not a tool)
  that has the LLM translate line by line and explain idioms. FastMCP
  supports prompts, and they show up as slash commands in Claude Code.
- Vocabulary notebook: save phrases you learned from songs, with the line
  they appeared in, and quiz yourself later.
- Genius annotations: their API serves community explanations of specific
  lines (not lyrics), which helps with "what does this line mean".
- Recently played → mood timeline ("my week in valence").
