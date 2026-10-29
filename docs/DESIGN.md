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
That's why `find_similar_by_lyrics` can blend in mood features and embeds
lyrics per stanza (§8.2). Our own check confirmed a related weakness: raw
scores are skewed by language.

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

Candidates from Last.fm, which have no Spotify IDs, are resolved to a
Spotify track before embedding (§8.3).

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

### 8.2 M3 (done): embeddings, `find_similar_by_lyrics`, `analyze_artists`

**Runtime (decided: ONNX, no PyTorch).** sentence-transformers would pull in
PyTorch: 568 MB installed on macOS, and the default Linux wheel bundles
CUDA (~2 GB). The model's official repo ships an ONNX export
(`onnx/model.onnx`, the same weights), so lyricmood runs it with
`onnxruntime` + `tokenizers` and does e5's mean pooling and L2
normalization itself (`core/embeddings.py`). The revision is pinned
(`614241f`) so a model update can't silently change stored vectors. The
smaller exports in the repo were not used: `model_O4` is a GPU fp16
optimization, and `qint8_avx512_vnni` targets Intel CPUs.

**Optional extra (decided).** `onnxruntime`, `tokenizers`,
`huggingface-hub` and `lancedb` live in the `embeddings` extra. Measured:
the base install is ~215 MB and the extra adds ~385 MB; the model download
is 480 MB. numpy and Polars are base dependencies, because the ranking
code and future mood playlists use them. Without the extra,
`find_similar_by_lyrics` raises a `ConfigError` with the install command.

**Prefix.** Every stored lyrics vector uses `"passage: "`. Lyrics are only
compared with lyrics, so the prefix just has to be consistent. Free-text
mood queries (M6) should use `"query: "` against these passages, which is
e5's asymmetric setup. The model ID stored with each vector includes the
prefix and revision.

**Chunking.** Lyrics are split into stanzas. Repeated stanzas (the chorus)
are kept once, so a 4× chorus doesn't dominate. Stanzas under 60
characters merge into the next, and at most 16 chunks are kept. The song
vector is the normalized mean of its chunk vectors. Chunks are stored too,
so `matching_lines` can show the most alike stanza pair. A single
512-token window would mostly embed verse 1.

**Storage.** LanceDB at `~/.lyricmood/vectors`, two tables: `lyrics` (key
`model:spotify_id`, isrc, vector) and `lyrics_chunks`. Upserts go through
`merge_insert`. Nothing is pre-embedded; only candidates are.

**Measured behaviour (2026-10-29).** Twelve invented lines: 3 topics (loss,
party, rebellion) × 2 languages (English, Finnish) × 2 lines.

| Setup | Nearest neighbour same topic | Best other-language match same topic | Mean sim same-topic vs same-language-other-topic |
|---|---|---|---|
| `passage:`, raw cosine | 0.92 | 1.00 | 0.857 vs 0.858 |
| `passage:`, centered | **1.00** | 1.00 | 0.071 vs 0.085 |
| `query:`, raw | 0.83 | 1.00 | 0.847 vs 0.836 |
| `query:`, centered | 0.92 | 1.00 | 0.083 vs 0.014 |

Rankings are mostly right, but raw scores are dominated by language: an
off-topic line in the same language scores as high, on average, as an
on-topic line. So:

- Vectors are **centered** on the compared set (seed + candidates, when
  there are ≥ 5) before cosine. Scores are then min-max scaled within the
  pool, which makes them relative within one search, never absolute.
- `tests/test_model_quality.py` keeps this check (nearest-neighbour topic
  accuracy ≥ 0.9). It runs locally when the model is downloaded and is
  skipped in CI.
- **Known limitation:** a pool that mixes languages still favours the
  seed's language somewhat. Possible fixes later: per-language centering
  (needs language detection, which was dropped in M2) or a larger model
  (e5-base, ~1.1 GB). Neither is planned until real use shows it matters.

**`find_similar_by_lyrics`**

- **Pool:** Last.fm `track.getSimilar` (25), `tag.getTopTracks` for the
  seed's top 2 tags (10 each), 5 similar artists × 2 top tracks, and
  ReccoBeats recommendations (15). Sources are interleaved round-robin and
  deduplicated by normalized artist + title, then capped at
  `LYRICMOOD_SIMILAR_POOL_SIZE` (40). A failing source becomes an entry in
  `missing`.
- **Resolution:** every candidate is resolved to Spotify (§8.3).
  ReccoBeats candidates already carry Spotify IDs and go through
  `get_track`, so every embedded song has a row in the SQLite `tracks`
  table.
- **`expand_pool=false`** (decided): ranks only songs already in the vector
  store (earlier searches), with no network beyond the seed. The spec
  didn't define `False`; the alternative, a smaller fresh pool, was
  rejected because it is slow and still needs the network.
- **Mood blend:** `mood_weight` w gives
  `score = (1-w)·lyrics + w·(1 - dist((valence, energy), seed)/√2)`. When
  either side has no ReccoBeats features, the mood term is skipped rather
  than counted as zero. The default is 0 (lyrics only, as the tool's name
  says); the docstring tells the LLM when to raise it.
- **Measured:** "Carnival of Rust": 39 candidates, 32 with lyrics, **55 s
  cold** (mostly LRCLIB at 2 req/s plus ~33 songs embedded). The next
  search took 13 s, because 7 transient LRCLIB failures were retried
  (errors aren't cached, only real misses). Fully warm: **0.8 s**.
  `expand_pool=false` took 1.4 s over 40 stored songs. Top result:
  Jeff Buckley's "Broken down and hungry for your love / With no way to
  feed it", matched against "Come feed the rain / 'Cause I'm thirsty for
  your love".

**`analyze_artists`**

- Per artist: top tracks (shared `artist_top` helper, the same as
  `get_artist_tracks`), Last.fm artist tags, ReccoBeats feature means
  (Polars `group_by`), and per-song excerpts (opening stanza + the stanza
  holding the most repeated line, cut to `LYRICMOOD_EXCERPT_MAX_CHARS`
  500). Shared tags across artists are listed. 3 artists × 4 songs gave
  ~6,400 characters of JSON.
- **Removed: an artist-to-artist lyrics similarity score.** One was
  prototyped (centroid cosine). With 3 artists, centering made every value
  negative, and the ranking mostly reflected language: the two
  English-writing bands came out "closest", the Finnish one farthest. The
  spec didn't ask for it, and the LLM judges shared themes better from the
  excerpts it already receives, so it was dropped instead of shipping a
  misleading number. `analyze_artists` therefore doesn't need the
  embeddings extra.

### 8.2a M4 (done): MCP server

- **`lyricmood/mcp_server.py`** builds a FastMCP (4.x) server from the
  registry. Each `ToolSpec` becomes a function whose signature is the input
  model's fields as keyword parameters (with pydantic constraints and
  descriptions), so clients see `track`, `k`, `mood_weight` and so on
  instead of one nested `args` object. Validation happens twice, once by
  FastMCP from the signature and once by the registry, so a malformed call
  never reaches a tool.
- **Results** are `registry.call_json` strings, one compact JSON text
  block. `output_schema=None` turns off FastMCP's structured copy, which
  some clients would show alongside the text and so double the tokens.
- **Errors:** a `LyricmoodError` becomes a FastMCP `ToolError` carrying the
  same user-facing message ("Run `uv run lyricmood auth spotify`...").
  Validation errors come back as tool errors too. Other exceptions are
  bugs and surface as generic failures.
- **One `AppContext` per server process**, created in the FastMCP lifespan
  and closed on shutdown, so all tool calls share the HTTP clients, the
  rate limiters, the SQLite connection and the loaded model.
- **Server instructions** (sent to the client at initialize) hold the only
  LLM guidance in the project besides the tool docstrings: the user may
  not be a native speaker, so explain idioms and translate key lines;
  "this song" means `now_playing`; warn before slow similarity searches;
  mention `missing` instead of guessing. This lives in the MCP layer, never
  in core.
- **Launch:** the README registers `.venv/bin/lyricmood-mcp` by absolute
  path rather than `uv run --directory ...`. It needs no flags (Gemini's
  `mcp add` parses flags after the command), it doesn't depend on the
  client's working directory, and it starts in ~0.6 s. Everything else
  (`.env`, the data dir) is found from the package location (§3).
- **Timeouts:** a song's first `find_similar_by_lyrics` takes ~55 s. Codex
  defaults to 60 s per tool call (`tool_timeout_sec`), and Claude Desktop
  has been reported to cancel at ~60 s, so the README raises Codex's limit
  and offers a CLI warm-up for Desktop. Claude Code's tool timeout is long
  by default. `lyricmood download-model` exists so the 480 MB model never
  downloads inside a tool call.
- **Verification (2026-10-29):**
  - A FastMCP client spawned the real binary over stdio and called
    `search_track`, `get_track_profile`, `find_similar_by_lyrics` and
    `now_playing` against the live APIs; bad arguments came back as tool
    errors.
  - `claude mcp list` reported "✓ Connected" for a project-scoped config
    in a scratch folder.
  - A headless `claude -p` chat test was not possible from inside another
    Claude Code session (nested sessions are blocked), so the in-chat test
    is left to the owner.
  - Unit tests run the server in-memory with mocked APIs and check that a
    newly registered tool appears without changes to the server.

### 8.3 M3: candidate identity (decided: option a)

Every candidate is resolved to a Spotify ID before embedding (one cached
search each), so it gets ReccoBeats features for the mood blend.

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

Option (b) was recommended at first to save Spotify calls. Working through
an example showed it would lose what makes results good: most candidates
would have no audio features, so the mood blend would be impossible. The
owner chose (a). The other argument for (a), more reliable lyrics matching
through Spotify's exact length, mostly went away once lyrics started
matching on names (§8.1).

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
  rank by name similarity and pick the closest length. Matches that are
  more than `LYRICMOOD_PLAYLIST_LENGTH_WARN_S` seconds off (default 15;
  0 turns it off) are still added, but listed in the "please check" report
  (decided). This goes in the README with M7. ytmusicapi's "songs" filter returns the album
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
