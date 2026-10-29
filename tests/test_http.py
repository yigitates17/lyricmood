import httpx
import pytest
import respx

from lyricmood.core.errors import ApiError, RateLimited
from lyricmood.core.http import ApiClient, RateLimiter
from tests.conftest import FAST_RETRY

BASE = "https://example.test"


class RefreshingClient(ApiClient):
    name = "Example"

    def __init__(self):
        super().__init__(BASE, limiter=RateLimiter(1000, burst=10), retry=FAST_RETRY)
        self.refreshed = 0

    async def _on_unauthorized(self) -> bool:
        self.refreshed += 1
        return True


@pytest.fixture
async def client():
    c = RefreshingClient()
    yield c
    await c.aclose()


@respx.mock(base_url=BASE)
async def test_retries_server_errors_then_succeeds(respx_mock, client):
    route = respx_mock.get("/x").mock(
        side_effect=[httpx.Response(503), httpx.Response(200, json={"ok": True})]
    )
    assert await client.get_json("/x") == {"ok": True}
    assert route.call_count == 2


@respx.mock(base_url=BASE)
async def test_short_retry_after_is_retried(respx_mock, client):
    route = respx_mock.get("/x").mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "0"}),
            httpx.Response(200, json=[]),
        ]
    )
    assert await client.get_json("/x") == []
    assert route.call_count == 2


@respx.mock(base_url=BASE)
async def test_long_retry_after_fails_fast(respx_mock, client):
    route = respx_mock.get("/x").mock(
        return_value=httpx.Response(429, headers={"Retry-After": "3600"})
    )
    with pytest.raises(RateLimited) as info:
        await client.get_json("/x")
    assert info.value.retry_after == 3600
    assert route.call_count == 1


@respx.mock(base_url=BASE)
async def test_persistent_429_raises_rate_limited(respx_mock, client):
    respx_mock.get("/x").mock(return_value=httpx.Response(429))
    with pytest.raises(RateLimited):
        await client.get_json("/x")


@respx.mock(base_url=BASE)
async def test_client_errors_are_not_retried(respx_mock, client):
    route = respx_mock.get("/x").mock(
        return_value=httpx.Response(404, json={"error": {"status": 404, "message": "nope"}})
    )
    with pytest.raises(ApiError, match="404: nope"):
        await client.get_json("/x")
    assert route.call_count == 1


@respx.mock(base_url=BASE)
async def test_network_errors_are_retried(respx_mock, client):
    route = respx_mock.get("/x").mock(side_effect=httpx.ConnectError("down"))
    with pytest.raises(ApiError, match="network error"):
        await client.get_json("/x")
    assert route.call_count == FAST_RETRY.attempts


@respx.mock(base_url=BASE)
async def test_401_refreshes_credentials_once(respx_mock, client):
    route = respx_mock.get("/x").mock(return_value=httpx.Response(401))
    with pytest.raises(ApiError, match="401"):
        await client.get_json("/x")
    assert client.refreshed == 1
    assert route.call_count == 2


@respx.mock(base_url=BASE)
async def test_post_is_not_retried_after_server_error(respx_mock, client):
    route = respx_mock.post("/x").mock(return_value=httpx.Response(503))
    with pytest.raises(ApiError, match="503"):
        await client.request("POST", "/x", json={})
    assert route.call_count == 1


@respx.mock(base_url=BASE)
async def test_post_is_retried_after_rate_limit_or_connect_error(respx_mock, client):
    route = respx_mock.post("/x").mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "0"}),
            httpx.ConnectError("refused"),
            httpx.Response(201, json={"ok": True}),
        ]
    )
    assert (await client.request("POST", "/x", json={})).status_code == 201
    assert route.call_count == 3


@respx.mock(base_url=BASE)
async def test_post_read_timeout_is_not_retried(respx_mock, client):
    route = respx_mock.post("/x").mock(side_effect=httpx.ReadTimeout("slow"))
    with pytest.raises(ApiError, match="may or may not have applied"):
        await client.request("POST", "/x", json={})
    assert route.call_count == 1
