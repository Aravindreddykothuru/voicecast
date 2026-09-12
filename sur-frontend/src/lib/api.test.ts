/**
 * The REST client's auth behaviour.
 *
 * Two things here are load-bearing and were never covered: that a stored
 * session token is attached to every request, and that a 401 drops it. The
 * backend now returns 401 in production for a request carrying only the
 * X-User-Email dev header, so a client that kept a dead session would
 * retry it forever instead of asking the user to log in again.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { API_BASE, ApiError, getCapabilities, listProjects, resolveUrl, startProcessing } from "./api";

function jsonResponse(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

describe("startProcessing", () => {
  afterEach(() => vi.restoreAllMocks());

  it("sends a pinned source language with the process request itself", async () => {
    // The New Dubbing screen used to pin a language with a second call to
    // /confirm-language, which 409s for a run that never parks at the gate,
    // so the choice was dropped. It must travel with /process.
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ id: "p1" }));
    vi.stubGlobal("fetch", fetchMock);

    await startProcessing("p1", {
      preserve_emotion: true,
      clone_voice: false,
      lip_sync_aware: false,
      review_language: false,
      source_language: "hi",
    });

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(`${API_BASE}/api/projects/p1/process`);
    expect(JSON.parse(init.body as string)).toMatchObject({ source_language: "hi", review_language: false });
  });
});

describe("resolveUrl", () => {
  it("leaves an absolute presigned URL alone", () => {
    const s3 = "https://bucket.s3.ap-south-2.amazonaws.com/key?sig=x";
    expect(resolveUrl(s3)).toBe(s3);
  });

  it("prefixes a backend-relative storage path", () => {
    expect(resolveUrl("/api/storage/files/a.mp4")).toBe(`${API_BASE}/api/storage/files/a.mp4`);
  });
});

describe("request auth", () => {
  beforeEach(() => localStorage.clear());
  afterEach(() => vi.restoreAllMocks());

  it("attaches the bearer token when a session is stored", async () => {
    localStorage.setItem("sur.auth", JSON.stringify({ token: "tok123" }));
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse([]));
    vi.stubGlobal("fetch", fetchMock);

    await listProjects();

    const headers = fetchMock.mock.calls[0][1].headers as Record<string, string>;
    expect(headers.Authorization).toBe("Bearer tok123");
  });

  it("sends no Authorization header when there is no session", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse([]));
    vi.stubGlobal("fetch", fetchMock);

    await listProjects();

    const headers = fetchMock.mock.calls[0][1].headers as Record<string, string>;
    expect(headers.Authorization).toBeUndefined();
  });

  it("clears the stored session on 401 so the app re-prompts", async () => {
    localStorage.setItem("sur.auth", JSON.stringify({ token: "expired" }));
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(jsonResponse({ detail: "Authentication required." }, 401)),
    );

    await expect(listProjects()).rejects.toBeInstanceOf(ApiError);
    expect(localStorage.getItem("sur.auth")).toBeNull();
  });

  it("keeps the session on a non-auth error", async () => {
    localStorage.setItem("sur.auth", JSON.stringify({ token: "good" }));
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse({ detail: "boom" }, 500)));

    await expect(listProjects()).rejects.toBeInstanceOf(ApiError);
    expect(localStorage.getItem("sur.auth")).not.toBeNull();
  });

  it("surfaces the backend's detail message, not just the status text", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(jsonResponse({ detail: "Incorrect email or password." }, 401)),
    );

    await expect(getCapabilities()).rejects.toThrow("Incorrect email or password.");
  });
});
