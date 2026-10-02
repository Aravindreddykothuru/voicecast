/**
 * Which identity the WebSocket carries, and HOW, is a security decision.
 *
 * The backend authorizes the subscription before accepting it, and only
 * trusts `user_email` outside production. If this builder sent the dev-stub
 * address while a real session existed, a logged-in user would be refused
 * their own project's events.
 *
 * The session token must never be in the URL. A query string is written to
 * the access log in plaintext, and uvicorn logged every JWT this app opened
 * a socket with:
 *
 *     WebSocket /ws/projects/<id>?token=eyJhbGciOi... [accepted]
 *
 * It travels in Sec-WebSocket-Protocol instead, which is what the second
 * argument to `new WebSocket(url, protocols)` becomes.
 */
import { beforeEach, describe, expect, it } from "vitest";
import { BEARER_SUBPROTOCOL, wsProtocols, wsUrl } from "./useProjectEvents";
import { USER_EMAIL } from "./api";

describe("wsUrl", () => {
  beforeEach(() => localStorage.clear());

  it("keeps the session token out of the URL entirely", () => {
    localStorage.setItem("sur.auth", JSON.stringify({ token: "abc.def.ghi" }));
    const url = wsUrl("p1");
    expect(url).not.toContain("abc.def.ghi");
    expect(url).not.toContain("token=");
    expect(url).not.toContain("user_email");
    expect(url).toBe("ws://localhost:8000/ws/projects/p1");
  });

  it("sends the session token as a subprotocol instead", () => {
    localStorage.setItem("sur.auth", JSON.stringify({ token: "abc.def.ghi" }));
    expect(wsProtocols()).toEqual([BEARER_SUBPROTOCOL, "abc.def.ghi"]);
  });

  it("offers no subprotocol when there is no session", () => {
    expect(wsProtocols()).toEqual([]);
  });

  it("falls back to the dev email only when there is no token", () => {
    const url = wsUrl("p1");
    expect(url).toContain(`user_email=${encodeURIComponent(USER_EMAIL)}`);
    expect(url).not.toContain("token=");
  });

  it("does not send a token key when the stored session has no token", () => {
    localStorage.setItem("sur.auth", JSON.stringify({ user: { id: "u1" } }));
    expect(wsUrl("p1")).toContain("user_email=");
  });

  it("survives corrupt storage instead of throwing", () => {
    localStorage.setItem("sur.auth", "not json {{{");
    expect(() => wsUrl("p1")).not.toThrow();
    expect(wsUrl("p1")).toContain("user_email=");
  });

  it("a crafted token cannot add query parameters, because it is not in the URL", () => {
    localStorage.setItem("sur.auth", JSON.stringify({ token: "a&admin=1" }));
    const url = wsUrl("p1");
    expect(url).not.toContain("admin");
    expect(url).toBe("ws://localhost:8000/ws/projects/p1");
  });

  it("targets ws:// derived from the http API base", () => {
    expect(wsUrl("p1")).toMatch(/^wss?:\/\//);
    expect(wsUrl("p1")).toContain("/ws/projects/p1");
  });
});
