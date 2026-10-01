/**
 * The capabilities gate must guard the studio, not the front door.
 *
 * The bug: CapabilitiesGate wrapped the whole app, so one failing
 * GET /api/capabilities -- which is what a developer sees every time the
 * backend is not running -- replaced every screen, landing included, with
 * "Cannot reach the backend". The product looked broken rather than offline,
 * and that is what "the page won't open" was.
 *
 * CONTRACTS.md #2 says the UI must never offer what the backend cannot
 * serve. The landing page offers nothing: no language picker, no emotion
 * list, no voice. So it must render with the API down, while every screen
 * that does offer those choices must still block.
 */
import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App from "@/App";

function mockCapabilitiesFailure() {
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    if (url.includes("/api/capabilities")) throw new TypeError("Failed to fetch");
    if (url.includes("/healthz")) throw new TypeError("Failed to fetch");
    throw new TypeError("Failed to fetch");
  }));
}

beforeEach(() => {
  localStorage.clear();
  mockCapabilitiesFailure();
});

afterEach(() => {
  vi.unstubAllGlobals();
  localStorage.clear();
});

describe("capabilities gate scope", () => {
  it("renders the landing page when /api/capabilities cannot be reached", async () => {
    render(<App />);

    // The landing page is the front door: it must survive the backend being
    // down, because it promises nothing the backend has to confirm.
    await waitFor(() => {
      expect(screen.getAllByText(/VOICECAST/i).length).toBeGreaterThan(0);
    });

    // And it must NOT be replaced by the blocking error screen.
    expect(screen.queryByText(/Cannot reach the backend/i)).not.toBeInTheDocument();
  });

  it("still blocks the studio, which does offer languages and emotions", async () => {
    // A signed-in user lands on the dashboard rather than the landing page.
    localStorage.setItem(
      "sur.auth",
      JSON.stringify({ token: "t", user: { email: "dev@sur.local", name: "Dev" } }),
    );

    render(<App />);

    await waitFor(() => {
      expect(screen.getByText(/Cannot reach the backend/i)).toBeInTheDocument();
    });
    // The contract holds where it matters: no guessed list is rendered.
    expect(screen.queryByText(/Hindi/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/Telugu/i)).not.toBeInTheDocument();
  });
});
