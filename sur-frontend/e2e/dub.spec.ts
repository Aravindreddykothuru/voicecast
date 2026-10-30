// A real dub driven entirely through the UI, checking that every control
// reaches the backend and that the UI reflects what actually happened.
import { expect, test, type Page } from "@playwright/test";

const VIDEO = process.env.E2E_VIDEO;
const LONG = 60 * 60 * 1000;

async function signUp(page: Page) {
  await page.goto("/");
  await page.getByRole("button", { name: "Sign Up" }).click();
  await page.locator("#ec-name").fill("E2E Tester");
  await page.locator("#ec-su-email").fill(`e2e+${Date.now()}@sur.local`);
  await page.locator("#ec-su-pw").fill("correct-horse-battery");
  await page.getByRole("button", { name: "Create account" }).click();
  await expect(page.getByRole("heading", { name: "Projects Dashboard" })).toBeVisible();
}

test("upload, confirm language, dub, re-voice a line, and export -- all through the UI", async ({ page }) => {
  test.skip(!VIDEO, "set E2E_VIDEO to a source video");
  const apiErrors: string[] = [];
  page.on("response", (r) => {
    if (r.url().includes("/api/") && r.status() >= 400) apiErrors.push(`${r.status()} ${r.request().method()} ${r.url()}`);
  });

  await signUp(page);
  // The header reflects real reachability (it used to say "Core: Online" regardless).
  await expect(page.getByText("API: online")).toBeVisible({ timeout: 45_000 });

  await page.getByRole("button", { name: "+ New Project" }).click();
  await expect(page.getByText("Preserve original pauses")).toHaveCount(0); // removed no-op control
  await page.locator('input[type="file"]').setInputFiles(VIDEO!);
  await page.getByPlaceholder("e.g. Meridian Documentary").fill("E2E UI dub");
  await page.getByRole("button", { name: /^Telugu/ }).click();
  await page.getByRole("button", { name: "Start Processing →" }).click();

  // The run parks at the gate with the backend's own detection.
  await expect(page.getByRole("heading", { name: "Processing" })).toBeVisible();
  await expect(page.getByText("Confirm source language")).toBeVisible({ timeout: LONG });
  await expect(page.getByText(/confidence · \d+\/\d+ spoken segments/)).toBeVisible();
  await expect(page.locator("text=English").first()).toBeVisible();
  await page.getByRole("button", { name: "Continue" }).click();

  await expect(page.getByRole("button", { name: "Open Editor →" })).toBeVisible({ timeout: LONG });
  await expect(page.getByText("✓ Complete")).toHaveCount(7);
  await expect(page.getByText("This run cannot succeed")).toHaveCount(0);

  await page.getByRole("button", { name: "Open Editor →" }).click();
  await expect(page.getByText(/^\d+ segments$/)).toBeVisible();
  const revoice = page.getByRole("button", { name: "Re-voice this line" });
  await expect(revoice).toBeEnabled();
  await revoice.click();
  await expect(page.getByText(/Regenerating|Done: the segment was re-voiced/)).toBeVisible();
  await expect(page.getByText("Done: the segment was re-voiced and the export re-muxed.")).toBeVisible({ timeout: LONG });

  await page.getByRole("button", { name: "Preview" }).click();
  await expect(page.getByText("Every clip fits before the next line (after timing fit).")).toBeVisible();

  await page.getByRole("button", { name: "Export", exact: true }).click();
  await expect(page.getByRole("button", { name: "↓ Download video" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Per-segment QA" })).toBeVisible();

  // The UI must not have made a single request the backend refused.
  expect(apiErrors).toEqual([]);
});
