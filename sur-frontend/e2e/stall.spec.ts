// The stranded-run condition, reproduced from the UI: start a run while no
// worker is consuming the queues. It used to spin as "processing" forever.
// Run with the API up and BOTH workers stopped, and a short
// STALL_AFTER_SECONDS on the API (e.g. 45).
import { expect, test } from "@playwright/test";

const VIDEO = process.env.E2E_VIDEO;

test("a run no worker picks up shows a stall error and can be restarted", async ({ page }) => {
  test.skip(!process.env.E2E_STALL || !VIDEO, "set E2E_STALL=1 and E2E_VIDEO, with workers stopped");

  await page.goto("/");
  await page.getByRole("button", { name: "Sign Up" }).click();
  await page.locator("#ec-name").fill("Stall Tester");
  await page.locator("#ec-su-email").fill(`stall+${Date.now()}@sur.local`);
  await page.locator("#ec-su-pw").fill("correct-horse-battery");
  await page.getByRole("button", { name: "Create account" }).click();

  await page.getByRole("button", { name: "+ New Project" }).click();
  await page.locator('input[type="file"]').setInputFiles(VIDEO!);
  await page.getByPlaceholder("e.g. Meridian Documentary").fill("E2E stall");
  await page.getByRole("button", { name: /^Telugu/ }).click();
  await page.getByRole("button", { name: "Start Processing →" }).click();

  await expect(page.getByText("No progress")).toBeVisible({ timeout: 5 * 60 * 1000 });
  await expect(page.getByText(/No worker has picked this run up in \d+ min/)).toBeVisible();

  // Restart is accepted (not a 409) and the run goes back to waiting.
  const restart = page.waitForResponse((r) => r.url().endsWith("/process") && r.request().method() === "POST");
  await page.getByRole("button", { name: "Restart the run" }).click();
  expect((await restart).status()).toBe(200);
  await expect(page.getByText("No progress")).toHaveCount(0, { timeout: 15_000 });

  await page.getByRole("button", { name: "Dashboard" }).click();
  await expect(page.getByText("E2E stall")).toBeVisible();
  await expect(page.getByText("Queued").first()).toBeVisible();
});
