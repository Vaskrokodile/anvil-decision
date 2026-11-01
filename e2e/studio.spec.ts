import { test, expect } from '@playwright/test';

/**
 * Studio smoke: walk all five steps with a starter dataset. No model weights
 * are downloaded — this verifies wiring, not inference.
 */
test('studio flow: probe → model → dataset → run → configure', async ({ page }) => {
  await page.goto('/#/studio');
  await expect(page.getByText('Studio / job')).toBeVisible();

  // 01 — hardware probe: Continue unlocks once the report resolves
  await expect(page.getByText('Hardware probe')).toBeVisible();
  const cont = page.getByRole('button', { name: 'Continue' });
  await expect(cont).toBeEnabled({ timeout: 15_000 });
  await cont.click();

  // 02 — model roster: pick the smallest always-enabled entry
  await expect(page.getByText('Choose an inference target')).toBeVisible();
  await page.getByRole('button', { name: /SmolLM2 135M/i }).click();
  await page.getByRole('button', { name: 'Continue' }).click();

  // 03 — dataset: load a bundled starter set
  await expect(page.getByText('Build the dataset')).toBeVisible();
  await page
    .locator('button')
    .filter({ hasText: /rows · \d+ classes ·/ })
    .first()
    .click();
  await expect(page.getByText(/rows · \d+ cols/).first()).toBeVisible({ timeout: 20_000 });
  await page.getByRole('button', { name: 'Continue' }).click();

  // 04 — run: engine picker offers jev / decoder / head / ft paths
  await expect(page.getByText('Run it on real text')).toBeVisible();
  await expect(page.getByRole('button', { name: /jev decision model/i })).toBeVisible();
  await expect(page.getByRole('button', { name: /anvil head/i })).toBeVisible();
  await page.getByRole('button', { name: 'Configure training' }).click();

  // 05 — configure: column mapping + train controls render
  await expect(page.getByText('Configure & train')).toBeVisible();
  await expect(page.getByRole('button', { name: /80\/20 split/i })).toBeVisible();
  await expect(page.getByRole('button', { name: /Export job bundle/i })).toBeEnabled();
});
