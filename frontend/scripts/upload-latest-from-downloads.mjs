import fs from 'node:fs';
import path from 'node:path';
import puppeteer from 'puppeteer';

const APP_URL = process.env.APP_URL ?? 'https://frontend-ten-henna-52.vercel.app';
const DOWNLOADS_DIR = process.env.DOWNLOADS_DIR ?? '/home/lokesh/Downloads';
const CHROME_PATH = process.env.CHROME_PATH ?? '/usr/bin/google-chrome';
const USER_DATA_DIR = process.env.USER_DATA_DIR ?? '/tmp/ai-video-editor-chrome-default';
const PROFILE_DIR = process.env.PROFILE_DIR ?? 'Default';

function latestVideoFile(dir) {
  const allowed = new Set(['.mp4', '.mov', '.webm', '.mkv']);
  const files = fs.readdirSync(dir)
    .map((name) => path.join(dir, name))
    .filter((file) => {
      try {
        const stat = fs.statSync(file);
        return stat.isFile() && allowed.has(path.extname(file).toLowerCase());
      } catch {
        return false;
      }
    })
    .map((file) => ({ file, mtimeMs: fs.statSync(file).mtimeMs }))
    .sort((a, b) => b.mtimeMs - a.mtimeMs);
  return files[0]?.file ?? null;
}

const videoFile = latestVideoFile(DOWNLOADS_DIR);
if (!videoFile) {
  console.error(`No video file found in ${DOWNLOADS_DIR}`);
  process.exit(1);
}
console.log(`Using video: ${videoFile}`);

const browser = await puppeteer.launch({
  headless: true,
  executablePath: CHROME_PATH,
  userDataDir: USER_DATA_DIR,
  args: ['--no-sandbox', '--disable-setuid-sandbox', `--profile-directory=${PROFILE_DIR}`],
});

try {
  const page = await browser.newPage();
  page.on('console', (msg) => console.log(`[browser:${msg.type()}] ${msg.text()}`));
  page.on('pageerror', (err) => console.error(`[pageerror] ${err.message}`));

  await page.goto(`${APP_URL}/`, { waitUntil: 'networkidle2', timeout: 120000 });
  await page.waitForSelector('input[type="file"]', { timeout: 30000 });
  const input = await page.$('input[type="file"]');
  if (!input) throw new Error('Upload input not found');
  await input.uploadFile(videoFile);

  await page.waitForNavigation({ waitUntil: 'networkidle2', timeout: 60000 }).catch(() => {});
  await new Promise((resolve) => setTimeout(resolve, 5000));

  console.log('Current URL:', page.url());
  const body = await page.evaluate(() => document.body.innerText.slice(0, 600));
  console.log('Body preview:', body);
} finally {
  await browser.close();
}
