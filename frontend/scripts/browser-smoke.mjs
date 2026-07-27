import puppeteer from "puppeteer";

const targetUrl = process.env.BROWSER_URL ?? "https://frontend-ten-henna-52.vercel.app";
const headless = (process.env.HEADLESS ?? "true").toLowerCase() !== "false";

const browser = await puppeteer.launch({
  headless,
  args: ["--no-sandbox", "--disable-setuid-sandbox"],
});

try {
  const page = await browser.newPage();
  page.on("console", (msg) => console.log(`[browser:${msg.type()}] ${msg.text()}`));
  page.on("pageerror", (err) => console.error(`[pageerror] ${err.message}`));

  await page.goto(targetUrl, { waitUntil: "networkidle2", timeout: 120000 });
  await new Promise((resolve) => setTimeout(resolve, 3000));

  const title = await page.title();
  console.log(`Loaded: ${targetUrl}`);
  console.log(`Title: ${title}`);
} finally {
  await browser.close();
}
