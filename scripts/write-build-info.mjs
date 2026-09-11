import { execFileSync } from "node:child_process";
import { readFile, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";

const packageRoot = resolve(process.argv[2] || "dist/Desktop2Stereo");
const output = resolve(process.argv[3] || join(packageRoot, "build-info.json"));
const safeSha = /^[0-9a-f]{7,64}$/i;
const safeVersion = /^[A-Za-z0-9][A-Za-z0-9._+-]{0,63}$/;

function gitSha() {
  const configured = String(process.env.D2S_GIT_SHA || process.env.GITHUB_SHA || "").trim();
  if (safeSha.test(configured)) return configured.toLowerCase();
  try {
    const detected = execFileSync("git", ["rev-parse", "HEAD"], { encoding: "utf8" }).trim();
    if (safeSha.test(detected)) return detected.toLowerCase();
  } catch {}
  throw new Error("A valid Git SHA is required for release build metadata.");
}

let appVersion = String(process.env.D2S_APP_VERSION || "").trim();
if (!appVersion) {
  const source = await readFile("src/desktop2stereo/utils/app_info.py", "utf8");
  appVersion = source.match(/^VERSION\s*=\s*[\"']([^\"']+)[\"']/m)?.[1] || "";
}
if (!safeVersion.test(appVersion)) throw new Error("A valid application version is required for release build metadata.");

await writeFile(
  output,
  `${JSON.stringify({ version: 1, product: "desktop2stereo", app_version: appVersion, git_sha: gitSha() }, null, 2)}\n`,
  "utf8",
);
console.log(`Build metadata written: ${output}`);
