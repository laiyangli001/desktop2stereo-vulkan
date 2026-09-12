import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";

const outputPath = resolve(process.argv[2] || "");
const inputPaths = process.argv.slice(3).map((path) => resolve(path));
const exactRequirement = /^([A-Za-z0-9][A-Za-z0-9_.-]*)(?:\[[^\]]+\])?===?([^\s;]+)$/;
const directUrlRequirement = /^([A-Za-z0-9][A-Za-z0-9_.-]*)(?:\[[^\]]+\])?\s*@\s*(https?:\/\/\S+)$/;
const sha256Fragment = /^sha256=([a-f0-9]{64})$/i;
const normalizedName = (name) => name.toLowerCase().replace(/[-_.]+/g, "-");

function fail(message) {
  throw new Error(`Release SBOM generation failed: ${message}`);
}

function parseDirectUrl(name, url, source) {
  const parsed = new URL(url);
  const hash = parsed.hash.slice(1);
  const hashMatch = hash.match(sha256Fragment);
  if (!hashMatch) fail(`${source} direct URL must include #sha256=<64 hex characters>`);
  const fileName = decodeURIComponent(parsed.pathname.split("/").pop() || "");
  const wheelMatch = fileName.match(/^[^/]+?-([0-9][^-]*)-(?:[^-]+-){2,}[^-]+\.whl$/i);
  return {
    name,
    normalized: normalizedName(name),
    version: wheelMatch ? wheelMatch[1] : "direct-url",
    url: parsed.toString(),
    sha256: hashMatch[1].toLowerCase(),
    source,
  };
}

async function collectRequirements(path, visited, entries) {
  if (visited.has(path)) return;
  visited.add(path);
  let content;
  try {
    content = await readFile(path, "utf8");
  } catch (error) {
    fail(`cannot read ${path}: ${error.message}`);
  }
  for (const [index, rawLine] of content.split(/\r?\n/).entries()) {
    const lineNumber = index + 1;
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) continue;
    const include = line.match(/^(?:-r|--requirement)\s+(.+)$/);
    if (include) {
      await collectRequirements(resolve(dirname(path), include[1].trim()), visited, entries);
      continue;
    }
    if (line.startsWith("-")) continue;
    const requirement = line.split(";", 1)[0].trim();
    const match = requirement.match(exactRequirement);
    if (match) {
      entries.push({ name: match[1], normalized: normalizedName(match[1]), version: match[2], source: `${path}:${lineNumber}` });
      continue;
    }
    const directUrl = requirement.match(directUrlRequirement);
    if (directUrl) {
      entries.push(parseDirectUrl(directUrl[1], directUrl[2], `${path}:${lineNumber}`));
      continue;
    }
    fail(`${path}:${lineNumber} is not pinned with == or a sha256-pinned direct URL: ${line}`);
  }
}

function serialNumber(components) {
  const digest = createHash("sha256").update(JSON.stringify(components)).digest("hex");
  return `urn:uuid:${digest.slice(0, 8)}-${digest.slice(8, 12)}-${digest.slice(12, 16)}-${digest.slice(16, 20)}-${digest.slice(20, 32)}`;
}

async function main() {
  if (!outputPath || inputPaths.length === 0) fail("usage: node scripts/write-release-sbom.mjs <output.json> <requirements.txt> [...]");
  const entries = [];
  await Promise.all(inputPaths.map((path) => collectRequirements(path, new Set(), entries)));
  const byName = new Map();
  for (const entry of entries) {
    const previous = byName.get(entry.normalized);
    if (previous && (previous.version !== entry.version || previous.sha256 !== entry.sha256)) {
      fail(`dependency ${entry.normalized} has conflicting versions ${previous.version} and ${entry.version}`);
    }
    byName.set(entry.normalized, entry);
  }
  const components = [...byName.values()].sort((left, right) => left.normalized.localeCompare(right.normalized)).map((entry) => ({
    type: "library",
    name: entry.name,
    version: entry.version,
    purl: `pkg:pypi/${entry.normalized}@${entry.version}`,
    ...(entry.url ? {
      externalReferences: [{ type: "distribution", url: entry.url }],
      properties: [{ name: "integrity", value: `sha256:${entry.sha256}` }],
    } : {}),
  }));
  const bom = {
    bomFormat: "CycloneDX",
    specVersion: "1.5",
    serialNumber: serialNumber(components),
    version: 1,
    metadata: {
      component: { type: "application", name: "desktop2stereo", version: process.env.D2S_RELEASE_VERSION || "unknown" },
      tools: [{ vendor: "Desktop2Stereo", name: "write-release-sbom.mjs", version: "1" }],
    },
    components,
  };
  await writeFile(outputPath, `${JSON.stringify(bom, null, 2)}\n`, "utf8");
  console.log(`Release SBOM written: ${outputPath} (${components.length} components)`);
}

main().catch((error) => {
  console.error(error.message);
  process.exitCode = 1;
});
