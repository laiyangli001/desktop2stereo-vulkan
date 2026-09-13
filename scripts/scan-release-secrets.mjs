import { readdir, readFile, stat } from "node:fs/promises";
import { join, relative, resolve } from "node:path";

const packageRoot = resolve(process.argv[2] || "");
const textExtensions = new Set([
  ".bat", ".bash", ".cfg", ".cpp", ".h", ".ini", ".json", ".md", ".mm", ".pem", ".plist",
  ".py", ".sh", ".toml", ".txt", ".xml", ".yml", ".yaml",
]);
const forbiddenPath = /(^|\/)(?:\.env(?:\..*)?|credentials(?:\..*)?|secrets?(?:\..*)?|.*\.(?:key|pfx|p12|p8))$/i;
const secretPatterns = [
  ["private key material", /-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----/],
  ["AWS access key", /\b(?:AKIA|ASIA)[0-9A-Z]{16}\b/],
  ["GitHub token", /\b(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})\b/],
  ["Slack token", /\bxox[baprs]-[0-9A-Za-z-]{20,}\b/],
  ["OpenAI-style API key", /\bsk-[A-Za-z0-9]{20,}\b/],
  ["embedded basic-auth URL", /https?:\/\/[^/\s:@]+:[^/\s@]+@/i],
  ["embedded JWT", /\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b/],
  [
    "literal credential value",
    /\b(?:api[_-]?key|client[_-]?secret|access[_-]?token|refresh[_-]?token)\b\s*[:=]\s*["'`]([A-Za-z0-9+/_=-]{20,})["'`]/i,
  ],
];

function fail(message) {
  throw new Error(`Release secret scan failed: ${message}`);
}

function relativePath(path) {
  return relative(packageRoot, path).replaceAll("\\", "/");
}

function isBundledRuntimeDependency(name) {
  return /^src\/python3\/(?:lib\/python3\.\d+\/site-packages|Lib\/site-packages)\//i.test(name);
}

function lineNumber(content, offset) {
  return content.slice(0, offset).split(/\r?\n/).length;
}

async function filesUnder(directory) {
  const result = [];
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const path = join(directory, entry.name);
    if (entry.isSymbolicLink()) fail(`symbolic links are not allowed: ${relativePath(path)}`);
    if (entry.isDirectory()) result.push(...await filesUnder(path));
    else if (entry.isFile()) result.push(path);
    else fail(`unsupported package entry: ${relativePath(path)}`);
  }
  return result;
}

async function main() {
  const rootInfo = await stat(packageRoot).catch(() => null);
  if (!rootInfo?.isDirectory()) fail(`package root is missing: ${packageRoot}`);

  for (const path of await filesUnder(packageRoot)) {
    const name = relativePath(path);
    if (forbiddenPath.test(name)) fail(`credential-like file: ${name}`);
    const extension = name.slice(name.lastIndexOf(".")).toLowerCase();
    if (!textExtensions.has(extension) || isBundledRuntimeDependency(name)) continue;
    const content = await readFile(path, "utf8");
    for (const [label, pattern] of secretPatterns) {
      const match = pattern.exec(content);
      if (match) fail(`${label} found in ${name}:${lineNumber(content, match.index)}`);
    }
  }
  console.log(`Release secret scan passed: ${packageRoot}`);
}

main().catch((error) => {
  console.error(error.message);
  process.exitCode = 1;
});
