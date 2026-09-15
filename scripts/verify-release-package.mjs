import { createHash } from "node:crypto";
import { readdir, readFile, stat } from "node:fs/promises";
import { join, relative, resolve } from "node:path";

const packageRoot = resolve(process.argv[2] || "");
const platform = String(process.argv[3] || "").trim().toLowerCase();
const requireProtectedCore = process.argv.includes("--require-protected-core");
const requiredFiles = {
  windows: ["build-info.json", "runtime-manifest.json", "src/Desktop2Stereo.exe", "src/python3/python.exe", "src/desktop2stereo/main.py", "src/desktop2stereo/icon/icon-256x256.ico", "src/desktop2stereo/icon/icon-256x256.png"],
  linux: ["build-info.json", "runtime-manifest.json", "src/Desktop2Stereo", "src/python3/bin/python", "src/desktop2stereo/main.py", "src/desktop2stereo/icon/icon-256x256.ico", "src/desktop2stereo/icon/icon-256x256.png"],
  macos: ["build-info.json", "runtime-manifest.json", "src/Desktop2Stereo-macos", "src/python3/bin/python", "src/desktop2stereo/main.py", "src/desktop2stereo/icon/icon-256x256.ico", "src/desktop2stereo/icon/icon-256x256.png"],
};
const textExtensions = new Set([".bat", ".bash", ".cpp", ".h", ".json", ".md", ".mm", ".pem", ".plist", ".py", ".sh", ".txt", ".yml", ".yaml"]);
const forbiddenPath = /(^|\/)(?:\.env(?:\..*)?|credentials|secrets?|.*\.(?:pfx|p12|p8))$/i;
const privateKeyMarker = /-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----/;
const protectedCoreFiles = {
  windows: [
    "src/desktop2stereo/protected/parallax-core.enc",
    "src/desktop2stereo/protected/native/windows/d2s_protected_core.dll",
  ],
  linux: [
    "src/desktop2stereo/protected/parallax-core.enc",
    "src/desktop2stereo/protected/native/linux/libd2s_protected_core.so",
  ],
  macos: [
    "src/desktop2stereo/protected/parallax-core.enc",
    "src/desktop2stereo/protected/native/macos/libd2s_protected_core.dylib",
  ],
};

function fail(message) {
  throw new Error(`Release package verification failed: ${message}`);
}

function relativePath(path) {
  const value = relative(packageRoot, path).replaceAll("\\", "/");
  if (!value || value.startsWith("../") || value.includes("/../") || /^[A-Za-z]:\//.test(value)) {
    fail(`unsafe package path ${value}`);
  }
  return value;
}

function isBundledRuntimeDependency(name) {
  return /^src\/python3\/(?:lib\/python3\.\d+\/site-packages|Lib\/site-packages)\//i.test(name);
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
  if (!requiredFiles[platform]) fail(`platform must be windows, linux, or macos (got ${platform || "empty"})`);
  const rootInfo = await stat(packageRoot).catch(() => null);
  if (!rootInfo?.isDirectory()) fail(`package root is missing: ${packageRoot}`);

  let runtimeManifest;
  try {
    runtimeManifest = JSON.parse(await readFile(join(packageRoot, "runtime-manifest.json"), "utf8"));
  } catch (error) {
    fail(`cannot read runtime-manifest.json: ${error.message}`);
  }
  if (
    runtimeManifest?.version !== 1 ||
    runtimeManifest.platform !== platform ||
    !/^\d+\.\d+\.\d+$/.test(String(runtimeManifest.python_version || "")) ||
    !/^https?:\/\//i.test(String(runtimeManifest.source_url || "")) ||
    !/^[a-f0-9]{64}$/i.test(String(runtimeManifest.archive_sha256 || ""))
  ) {
    fail("runtime-manifest.json must contain version, matching platform, Python version, source URL, and SHA-256");
  }

  let buildInfo;
  try {
    buildInfo = JSON.parse(await readFile(join(packageRoot, "build-info.json"), "utf8"));
  } catch (error) {
    fail(`cannot read build-info.json: ${error.message}`);
  }
  if (
    buildInfo?.version !== 1 ||
    buildInfo.product !== "desktop2stereo" ||
    !/^[A-Za-z0-9][A-Za-z0-9._+-]{0,63}$/.test(String(buildInfo.app_version || "")) ||
    !/^[a-f0-9]{7,64}$/i.test(String(buildInfo.git_sha || ""))
  ) {
    fail("build-info.json must contain product, application version, and Git SHA");
  }

  const manifestPath = join(packageRoot, "release-manifest.json");
  let manifest;
  try {
    manifest = JSON.parse(await readFile(manifestPath, "utf8"));
  } catch (error) {
    fail(`cannot read release-manifest.json: ${error.message}`);
  }
  if (manifest?.version !== 1 || !Array.isArray(manifest.files) || !String(manifest.commit || "").trim()) {
    fail("manifest must contain version=1, commit, and files");
  }

  const actualFiles = await filesUnder(packageRoot);
  const actualPaths = new Set();
  for (const path of actualFiles) {
    const name = relativePath(path);
    if (name === "release-manifest.json" || name === "release-manifest.json.sig") continue;
    if (forbiddenPath.test(name)) fail(`forbidden credential-like file: ${name}`);
    actualPaths.add(name);
    const extension = name.slice(name.lastIndexOf(".")).toLowerCase();
    if (textExtensions.has(extension) && !isBundledRuntimeDependency(name)) {
      const content = await readFile(path, "utf8");
      if (privateKeyMarker.test(content)) fail(`private key material found in ${name}`);
    }
  }

  const manifestPaths = new Set();
  for (const entry of manifest.files) {
    if (!entry || typeof entry.path !== "string" || manifestPaths.has(entry.path)) fail("manifest contains an invalid or duplicate path");
    if (entry.path === "release-manifest.json" || entry.path === "release-manifest.json.sig" || entry.path.startsWith("../") || entry.path.includes("/../")) {
      fail(`manifest contains an unsafe path: ${entry.path}`);
    }
    manifestPaths.add(entry.path);
    const path = resolve(packageRoot, entry.path);
    if (relativePath(path) !== entry.path) fail(`manifest path escapes package root: ${entry.path}`);
    const info = await stat(path).catch(() => null);
    if (!info?.isFile()) fail(`manifest file is missing: ${entry.path}`);
    const bytes = await readFile(path);
    const digest = createHash("sha256").update(bytes).digest("hex");
    if (entry.size !== bytes.length || entry.sha256 !== digest) fail(`manifest hash or size mismatch: ${entry.path}`);
  }

  for (const name of actualPaths) if (!manifestPaths.has(name)) fail(`file is not covered by manifest: ${name}`);
  for (const name of manifestPaths) if (!actualPaths.has(name)) fail(`manifest covers a missing file: ${name}`);
  for (const name of requiredFiles[platform]) {
    if (!actualPaths.has(name)) fail(`required ${platform} release file is missing: ${name}`);
  }
  if (requireProtectedCore) {
    for (const name of protectedCoreFiles[platform]) {
      if (!actualPaths.has(name)) fail(`required protected core file is missing: ${name}`);
    }
    const encryptedResource = await readFile(join(packageRoot, protectedCoreFiles[platform][0]));
    if (encryptedResource.includes(Buffer.from("PARALLAX_RESOLVER_VERSION"))) {
      fail("protected core resource contains plaintext formula markers");
    }
  }
  console.log(`Release package verification passed: ${platform}, ${manifest.files.length} files`);
}

main().catch((error) => {
  console.error(error.message);
  process.exitCode = 1;
});
