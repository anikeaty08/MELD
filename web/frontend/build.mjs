import { mkdirSync, copyFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { build } from "esbuild";

const here = dirname(fileURLToPath(import.meta.url));
const srcDir = resolve(here, "src");
const outDir = resolve(here, "..", "static");

mkdirSync(outDir, { recursive: true });

await build({
  entryPoints: [resolve(srcDir, "main.jsx")],
  bundle: true,
  outfile: resolve(outDir, "app.js"),
  format: "iife",
  target: ["es2020"],
  jsx: "automatic",
  sourcemap: false,
  minify: false,
  loader: {
    ".css": "css",
  },
});

copyFileSync(resolve(srcDir, "index.html"), resolve(outDir, "index.html"));
