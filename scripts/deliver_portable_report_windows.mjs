#!/usr/bin/env node

import {
  mkdirSync,
  readFileSync,
  renameSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { dirname, join, resolve } from "node:path";
import { pathToFileURL } from "node:url";


function parseArguments(argv) {
  const options = {};
  for (let index = 0; index < argv.length; index += 2) {
    const key = argv[index]?.replace(/^--/, "");
    const value = argv[index + 1];
    if (!key || !value) throw new Error("Expected --plugin-root, --input, and --output.");
    options[key] = value;
  }
  for (const key of ["plugin-root", "input", "output"]) {
    if (!options[key]) throw new Error(`Missing --${key}.`);
  }
  return options;
}


const options = parseArguments(process.argv.slice(2));
const pluginRoot = resolve(options["plugin-root"]);
const inputPath = resolve(options.input);
const outputPath = resolve(options.output);
const scriptsRoot = join(pluginRoot, "skills", "build-report", "scripts");

const { buildPortableArtifact } = await import(
  pathToFileURL(join(scriptsRoot, "build_portable_artifact.mjs")).href
);
const { extractPortableChartSvgs } = await import(
  pathToFileURL(join(scriptsRoot, "extract_portable_chart_svgs.mjs")).href
);
const {
  verifyPortableArtifact,
  verifyPortableArtifactStructure,
} = await import(pathToFileURL(join(scriptsRoot, "verify_portable_artifact.mjs")).href);

const artifact = JSON.parse(readFileSync(inputPath, "utf8"));
const candidatePath = `${outputPath}.candidate.html`;
const screenshotPath = `${outputPath}.verification-failure.png`;
const receiptPath = `${outputPath}.delivery-receipt.json`;

mkdirSync(dirname(outputPath), { recursive: true });
rmSync(candidatePath, { force: true });
rmSync(screenshotPath, { force: true });

try {
  let html = buildPortableArtifact(artifact);
  writeFileSync(candidatePath, html, "utf8");

  const staticCharts = await extractPortableChartSvgs({
    actionTimeoutMs: 2_500,
    htmlPath: candidatePath,
    readyTimeoutMs: 5_000,
  });
  html = buildPortableArtifact(artifact, { staticCharts });

  const deliveryStyles = [
    '<style id="portable-windows-scrollbar-fix">',
    "@media screen{html,body{max-width:100%;overflow-x:clip}}",
    "</style>",
    '<style id="portable-pdf-table-fit">',
    "@media print{.portable-markdown h1,.portable-markdown h2,.portable-markdown h3{break-after:avoid-page!important}.portable-markdown p,.portable-markdown li{orphans:3;widows:3}.portable-table-scroll{width:100%!important;max-width:100%!important;overflow:visible!important}.portable-table-scroll::-webkit-scrollbar{display:none!important}.portable-table-scroll table{width:100%!important;max-width:100%!important;table-layout:fixed!important}.portable-table-scroll thead{display:table-header-group}.portable-table-scroll tr{break-inside:avoid}.portable-table-scroll th,.portable-table-scroll td{height:auto!important;padding:3px 4px!important;overflow:visible!important;font-size:7px!important;line-height:1.2!important;overflow-wrap:anywhere!important;white-space:normal!important;text-overflow:clip!important}.portable-table-note{font-size:7px!important;line-height:1.2!important}}",
    "</style>",
  ].join("");
  if (!html.includes("</head>")) throw new Error("Portable HTML has no closing head tag.");
  html = html.replace("</head>", `${deliveryStyles}</head>`);
  writeFileSync(candidatePath, html, "utf8");

  const structural = verifyPortableArtifactStructure({
    artifactPath: inputPath,
    htmlPath: candidatePath,
  });
  const browser = await verifyPortableArtifact({
    actionTimeoutMs: 2_500,
    artifactPath: inputPath,
    htmlPath: candidatePath,
    readyTimeoutMs: 5_000,
    screenshotPath,
    timeoutMs: 10_000,
  });

  rmSync(outputPath, { force: true });
  renameSync(candidatePath, outputPath);

  const receipt = {
    ok: true,
    html: outputPath,
    artifact: inputPath,
    stages: {
      validation: "passed",
      package: "passed",
      static_chart_extraction: "passed",
      structural_verification: "passed",
      verification: "passed",
    },
    style_fixes: [
      {
        scope: "screen only",
        reason: "Windows Chromium reserves classic scrollbar width while the shared header uses 100vw",
        css: "@media screen{html,body{max-width:100%;overflow-x:clip}}",
      },
      {
        scope: "print only",
        reason: "Wide analytical tables must fit the PDF page without scrollbars or clipped columns",
        css_id: "portable-pdf-table-fit",
      },
    ],
    counts: browser.counts ?? structural.counts,
    sourceDialog: browser.sourceDialog,
    sourceInteraction: browser.sourceInteraction,
    viewports: browser.viewports,
    timings: browser.timings,
  };
  writeFileSync(receiptPath, JSON.stringify(receipt, null, 2), "utf8");
  process.stdout.write(`${JSON.stringify(receipt, null, 2)}\n`);
} catch (error) {
  process.stderr.write(`${error?.stack || error}\n`);
  process.exitCode = 1;
}
