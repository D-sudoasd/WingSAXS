#!/usr/bin/env node
'use strict';

// Export the JAC manuscript Markdown to an editable Word document.
// Usage: node scripts/export_jac_manuscript.cjs [input.md] [output.docx]
// The exporter uses docx-js and JSZip from local node_modules, NODE_PATH, or
// the bundled Codex Node runtime next to process.execPath.

const fs = require('node:fs');
const path = require('node:path');

function loadPackage(name) {
  const roots = [
    path.resolve(__dirname, '..', 'node_modules'),
    ...(process.env.NODE_PATH || '').split(path.delimiter).filter(Boolean),
    path.resolve(path.dirname(process.execPath), '..', 'node_modules'),
  ];
  for (const root of [...new Set(roots)]) {
    try {
      return require(path.join(root, name));
    } catch (error) {
      if (error.code !== 'MODULE_NOT_FOUND' && error.code !== 'ERR_PACKAGE_PATH_NOT_EXPORTED') {
        throw error;
      }
    }
  }
  try {
    return require(name);
  } catch (error) {
    throw new Error(`Cannot load ${name}. Install it in node_modules or set NODE_PATH to its package directory.`, { cause: error });
  }
}

const docx = loadPackage('docx');
const JSZip = loadPackage('jszip');
const {
  AlignmentType,
  BorderStyle,
  Document,
  ExternalHyperlink,
  HeadingLevel,
  ImageRun,
  LevelFormat,
  Packer,
  Paragraph,
  ShadingType,
  Table,
  TableCell,
  TableRow,
  TextRun,
  WidthType,
} = docx;

const MATH_NS = 'http://schemas.openxmlformats.org/officeDocument/2006/math';
const inputPath = path.resolve(process.argv[2] || path.join(__dirname, '..', 'docs', 'jac', 'manuscript.md'));
const outputPath = path.resolve(process.argv[3] || path.join(path.dirname(inputPath), 'manuscript.docx'));
const markdown = fs.readFileSync(inputPath, 'utf8').replace(/^\uFEFF/, '');
const mathFragments = new Map();
const unsupportedMathCommands = new Set();
let nextMathId = 1;

function xmlEscape(value) {
  return String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function mathRun(value, style = null) {
  const styleXml = style ? `<m:rPr><m:sty m:val="${style}"/></m:rPr>` : '';
  return `<m:r>${styleXml}<m:t xml:space="preserve">${xmlEscape(value)}</m:t></m:r>`;
}

function mathArgument(tokens, state) {
  while (/^\s+$/.test(tokens[state.index] || '')) state.index += 1;
  if (tokens[state.index] === '{') {
    state.index += 1;
    const content = mathSequence(tokens, state, true).join('');
    if (tokens[state.index] === '}') state.index += 1;
    return content || mathRun(' ');
  }
  if (state.index >= tokens.length) return mathRun(' ');
  return mathAtom(tokens, state);
}

function mathCommand(command, tokens, state) {
  const symbols = {
    approx: '≈', cdot: '·', chi: 'χ', delta: 'δ', epsilon: 'ε', lambda: 'λ', nabla: '∇',
    partial: '∂', pi: 'π', prime: '′', simeq: '≃', vartheta: 'ϑ',
  };
  if (command === 'frac') {
    const numerator = mathArgument(tokens, state);
    const denominator = mathArgument(tokens, state);
    return `<m:f><m:fPr/><m:num>${numerator}</m:num><m:den>${denominator}</m:den></m:f>`;
  }
  if (command === 'mathbf' || command === 'mathrm' || command === 'text') {
    const argument = mathArgument(tokens, state);
    if (command === 'mathbf') return argument.replace(/<m:r>/g, '<m:r><m:rPr><m:sty m:val="b"/></m:rPr>');
    if (command === 'mathrm') return argument.replace(/<m:r>/g, '<m:r><m:rPr><m:sty m:val="p"/></m:rPr>');
    return argument;
  }
  if (command === 'left' || command === 'right') return '';
  if (command === 'quad' || command === 'qquad' || command === ',') return mathRun(' ');
  if (command === 'sin' || command === 'cos' || command === 'tan' || command === 'log') {
    return mathRun(command, 'p');
  }
  if (symbols[command]) return mathRun(symbols[command]);
  if (command === '{' || command === '}') return mathRun(command);
  // Preserve an unfamiliar LaTeX command visibly instead of dropping it.
  unsupportedMathCommands.add(command);
  return mathRun(`\\${command}`);
}

function mathAtom(tokens, state) {
  const token = tokens[state.index++];
  if (token === '{') {
    const content = mathSequence(tokens, state, true).join('');
    if (tokens[state.index] === '}') state.index += 1;
    return content;
  }
  if (token && token.startsWith('\\')) {
    return mathCommand(token.slice(1), tokens, state);
  }
  if (token === "'") return mathRun('′');
  return mathRun(token === '~' ? ' ' : token);
}

function mathSequence(tokens, state, stopAtBrace = false) {
  const nodes = [];
  while (state.index < tokens.length) {
    const token = tokens[state.index];
    if (token === '}' && stopAtBrace) break;
    if (/^\s+$/.test(token)) {
      state.index += 1;
      continue;
    }
    if (token === '^' || token === '_') {
      state.index += 1;
      const argument = mathArgument(tokens, state);
      const base = nodes.pop();
      if (!base) {
        nodes.push(mathRun(token));
        continue;
      }
      if (token === '^') nodes.push(`<m:sSup><m:sSupPr/><m:e>${base}</m:e><m:sup>${argument}</m:sup></m:sSup>`);
      else nodes.push(`<m:sSub><m:sSubPr/><m:e>${base}</m:e><m:sub>${argument}</m:sub></m:sSub>`);
      continue;
    }
    nodes.push(mathAtom(tokens, state));
  }
  return nodes;
}

function latexToOmml(latex) {
  const tokens = latex.match(/\\[A-Za-z]+|\\.|[{}^_]|[^\s]/g) || [];
  const state = { index: 0 };
  const body = mathSequence(tokens, state).join('');
  return `<m:oMath>${body}</m:oMath>`;
}

function mathPlaceholder(latex) {
  const placeholder = `__JAC_MATH_${String(nextMathId++).padStart(4, '0')}__`;
  mathFragments.set(placeholder, latexToOmml(latex.trim()));
  return new TextRun(placeholder);
}

function plainText(value) {
  return value.replace(/\\([\\`*_{}\[\]()#+.!-])/g, '$1');
}

function parseInline(text) {
  const children = [];
  const tokenPattern = /\\\(([\s\S]*?)\\\)|\[((?:\\.|[^\]])+)\]\((https?:\/\/[^)\s]+)\)|\*\*([\s\S]+?)\*\*|\*([^*]+)\*|`([^`]+)`/g;
  let cursor = 0;
  let match;
  while ((match = tokenPattern.exec(text)) !== null) {
    if (match.index > cursor) children.push(new TextRun(plainText(text.slice(cursor, match.index))));
    if (match[1] !== undefined) {
      children.push(mathPlaceholder(match[1]));
    } else if (match[2] !== undefined) {
      children.push(new ExternalHyperlink({
        link: match[3],
        children: [new TextRun({ text: plainText(match[2]), style: 'Hyperlink' })],
      }));
    } else if (match[4] !== undefined) {
      children.push(new TextRun({ text: plainText(match[4]), bold: true }));
    } else if (match[5] !== undefined) {
      children.push(new TextRun({ text: plainText(match[5]), italics: true }));
    } else if (match[6] !== undefined) {
      children.push(new TextRun({ text: match[6], font: 'Consolas', size: 20 }));
    }
    cursor = tokenPattern.lastIndex;
  }
  if (cursor < text.length) children.push(new TextRun(plainText(text.slice(cursor))));
  return children.length ? children : [new TextRun('')];
}

function isBlockStart(line) {
  const trimmed = line.trim();
  return /^(#{1,6})\s+/.test(trimmed)
    || /^!\[[^\]]*\]\([^)]+\)$/.test(trimmed)
    || trimmed === '\\['
    || /^>/.test(trimmed)
    || /^[-*+]\s+/.test(trimmed)
    || /^\d+[.)]\s+/.test(trimmed)
    || /^\|.*\|$/.test(trimmed)
    || /^---+$/.test(trimmed);
}

const contentWidth = 9026;
function imageParagraph(altText, source) {
  const decodedPath = decodeURIComponent(source);
  const imagePath = path.resolve(path.dirname(inputPath), decodedPath);
  if (!fs.existsSync(imagePath) || !fs.statSync(imagePath).isFile()) {
    throw new Error(`Markdown image does not exist: ${imagePath}`);
  }
  const data = fs.readFileSync(imagePath);
  const pngSignature = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
  if (data.length < 24 || !data.subarray(0, 8).equals(pngSignature) || data.toString('ascii', 12, 16) !== 'IHDR') {
    throw new Error(`Markdown image must be a valid PNG: ${imagePath}`);
  }
  const sourceWidth = data.readUInt32BE(16);
  const sourceHeight = data.readUInt32BE(20);
  if (!sourceWidth || !sourceHeight) throw new Error(`PNG has invalid dimensions: ${imagePath}`);

  // 9026 twips is the A4 content width at 1-inch side margins. Keep figures
  // within that width and 7 inches of height without changing their ratio.
  const maxWidthPx = (contentWidth / 1440) * 96;
  const maxHeightPx = 7 * 96;
  const scale = Math.min(maxWidthPx / sourceWidth, maxHeightPx / sourceHeight, 1);
  const width = Math.max(1, Math.round(sourceWidth * scale));
  const height = Math.max(1, Math.round(sourceHeight * scale));
  const description = altText || path.basename(imagePath);
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    keepNext: true,
    spacing: { before: 120, after: 120, line: 300 },
    children: [new ImageRun({
      type: 'png',
      data,
      transformation: { width, height },
      altText: { title: description, description, name: path.basename(imagePath) },
    })],
  });
}

function tableFromRows(rows) {
  const columnCount = Math.max(...rows.map((row) => row.length));
  const baseWidth = Math.floor(contentWidth / columnCount);
  const widths = Array.from({ length: columnCount }, (_, index) => (
    index === columnCount - 1 ? contentWidth - baseWidth * (columnCount - 1) : baseWidth
  ));
  const edge = { style: BorderStyle.SINGLE, size: 4, color: 'D1D5DB' };
  const borders = { top: edge, bottom: edge, left: edge, right: edge, insideHorizontal: edge, insideVertical: edge };
  return new Table({
    width: { size: contentWidth, type: WidthType.DXA },
    columnWidths: widths,
    rows: rows.map((row, rowIndex) => new TableRow({
      tableHeader: rowIndex === 0,
      children: Array.from({ length: columnCount }, (_, colIndex) => new TableCell({
        width: { size: widths[colIndex], type: WidthType.DXA },
        borders,
        margins: { top: 90, bottom: 90, left: 120, right: 120 },
        ...(rowIndex === 0 ? { shading: { fill: 'EAF2F8', type: ShadingType.CLEAR } } : {}),
        children: [new Paragraph({
          children: parseInline(row[colIndex] || ''),
          spacing: { before: 0, after: 0, line: 300 },
        })],
      })),
    })),
  });
}

function parseMarkdown(source) {
  const lines = source.replace(/\r\n?/g, '\n').split('\n');
  const elements = [];
  let index = 0;

  while (index < lines.length) {
    const line = lines[index];
    const trimmed = line.trim();
    if (!trimmed || /^---+$/.test(trimmed)) {
      index += 1;
      continue;
    }

    const heading = trimmed.match(/^(#{1,6})\s+(.+)$/);
    if (heading) {
      const level = heading[1].length;
      if (level === 1 && elements.length === 0) {
        elements.push(new Paragraph({ style: 'Title', alignment: AlignmentType.CENTER, children: parseInline(heading[2]) }));
      } else {
        const headingMap = {
          1: HeadingLevel.HEADING_1,
          2: HeadingLevel.HEADING_1,
          3: HeadingLevel.HEADING_2,
          4: HeadingLevel.HEADING_3,
          5: HeadingLevel.HEADING_3,
          6: HeadingLevel.HEADING_3,
        };
        elements.push(new Paragraph({ heading: headingMap[level], children: parseInline(heading[2]) }));
      }
      index += 1;
      continue;
    }

    const image = trimmed.match(/^!\[([^\]]*)\]\(([^)]+)\)$/);
    if (image) {
      elements.push(imageParagraph(plainText(image[1]), image[2]));
      index += 1;
      continue;
    }

    if (trimmed === '\\[') {
      index += 1;
      const equationLines = [];
      while (index < lines.length && lines[index].trim() !== '\\]') equationLines.push(lines[index++].trim());
      if (index < lines.length) index += 1;
      const latex = equationLines.join(' ');
      elements.push(new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 120, after: 120, line: 300 },
        children: [mathPlaceholder(latex)],
      }));
      continue;
    }

    if (/^>/.test(trimmed)) {
      const quoteLines = [];
      while (index < lines.length && /^\s*>/.test(lines[index])) quoteLines.push(lines[index++].replace(/^\s*>\s?/, '').trim());
      elements.push(new Paragraph({ style: 'Quote', children: parseInline(quoteLines.join(' ')) }));
      continue;
    }

    if (/^\|.*\|$/.test(trimmed)) {
      const tableLines = [];
      while (index < lines.length && /^\s*\|.*\|\s*$/.test(lines[index])) tableLines.push(lines[index++].trim());
      const rows = tableLines
        .filter((row) => !/^\|?\s*:?-{3,}/.test(row))
        .map((row) => row.replace(/^\||\|$/g, '').split('|').map((cell) => cell.trim()));
      if (rows.length) elements.push(tableFromRows(rows));
      continue;
    }

    const unordered = trimmed.match(/^[-*+]\s+(.+)$/);
    const ordered = trimmed.match(/^\d+[.)]\s+(.+)$/);
    if (unordered || ordered) {
      const isOrdered = Boolean(ordered);
      const reference = isOrdered ? 'jacNumbers' : 'jacBullets';
      while (index < lines.length) {
        const item = lines[index].trim().match(isOrdered ? /^\d+[.)]\s+(.+)$/ : /^[-*+]\s+(.+)$/);
        if (!item) break;
        elements.push(new Paragraph({
          numbering: { reference, level: 0 },
          children: parseInline(item[1]),
        }));
        index += 1;
      }
      continue;
    }

    const paragraphLines = [trimmed];
    index += 1;
    while (index < lines.length && lines[index].trim() && !isBlockStart(lines[index])) {
      paragraphLines.push(lines[index].trim());
      index += 1;
    }
    elements.push(new Paragraph({ children: parseInline(paragraphLines.join(' ')) }));
  }
  return elements;
}

async function main() {
  const children = parseMarkdown(markdown);
  if (unsupportedMathCommands.size) {
    const commands = [...unsupportedMathCommands].sort().map((command) => `\\${command}`).join(', ');
    throw new Error(`Add an Office Math mapping for these manuscript commands before export: ${commands}`);
  }
  const document = new Document({
    creator: 'WingSAXS authors',
    title: 'WingSAXS: measuring butterfly trajectories in small-angle X-ray scattering sequences',
    subject: 'Working draft for Journal of Applied Crystallography, Computer Programs',
    description: `Generated from ${path.basename(inputPath)} using scripts/export_jac_manuscript.cjs`,
    styles: {
      default: {
        document: { run: { font: 'Arial', size: 22, color: '202124' } },
        paragraph: { spacing: { before: 0, after: 140, line: 360 } },
      },
      paragraphStyles: [
        { id: 'Title', name: 'Title', basedOn: 'Normal', next: 'Normal', quickFormat: true,
          run: { font: 'Arial', size: 34, bold: true, color: '111827' },
          paragraph: { alignment: AlignmentType.CENTER, spacing: { before: 0, after: 300, line: 360 }, keepNext: true } },
        { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true,
          run: { font: 'Arial', size: 28, bold: true, color: '111827' },
          paragraph: { spacing: { before: 300, after: 140, line: 300 }, outlineLevel: 0, keepNext: true } },
        { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true,
          run: { font: 'Arial', size: 24, bold: true, color: '1F2937' },
          paragraph: { spacing: { before: 240, after: 100, line: 300 }, outlineLevel: 1, keepNext: true } },
        { id: 'Heading3', name: 'Heading 3', basedOn: 'Normal', next: 'Normal', quickFormat: true,
          run: { font: 'Arial', size: 22, bold: true, color: '1F2937' },
          paragraph: { spacing: { before: 180, after: 80, line: 300 }, outlineLevel: 2, keepNext: true } },
        { id: 'Quote', name: 'Quote', basedOn: 'Normal', next: 'Normal',
          run: { italics: true, color: '4B5563' },
          paragraph: { indent: { left: 480 }, spacing: { before: 100, after: 160, line: 360 } } },
      ],
    },
    numbering: {
      config: [
        { reference: 'jacBullets', levels: [{ level: 0, format: LevelFormat.BULLET, text: '•', alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
        { reference: 'jacNumbers', levels: [{ level: 0, format: LevelFormat.DECIMAL, text: '%1.', alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      ],
    },
    sections: [{
      properties: {
        page: {
          size: { width: 11906, height: 16838 },
          margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
        },
      },
      children,
    }],
  });

  const buffer = await Packer.toBuffer(document);
  const archive = await JSZip.loadAsync(buffer);
  const part = archive.file('word/document.xml');
  if (!part) throw new Error('Generated Word package is missing word/document.xml.');
  let documentXml = await part.async('string');
  let replacements = 0;
  for (const [placeholder, omml] of mathFragments) {
    const escaped = placeholder.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    // Office Math is a paragraph-level run-content element. Replace the whole
    // placeholder w:r, rather than nesting m:oMath inside a normal text run.
    const pattern = new RegExp(`<w:r(?:\\s+[^>]*)?>(?:<w:rPr>[\\s\\S]*?<\\/w:rPr>)?<w:t(?:\\s+[^>]*)?>${escaped}<\\/w:t><\\/w:r>`);
    if (!pattern.test(documentXml)) throw new Error(`Could not locate equation placeholder ${placeholder} in generated XML.`);
    documentXml = documentXml.replace(pattern, omml);
    replacements += 1;
  }
  if (replacements !== mathFragments.size) throw new Error('Not all equation placeholders were replaced.');
  if (!documentXml.includes('xmlns:m=')) {
    documentXml = documentXml.replace('<w:document', `<w:document xmlns:m="${MATH_NS}"`);
  }
  archive.file('word/document.xml', documentXml);

  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  const output = await archive.generateAsync({
    type: 'nodebuffer',
    compression: 'DEFLATE',
    compressionOptions: { level: 9 },
  });
  fs.writeFileSync(outputPath, output);
  console.log(`Created ${outputPath}`);
  console.log(`Paragraphs/elements: ${children.length}; editable equations: ${mathFragments.size}`);
}

main().catch((error) => {
  console.error(error.stack || error.message || String(error));
  process.exitCode = 1;
});
