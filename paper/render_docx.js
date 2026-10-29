// Render paper/content.json to a Word manuscript.  Usage: node render_docx.js content.json out.docx "<repo url>"
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, WidthType, AlignmentType,
  BorderStyle, ImageRun, ShadingType, LevelFormat, Footer, PageNumber, TabStopType, HeadingLevel,
} = require("docx");

const [, , inFile, outFile, repoUrl] = process.argv;
const C = JSON.parse(fs.readFileSync(inFile, "utf8"));
const FONT = "Times New Roman";
const CM = 567; // DXA per cm
const TEXT_W = 16 * CM;

const run = (text, o = {}) => new TextRun({ text, font: FONT, size: o.size || 22, bold: o.bold, italics: o.italics, highlight: o.highlight });

function runsWithRepo(text, o = {}) {
  if (!text.includes("{REPO}")) return [run(text, o)];
  const [a, b] = text.split("{REPO}");
  return [run(a, o), run(repoUrl, { ...o, highlight: "yellow" }), run(b, o)];
}

const body = (text, o = {}) => new Paragraph({
  children: runsWithRepo(text, o), alignment: o.align || AlignmentType.JUSTIFIED,
  spacing: { after: 120, line: 276 },
});

const children = [];
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 },
  children: [run(C.title, { bold: true, size: 30 })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 }, children: [run(C.authors, { size: 23 })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 }, children: [run(C.affiliation, { size: 19 })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 300 }, children: [run(C.corresponding, { size: 19 })] }));
children.push(new Paragraph({ spacing: { after: 80 }, children: [run("Abstract", { bold: true, size: 24 })] }));
children.push(body(C.abstract));
children.push(new Paragraph({ spacing: { after: 240 }, children: [run("Keywords: ", { bold: true }), run(C.keywords)] }));

const thin = { style: BorderStyle.SINGLE, size: 4, color: "000000" };
const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };

function makeTable(t) {
  const widths = t.widths.map((w) => Math.round(w * CM));
  const total = widths.reduce((a, b) => a + b, 0);
  const cell = (text, i, isHeader, isLast) => new TableCell({
    width: { size: widths[i], type: WidthType.DXA },
    margins: { top: 40, bottom: 40, left: 70, right: 70 },
    shading: isHeader ? { type: ShadingType.CLEAR, fill: "EDEDED", color: "auto" } : undefined,
    borders: { top: isHeader ? thin : none, bottom: isHeader || isLast ? thin : none, left: none, right: none },
    children: [new Paragraph({ spacing: { before: 0, after: 0, line: 240 }, children: [run(String(text), { size: 17, bold: isHeader })],
      alignment: i <= 2 || text === "" ? AlignmentType.LEFT : AlignmentType.CENTER })],
  });
  const rows = [new TableRow({ tableHeader: true, children: t.header.map((h, i) => cell(h, i, true, false)) })];
  t.rows.forEach((r, ri) => rows.push(new TableRow({ cantSplit: true,
    children: r.map((v, i) => cell(v, i, false, ri === t.rows.length - 1)) })));
  return new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: widths, rows,
    alignment: AlignmentType.CENTER });
}

for (const b of C.blocks) {
  if (b.type === "heading") {
    children.push(new Paragraph({ heading: b.level === 1 ? HeadingLevel.HEADING_1 : HeadingLevel.HEADING_2,
      spacing: { before: b.level === 1 ? 280 : 200, after: 100 }, keepNext: true,
      children: [run(b.text, { bold: true, size: b.level === 1 ? 24 : 22 })] }));
  } else if (b.type === "para") {
    // bold lead-in for "X. rest" ablation paragraphs
    const m = b.text.match(/^(Equal-weight FedAvg\.|Misconfigured defences\.)\s(.*)$/s);
    if (m) children.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 120, line: 276 },
      children: [run(m[1] + " ", { italics: true }), run(m[2])] }));
    else children.push(body(b.text));
  } else if (b.type === "bullet") {
    children.push(new Paragraph({ numbering: { reference: "bul", level: 0 }, alignment: AlignmentType.JUSTIFIED,
      spacing: { after: 80, line: 276 }, children: [run(b.text)] }));
  } else if (b.type === "equation") {
    children.push(new Paragraph({ spacing: { before: 80, after: 120 },
      tabStops: [{ type: TabStopType.CENTER, position: TEXT_W / 2 }, { type: TabStopType.RIGHT, position: TEXT_W }],
      children: [run("\t" + b.text + "\t(" + b.num + ")")] }));
  } else if (b.type === "figure") {
    const buf = fs.readFileSync(b.path);
    const w = buf.readUInt32BE(16), h = buf.readUInt32BE(20);
    const wpx = b.width_cm * 37.8;
    children.push(new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 60 },
      children: [new ImageRun({ type: "png", data: buf, transformation: { width: Math.round(wpx), height: Math.round(wpx * h / w) } })] }));
    children.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 200 },
      children: [run(b.caption, { size: 19, italics: false })] }));
  } else if (b.type === "table") {
    children.push(new Paragraph({ keepNext: true, spacing: { before: 160, after: 80 }, alignment: AlignmentType.JUSTIFIED,
      children: [run(b.caption, { size: 19 })] }));
    children.push(makeTable(b));
    children.push(new Paragraph({ spacing: { after: 200 }, children: b.note ? [run(b.note, { size: 17, italics: true })] : [] }));
  }
}

children.push(new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 280, after: 100 },
  children: [run("Declarations", { bold: true, size: 24 })] }));
for (const [k, v] of C.declarations) children.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED,
  spacing: { after: 100 }, children: [run(k + " ", { bold: true }), ...runsWithRepo(v)] }));
children.push(new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 280, after: 100 },
  children: [run("References", { bold: true, size: 24 })] }));
C.references.forEach((r, i) => children.push(new Paragraph({ spacing: { after: 60 },
  indent: { left: 0.9 * CM, hanging: 0.9 * CM }, children: [run(`[${i + 1}]\t${r}`, { size: 19 })],
  tabStops: [{ type: TabStopType.LEFT, position: 0.9 * CM }] })));

const doc = new Document({
  creator: "Abhinav Rajesh Pittan, Rishabh Rajesh", title: C.title,
  styles: { default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 24, bold: true, color: "000000" }, paragraph: { outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 22, bold: true, color: "000000" }, paragraph: { outlineLevel: 1 } } ] },
  numbering: { config: [{ reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•",
    alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 0.8 * CM, hanging: 0.4 * CM } } } }] }] },
  sections: [{
    properties: { page: { margin: { top: 2.5 * CM, bottom: 2.5 * CM, left: 2.5 * CM, right: 2.5 * CM } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 18 })] })] }) },
    children,
  }],
});
Packer.toBuffer(doc).then((b) => { fs.writeFileSync(outFile, b); console.log("wrote", outFile); });
