// Rebuilt by update_summary.py using the artifact-tool runtime outside this repo.
import fs from 'node:fs/promises';
import path from 'node:path';
import { SpreadsheetFile, Workbook } from '@oai/artifact-tool';

const [inputPath, outputPath, previewDirectory] = process.argv.slice(2);
const data = JSON.parse(await fs.readFile(inputPath, 'utf8'));
const workbook = Workbook.create();
const comparison = workbook.worksheets.add('Comparison');
const validation = workbook.worksheets.add('Validation');
const methods = workbook.worksheets.add('Methods');
const colors = { ink: '#172B4D', muted: '#526477', header: '#263E5A', pale: '#EEF3F8' };
const missing = value => value === null || value === undefined ? 'n.a.' : value;

function column(index) {
  let value = index + 1, label = '';
  while (value) { label = String.fromCharCode(65 + (value - 1) % 26) + label; value = Math.floor((value - 1) / 26); }
  return label;
}

function table(sheet, title, context, headers, rows, widths, name) {
  sheet.showGridLines = false;
  const end = rows.length + 6;
  const last = column(headers.length - 1);
  const area = sheet.getRange(`A1:${last}${end}`);
  area.format.font = { name: 'Arial', size: 10, color: colors.ink };
  area.format.rowHeight = 24;
  area.format.verticalAlignment = 'center';
  area.format.wrapText = false;
  sheet.getRange('A2').values = [[title]];
  sheet.getRange('A2').format.font = { name: 'Arial', size: 15, bold: true, color: colors.ink };
  sheet.getRange('A2:M2').format.borders = { bottom: { style: 'thin', color: '#B4C4D4' } };
  sheet.getRange('A3').values = [[context]];
  sheet.getRange('A3').format.font = { name: 'Arial', size: 10, italic: true, color: colors.muted };
  sheet.getRange('A4').values = [['Refresh: python results/update_summary.py; local file changes can be monitored with --watch.']];
  sheet.getRange('A4').format.font = { name: 'Arial', size: 10, italic: true, color: colors.muted };
  sheet.getRange(`A6:${last}6`).values = [headers];
  if (rows.length) sheet.getRange(`A7:${last}${end}`).values = rows.map(row => row.map(missing));
  const nativeTable = sheet.tables.add(`A6:${last}${end}`, true, name);
  nativeTable.style = 'TableStyleMedium2';
  nativeTable.showFilterButton = true;
  sheet.getRange(`A6:${last}6`).format = {
    fill: colors.header, font: { name: 'Arial', size: 10, bold: true, color: '#FFFFFF' },
    horizontalAlignment: 'center', verticalAlignment: 'center', wrapText: true,
    rowHeight: 40,
    borders: { insideVertical: { style: 'thin', color: '#FFFFFF' } },
  };
  for (let i = 0; i < widths.length; i++) sheet.getRange(`${column(i)}1:${column(i)}${end}`).format.columnWidthPx = widths[i];
  sheet.freezePanes.freezeRows(6);
  sheet.freezePanes.freezeColumns(2);
  return end;
}

const comparisonRows = data.comparison.map(r => [
  r.plan_id, r.method, r.report, r.draws, r.steps,
  r.test_relative_l2, r.test_extrap_relative_l2,
  r.test_edge_relative_l2, r.test_extrap_edge_relative_l2,
  r.test_gradient_relative_l2, r.test_extrap_gradient_relative_l2,
  r.test_masked_mse, r.test_extrap_masked_mse,
  r.prediction_ms, r.hardware, r.run, r.test_source, r.test_extrap_source,
]);
const comparisonEnd = table(comparison, 'Microstrip methods and results',
  'Mean design errors; lower is better. Test: 881 designs. Extrapolation: 979. Rows use the validation-selected sampler.',
  ['Plan ID', 'Method', 'Report', 'Draws K', 'DDIM steps', 'Test L2', 'Extrap L2', 'Test edge', 'Extrap edge', 'Test gradient', 'Extrap gradient', 'Test mean MSE', 'Extrap mean MSE', 'Prediction ms', 'Hardware', 'Run / task', 'Test source', 'Extrap source'],
  comparisonRows, [80, 220, 190, 65, 75, 100, 100, 100, 100, 110, 110, 120, 120, 110, 205, 420, 700, 700], 'HeldOutComparison');
comparison.tabColor = colors.header;
comparison.getRange(`F7:K${comparisonEnd}`).setNumberFormat('0.00%');
comparison.getRange(`L7:M${comparisonEnd}`).setNumberFormat('0.000E+00');
comparison.getRange(`N7:N${comparisonEnd}`).setNumberFormat('0.000');
comparison.getRange(`D7:N${comparisonEnd}`).format.horizontalAlignment = 'right';

const validationScores = data.scores.filter(r => r.split === 'val');
const validationRows = validationScores.map(r => [r.plan_id, r.method, r.report, r.draws, r.steps, r.designs,
  r.relative_l2, r.edge_relative_l2, r.gradient_relative_l2, r.masked_mse, r.masked_mse_global,
  r.epoch, r.checkpoint, r.masked_mse_global === undefined ? 'n.a.' : (r.selected ? 'yes' : 'no'), r.variance, r.source, r.run]);
const validationEnd = table(validation, 'Validation trials and draw studies',
  'N identifies full validation versus panel studies. Frozen winner uses saved selection.json. Global MSE is the round-two selection metric.',
  ['Plan ID', 'Method / task', 'Report', 'Draws K', 'DDIM steps', 'N designs', 'Relative L2', 'Edge L2', 'Gradient L2', 'Mean design MSE', 'Global pixel MSE', 'Epoch', 'Checkpoint', 'Frozen winner', 'Mean variance', 'Source', 'Run / task'],
  validationRows, [80, 220, 190, 65, 75, 80, 100, 100, 110, 125, 125, 75, 120, 105, 125, 700, 450], 'ValidationResults');
validation.getRange(`G7:I${validationEnd}`).setNumberFormat('0.00%');
validation.getRange(`J7:K${validationEnd}`).setNumberFormat('0.000E+00');
validation.getRange(`O7:O${validationEnd}`).setNumberFormat('0.000E+00');
validation.getRange(`D7:L${validationEnd}`).format.horizontalAlignment = 'right';
validation.getRange(`N7:N${validationEnd}`).conditionalFormats.add('containsText', { text: 'yes', format: { fill: '#DDEDDD', font: { bold: true, color: '#24572A' } } });

const methodColumns = [
  ['plan_id', 'Plan ID', 80], ['method', 'Method / task', 220],
  ['results_available', 'Results available', 270], ['architecture', 'Architecture and conditioning', 350],
  ['learned_modules', 'Learned modules', 110, '#,##0'], ['layer_breakdown', 'Layer / block breakdown', 400],
  ['input_shape', 'Input shape', 440], ['output_shape', 'Output shape', 280],
  ['architecture_profile', 'Architecture profile', 280], ['layer_shapes', 'Stage shapes', 1050],
  ['loss', 'Loss', 250], ['learning_rate', 'Learning rate', 100, '0.0000'],
  ['parameters', 'Parameters', 110, '#,##0'], ['precision', 'Precision', 85], ['seed', 'Seed', 70],
  ['new_train_seconds', 'New training s', 125, '0.000'], ['new_train_hours', 'New training h', 115, '0.000'],
  ['total_train_seconds', 'Total training s', 125, '0.000'], ['total_train_hours', 'Total training h', 115, '0.000'],
  ['best_epoch', 'Best checkpoint epoch', 270], ['stop_reason', 'Stop reason', 150], ['hardware', 'Hardware', 205],
  ['selection', 'Selection protocol', 850], ['source', 'Metrics source', 700], ['config_source', 'Configuration source', 850],
  ['related_plan_ids', 'Related plan IDs', 120], ['architecture_source', 'Architecture source', 1000],
];
const methodColumn = key => column(methodColumns.findIndex(c => c[0] === key));
const methodRows = data.methods.map(m => methodColumns.map(([key]) => m[key]));
const methodEnd = table(methods, 'Methods and recorded training cost',
  'Source-derived layers and shapes; B = batch, P = spatial points. Indexed layer details appear below. Validation-only results are labeled explicitly.',
  methodColumns.map(c => c[1]), methodRows, methodColumns.map(c => c[2]), 'MethodDescriptions');
methods.getRange(`A7:AA${methodEnd}`).format.rowHeight = 48;
for (const key of ['results_available', 'architecture', 'layer_breakdown', 'input_shape', 'output_shape', 'layer_shapes']) {
  const c = methodColumn(key);
  methods.getRange(`${c}7:${c}${methodEnd}`).format.wrapText = true;
}
for (const [key, , , format] of methodColumns) {
  if (format) {
    const c = methodColumn(key);
    methods.getRange(`${c}7:${c}${methodEnd}`).setNumberFormat(format);
    methods.getRange(`${c}7:${c}${methodEnd}`).format.horizontalAlignment = 'right';
  }
}
for (let i = 0; i < data.methods.length; i++) {
  const row = i + 7;
  const m = data.methods[i];
  if (m.new_train_seconds !== null) methods.getRange(`${methodColumn('new_train_hours')}${row}`).formulas = [[`=${methodColumn('new_train_seconds')}${row}/3600`]];
  if (m.total_train_seconds !== null) methods.getRange(`${methodColumn('total_train_hours')}${row}`).formulas = [[`=${methodColumn('total_train_seconds')}${row}/3600`]];
}

for (const sheet of [comparison, validation, methods]) {
  sheet.getRange('A5').values = [['Plan IDs: docs/project-plan.md. Experiment 5 is shared DDIM sampling; related IDs are recorded on Methods.']];
  sheet.getRange('A5').format.font = { name: 'Arial', size: 10, italic: true, color: colors.muted };
}

// Definitions have one home, alongside method inputs rather than repeated footers.
const definitionStart = methodEnd + 3;
methods.getRange(`A${definitionStart}`).values = [['Metric and timing conventions']];
methods.getRange(`A${definitionStart}`).format.font = { name: 'Arial', size: 12, bold: true, color: colors.ink };
for (let i = 0; i < data.definitions.length; i++) {
  methods.getRange(`A${definitionStart + 1 + i}`).values = [[data.definitions[i]]];
}
const planStart = definitionStart + data.definitions.length + 3;
methods.getRange(`A${planStart}:C${planStart}`).values = [['Plan ID', 'Experiment', 'Source']];
const planRows = data.plan_order.map(id => [id, data.plan_experiments[id], 'docs/project-plan.md']);
methods.getRange(`A${planStart + 1}:C${planStart + planRows.length}`).values = planRows;
methods.getRange(`A${planStart}:C${planStart + planRows.length}`).format.font = { name: 'Arial', size: 10, color: colors.ink };
methods.getRange(`B${planStart + 1}:B${planStart + planRows.length}`).format.wrapText = true;
methods.getRange(`A${planStart + 1}:C${planStart + planRows.length}`).format.rowHeight = 55;
methods.getRange(`A${planStart}:C${planStart}`).format.font = { name: 'Arial', size: 10, bold: true, color: colors.ink };

const layerStart = planStart + planRows.length + 4;
methods.getRange(`A${layerStart - 2}`).values = [['Indexed architecture layer shapes']];
methods.getRange(`A${layerStart - 2}`).format.font = { name: 'Arial', size: 12, bold: true, color: colors.ink };
methods.getRange(`A${layerStart - 1}`).values = [['Match the profile to the method row above. Index counts learned modules; fixed and non-neural operations use n.a. Attention projections are grouped.']];
methods.getRange(`A${layerStart - 1}`).format.font = { name: 'Arial', size: 10, italic: true, color: colors.muted };
const layerRows = [];
const layerPreviewRanges = [];
for (const profile of data.architectures) {
  const first = layerStart + 1 + layerRows.length;
  for (const layer of profile.layers) {
    layerRows.push([layer.index, profile.profile, layer.name, layer.operation, layer.input_shape, layer.output_shape, layer.weight_shape]);
  }
  layerPreviewRanges.push([`A${first}:G${Math.min(first + profile.layers.length - 1, first + 8)}`, profile.profile + '.png']);
}
const layerEnd = layerStart + layerRows.length;
methods.getRange(`A${layerStart}:G${layerStart}`).values = [['Layer #', 'Architecture profile', 'Layer / stage', 'Operation', 'Input shape', 'Output shape', 'Weight shape']];
methods.getRange(`A${layerStart + 1}:G${layerEnd}`).values = layerRows.map(row => row.map(missing));
methods.getRange(`A${layerStart}:G${layerEnd}`).format.font = { name: 'Arial', size: 10, color: colors.ink };
methods.getRange(`A${layerStart + 1}:G${layerEnd}`).format.rowHeight = 28;
methods.getRange(`A${layerStart + 1}:G${layerEnd}`).format.verticalAlignment = 'center';
methods.tables.add(`A${layerStart}:G${layerEnd}`, true, 'ArchitectureLayerShapes').style = 'TableStyleMedium2';
methods.getRange(`A${layerStart}:G${layerStart}`).format = { fill: colors.header, font: { name: 'Arial', size: 10, bold: true, color: '#FFFFFF' }, rowHeight: 32, horizontalAlignment: 'center' };

workbook.recalculate();
const inspected = await workbook.inspect({ kind: 'table', range: 'Comparison!A6:H10', include: 'values,formulas', tableMaxRows: 5, tableMaxCols: 8, maxChars: 2200 });
console.log(inspected.ndjson);
const errors = await workbook.inspect({ kind: 'match', searchTerm: '#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#NULL!|#SPILL!|#CALC!', options: { useRegex: true, maxResults: 30 }, maxChars: 1500 });
if (/#REF!|#DIV\/0!|#VALUE!|#NAME\?|#NUM!|#NULL!|#SPILL!|#CALC!/.test(errors.ndjson)) throw new Error('Workbook contains formula errors: ' + errors.ndjson);
for (let i = 0; i < data.methods.length; i++) {
  const row = i + 7;
  if (data.methods[i].new_train_seconds !== null) {
    const value = methods.getRange(`${methodColumn('new_train_hours')}${row}`).values[0][0];
    if (Math.abs(value - data.methods[i].new_train_seconds / 3600) > 1e-10) throw new Error(`Training-hour conversion failed at row ${row}`);
  }
}
if (previewDirectory) {
  await fs.mkdir(previewDirectory, { recursive: true });
  for (const [sheetName, range, filename] of [
    ['Comparison', `A1:N${comparisonEnd}`, 'comparison.png'],
    ['Methods', `A1:F${methodEnd}`, 'methods.png'],
  ]) {
    const preview = await workbook.render({ sheetName, range, scale: 1.5, format: 'png' });
    await fs.writeFile(path.join(previewDirectory, filename), new Uint8Array(await preview.arrayBuffer()));
  }
}
const output = await SpreadsheetFile.exportXlsx(workbook);
// Artifact Tool writes inspection sidecars beside its export. Keep those in
// the temporary authoring workspace, leaving only the workbook in results/.
const exportPath = path.join(path.dirname(inputPath), 'comparison.xlsx');
await output.save(exportPath);
await fs.copyFile(exportPath, outputPath + '.tmp');
await fs.rename(outputPath + '.tmp', outputPath);
console.log(`Saved ${outputPath}`);
